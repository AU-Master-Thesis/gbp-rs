use std::collections::HashMap;

use crate::utils::Indent;
use crate::utils::PrettyPrint;

use super::factorgraph::FactorGraphNode;
use super::factorgraph::FactorMessages;
use super::{
    factorgraph::{FactorIndex, NodeIndex, VariableIndex},
    message::{Eta, Lam, Message, Mu},
};
use gbp_linalg::{pretty_print_matrix, Float, Matrix, Vector};
// use gbp_multivariate_normal::dummy_normal::DummyNormal;
use gbp_multivariate_normal::MultivariateNormal;
use ndarray_inverse::Inverse;
use tap::Tap;

pub type Inbox = HashMap<FactorIndex, Message>;

/// A variable in the factor graph.
#[derive(Debug, Clone)]
pub struct Variable {
    /// In **gbpplanner** the `prior` is stored in 2 separate variables:
    /// 1. `eta_prior_` Information vector of prior on variable (essentially like a unary factor)
    /// 2. `lam_prior_` Precision matrix of prior on variable (essentially like a unary factor)
    // pub prior: MultivariateNormal,
    // pub belief: MultivariateNormal,
    /// Degrees of freedom. For 2D case n_dofs_ = 4 ([x,y,xdot,ydot])
    pub dofs: usize,

    pub eta_prior: Vector<Float>,
    pub lam_prior: Matrix<Float>,
    pub eta: Vector<Float>,
    pub lam: Matrix<Float>,
    pub mu: Vector<Float>,
    pub sigma: Matrix<Float>,

    /// Flag to indicate if the variable's covariance is finite, i.e. it does not contain NaNs or Infs
    /// In gbpplanner it is used to control if a variable can be rendered.
    // pub valid: bool,
    /// Mailbox for incoming message storage
    inbox: Inbox,
}

impl Variable {
    // pub fn new(prior: MultivariateNormal, dofs: usize) -> UninsertedVariable {
    //     UninsertedVariable { prior, dofs }
    //     // Self {
    //     //     node_index: None,
    //     //     prior: prior.clone(),
    //     //     belief: prior,
    //     //     dofs,
    //     //     inbox: Inbox::new(),
    //     // }
    // }

    #[must_use]
    pub fn new(mu_prior: Vector<Float>, mut lam_prior: Matrix<Float>, dofs: usize) -> Self {
        // if (!lam_prior_.allFinite()) lam_prior_.setZero();
        // if !prior.precision_matrix().iter().all(|x| x.is_finite()) {
        //     prior.precision_matrix().fill(0.0);
        // }
        if !lam_prior.iter().all(|x| x.is_finite()) {
            lam_prior.fill(0.0);
        }

        // pretty_print_matrix!(&lam_prior);

        let eta_prior = lam_prior.dot(&mu_prior);
        let sigma = lam_prior
            .inv()
            .unwrap_or_else(|| Matrix::<Float>::zeros((dofs, dofs)));
        let eta = eta_prior.clone();
        let lam = lam_prior.clone();

        Self {
            dofs,
            eta_prior,
            lam_prior,
            eta,
            lam,
            mu: mu_prior,
            sigma,
            inbox: Inbox::new(),
        }

        //
        // Self {
        //     prior: prior.clone(),
        //     belief: prior,
        //     dofs,
        //     inbox: Inbox::new(),
        // }
    }

    // pub fn new(mut mu_prior: Vector<Float>, mut

    //
    // pub fn set_node_index(&mut self, node_index: NodeIndex) {
    //     match self.node_index {
    //         Some(_) => panic!("The node index is already set"),
    //         None => self.node_index = Some(node_index),
    //     }
    // }
    //
    // pub fn get_node_index(&self) -> NodeIndex {
    //     match self.node_index {
    //         Some(node_index) => node_index,
    //         None => panic!("The node index has not been set"),
    //     }
    // }

    pub fn send_message(&mut self, from: FactorIndex, message: Message) {
        let _ = self.inbox.insert(from, message);
    }

    // TODO: why never used?
    pub fn read_message_from(&mut self, from: FactorIndex) -> Option<&Message> {
        self.inbox.get(&from)
    }

    /// Change the prior of the variable.
    /// It updates the belief of the variable.
    /// The prior acts as the pose factor
    /// Called `Variable::change_variable_prior` in **gbpplanner**
    pub fn change_prior(
        &mut self,
        mean: &Vector<Float>,
        indices_of_adjacent_factors: Vec<FactorIndex>,
        // ) -> HashMap<NodeIndex, Message> {
    ) -> FactorMessages {
        self.eta_prior = self.lam_prior.dot(mean);
        self.mu = mean.clone();

        // self.prior
        //     .update_information_vector(&self.prior.precision_matrix().dot(mean));
        // self.prior.information_vector = self.prior.precision_matrix.dotTHIS.(&mean);
        // QUESTION: why cache mu?
        // mu_ = new_mu;
        // belief_ = Message {eta_, lam_, mu_};
        // FIXME: we probably never update the belief of the variable
        // dbg!(&self.belief);

        // NOTE: this IS DIFFERENT FROM gbpplanner

        // self.belief = self.prior.clone();

        indices_of_adjacent_factors
            .into_iter()
            .map(|factor_index| {
                let message = Message::new(
                    Eta(self.eta.clone()),
                    Lam(self.lam.clone()),
                    Mu(self.mu.clone()),
                );
                (factor_index, message)
            })
            .collect()
    }

    // /***********************************************************************************************************/
    // // Variable belief update step:
    // // Aggregates all the messages from its connected factors (begins with the prior, as this is effectively a unary factor)
    // // The valid_ flag is useful for drawing the variable.
    // // Finally the outgoing messages to factors is created.
    // /***********************************************************************************************************/
    /// Variable Belief Update step (Step 1 in the GBP algorithm)
    /// called `Variable::update_belief` in **gbpplanner**
    pub fn update_belief_and_create_responses(&mut self) -> FactorMessages {
        // Collect messages from all other factors, begin by "collecting message from pose factor prior"
        self.eta = self.eta_prior.clone();
        self.lam = self.lam_prior.clone();

        // // TODO: wrap in unsafe block for perf:
        // unsafe {
        //     self.belief
        //         .set_information_vector(self.prior.information_vector());
        //     self.belief
        //         .set_precision_matrix(self.prior.precision_matrix());
        // }

        // Go through received messages and update belief
        for (_, message) in self.inbox.iter() {
            let Some(payload) = message.payload() else {
                // empty message
                // println!("skipping empty message");
                continue;
            };
            self.eta = &self.eta + &payload.eta;
            self.lam = &self.lam + &payload.lam;
            // if message.is_empty() {
            //     continue;
            // }
            // unsafe {
            //     // self.belief.add_assign_information_vector(&message.information_vector());
            //     self.belief
            //         .add_assign_information_vector(normal.information_vector());
            //     // self.belief.add_assign_precision_matrix(&message.precision_matrix());
            //     self.belief
            //         .add_assign_precision_matrix(normal.precision_matrix());
            // }
        }

        // pretty_print_matrix!(&self.lam);
        // Update belief
        self.sigma = self
            .lam
            // .tap(|it| pretty_print_matrix!(it))
            .inv()
            .unwrap_or_else(|| Matrix::<Float>::zeros((self.dofs, self.dofs)));
        let valid = self.sigma.iter().all(|x| x.is_finite());
        if valid {
            self.mu = self.sigma.dot(&self.eta);
        }

        // Update the internal invariant of the belief, needed after the previous call to {add_assign,set}_{information_vector,precision_matrix}
        // which violates that the mean, information vector and precision matrix are consistent, for performance reasons.
        // self.belief.update();

        // let valid = self.belief.covariance().iter().all(|x| x.is_finite());
        // if valid {
        // TODO: is this meaningful?
        // if (valid_) mu_ = sigma_ * eta_;
        // }
        // }

        self.inbox
            .iter()
            .map(|(&factor_index, received_message)| {
                let response = received_message.payload().map_or_else(
                    || {
                        Message::new(
                            Eta(self.eta.clone()),
                            Lam(self.lam.clone()),
                            Mu(self.mu.clone()),
                        )
                    },
                    |gaussian| {
                        Message::new(
                            Eta(&self.eta - &gaussian.eta),
                            Lam(&self.lam - &gaussian.lam),
                            Mu(&self.mu - &gaussian.mu),
                        )
                    },
                );
                (factor_index, response)
            })
            .collect()
    }
}
