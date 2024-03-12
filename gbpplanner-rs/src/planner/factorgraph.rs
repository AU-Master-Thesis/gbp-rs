use std::collections::HashMap;
use std::ops::{AddAssign, Range};

use bevy::prelude::*;
use gbp_linalg::{pretty_print_matrix, Float, Matrix, Vector};
use ndarray::s;
use num_traits::Zero;
use petgraph::Undirected;

use super::factor::{Factor, FactorKind};
use super::marginalise_factor_distance::marginalise_factor_distance;
use super::message::Message;
use super::robot::RobotId;
use super::variable::Variable;

pub mod graphviz {
    use crate::planner::factor::InterRobotConnection;

    pub struct Node {
        pub index: usize,
        pub kind: NodeKind,
    }

    impl Node {
        pub fn color(&self) -> &'static str {
            self.kind.color()
        }

        pub fn shape(&self) -> &'static str {
            self.kind.shape()
        }

        pub fn width(&self) -> &'static str {
            self.kind.width()
        }
    }

    pub enum NodeKind {
        Variable { x: f32, y: f32 },
        InterRobotFactor(InterRobotConnection),
        // InterRobotFactor {
        //     /// The id of the robot the interrobot factor is connected to
        //     other_robot_id: RobotId,
        //     /// The index of the variable in the other robots factorgraph, that the interrobot factor is connected with
        //     variable_index_in_other_robot: usize,
        // },
        DynamicFactor,
        ObstacleFactor,
        PoseFactor,
    }

    impl NodeKind {
        pub fn color(&self) -> &'static str {
            match self {
                Self::Variable { .. } => "#eff1f5",         // latte base (white)
                Self::InterRobotFactor { .. } => "#a6da95", // green
                Self::DynamicFactor => "#8aadf4",           // blue
                Self::ObstacleFactor => "#c6a0f6",          // mauve (purple)
                Self::PoseFactor => "#ee99a0",              // maroon (red)
            }
        }

        pub fn shape(&self) -> &'static str {
            match self {
                Self::Variable { .. } => "circle",
                _ => "square",
            }
        }

        // TODO: return a float
        pub fn width(&self) -> &'static str {
            match self {
                Self::Variable { .. } => "0.8",
                _ => "0.2",
            }
        }
    }

    pub struct Edge {
        pub from: usize,
        pub to: usize,
    }
}

// TODO: implement for each and use
pub trait FactorGraphNode {
    fn messages_received(&self) -> usize;
    fn messages_sent(&self) -> usize;
}

/// How the messages are passed between factors and variables in the connected factorgraphs.
// #[derive(Debug)]
pub enum MessagePassingMode {
    /// Messages are passed within a robot's own factorgraph.
    Internal,
    /// Messages are passed between a robot factorgraph and other robots factorgraphs.
    External,
}

// pub type Inbox<T> = HashMap<NodeIndex, Message<T>>;
pub type Inbox = HashMap<NodeIndex, Message>;

#[derive(Debug, Clone)]
pub enum Node {
    Factor(Factor),
    // TODO: wrap in Box<>
    Variable(Variable),
}

#[derive(Debug, Clone, Copy)]
pub enum EdgeConnection {
    Inter,
    Intra,
}

// pub struct Node {
//     kind: NodeKind,
//     messages_received: usize,
//     messages_sent: usize,
// }

impl Node {
    // pub fn set_node_index(&mut self, index: NodeIndex) {
    //     match self {
    //         Self::Factor(factor) => factor.set_node_index(index),
    //         Self::Variable(variable) => variable.set_node_index(index),
    //     }
    // }
    // pub fn get_node_index(&mut self) -> NodeIndex {
    //     match self {
    //         Self::Factor(factor) => factor.get_node_index(),
    //         Self::Variable(variable) => variable.get_node_index(),
    //     }
    // }

    /// Returns `true` if the node is [`Factor`].
    ///
    /// [`Factor`]: Node::Factor
    #[must_use]
    pub fn is_factor(&self) -> bool {
        matches!(self, Self::Factor(..))
    }

    /// Returns `Some(&Factor)` if the node]s variant is [`Factor`], otherwise `None`.
    pub fn as_factor(&self) -> Option<&Factor> {
        if let Self::Factor(v) = self {
            Some(v)
        } else {
            None
        }
    }

    /// Returns `Some(&mut Factor)` if the node]s variant is [`Factor`], otherwise `None`.
    pub fn as_factor_mut(&mut self) -> Option<&mut Factor> {
        if let Self::Factor(v) = self {
            Some(v)
        } else {
            None
        }
    }

    /// Returns `true` if the node is [`Variable`].
    ///
    /// [`Variable`]: Node::Variable
    #[must_use]
    pub fn is_variable(&self) -> bool {
        matches!(self, Self::Variable(..))
    }

    /// Returns `Some(&Variable)` if the node]s variant is [`Variable`], otherwise `None`.
    pub fn as_variable(&self) -> Option<&Variable> {
        if let Self::Variable(v) = self {
            Some(v)
        } else {
            None
        }
    }

    /// Returns `Some(&mut Variable)` if the node]s variant is [`Variable`], otherwise `None`.
    pub fn as_variable_mut(&mut self) -> Option<&mut Variable> {
        if let Self::Variable(v) = self {
            Some(v)
        } else {
            None
        }
    }
}

/// The type used to represent indices into the nodes of the factorgraph.
/// This is just a type alias for `petgraph::graph::NodeIndex`, but
/// we make an alias for it here, such that it is easier to use the same
/// index type across modules, as the various node index types `petgraph`
/// are not interchangeable.
pub type NodeIndex = petgraph::graph::NodeIndex;
/// The type used to represent indices into the nodes of the factorgraph.
pub type EdgeIndex = petgraph::graph::EdgeIndex;
/// A factorgraph is an undirected graph
pub type Graph = petgraph::graph::Graph<Node, (), Undirected, u32>;
// pub type Graph<T> = petgraph::graph::Graph<Node<T>, (), Undirected>;

/// Record type used to keep track of how many factors and variables
/// there are in the factorgraph. We keep track of these counts internally in the
/// factorgraph, such a query for the counts, is **O(1)**.
/// TODO: redundant now
#[derive(Debug, Clone, Copy)]
pub struct NodeCount {
    pub factors: usize,
    pub variables: usize,
}

/// A factor graph is a bipartite graph consisting of two types of nodes: factors and variables.
/// Factors and variables are stored in separate btree maps, that are indexed by a unique tuple of (robot_id, node_id).
#[derive(Component, Debug)]
pub struct FactorGraph {
    /// The underlying graph data structure
    graph: Graph,
    // / tracks how many variable and factor nodes there are in the graph.
    // node_count: NodeCount,
    /// In **gbpplanner** the sequence in which variables are inserted/created in the graph
    /// is meaningful. `self.graph` does not capture this ordering, so we use an extra queue
    /// to manage the order in which variables are inserted/removed from the graph.
    /// **IMPORTANT** we have to manually ensure the invariant that `self.graph` and this field
    /// is consistent at all time.
    variable_indices: Vec<NodeIndex>,
    factor_indices: Vec<NodeIndex>,
}

pub struct Factors<'a> {
    graph: &'a Graph,
    factor_indices: std::slice::Iter<'a, NodeIndex>,
}

impl<'a> Factors<'a> {
    pub fn new(graph: &'a Graph, factor_indices: &'a [NodeIndex]) -> Self {
        Self {
            graph,
            factor_indices: factor_indices.iter(),
        }
    }
}

impl<'a> Iterator for Factors<'a> {
    type Item = (NodeIndex, &'a Factor);

    fn next(&mut self) -> Option<Self::Item> {
        let &index = self.factor_indices.next()?;
        let node = &self.graph[index];
        node.as_factor().map(|factor| (index, factor))
    }
}

pub struct Variables<'a> {
    graph: &'a Graph,
    variable_indices: std::slice::Iter<'a, NodeIndex>,
}

impl<'a> Variables<'a> {
    pub fn new(graph: &'a Graph, variable_indices: &'a [NodeIndex]) -> Self {
        Self {
            graph,
            variable_indices: variable_indices.iter(),
        }
    }
}

impl<'a> Iterator for Variables<'a> {
    type Item = (NodeIndex, &'a Variable);

    fn next(&mut self) -> Option<Self::Item> {
        let &index = self.variable_indices.next()?;
        let node = &self.graph[index];
        node.as_variable().map(|variable| (index, variable))
    }
}

impl FactorGraph {
    /// Construct a new empty factorgraph
    pub fn new() -> Self {
        Self {
            graph: Graph::new_undirected(),
            // node_count: NodeCount {
            //     factors: 0usize,
            //     variables: 0usize,
            // },
            variable_indices: Vec::new(),
            factor_indices: Vec::new(),
        }
    }

    pub fn variables(&self) -> Variables<'_> {
        Variables::new(&self.graph, &self.variable_indices)
    }

    pub fn factors(&self) -> Factors<'_> {
        Factors::new(&self.graph, &self.factor_indices)
    }

    /// Returns an `Iterator` over the variable nodes in the factorgraph.
    /// Variables are ordered by creation time.
    pub fn variables_ordered(&self) -> impl Iterator<Item = &Variable> {
        self.variable_indices
            .iter()
            .filter_map(move |&node_index| self.graph[node_index].as_variable())
    }

    pub fn add_variable(&mut self, variable: Variable) -> NodeIndex {
        let node_index = self.graph.add_node(Node::Variable(variable));
        // self.graph[node_index].set_node_index(node_index);
        self.variable_indices.push(node_index);
        // self.node_count.variables += 1;
        node_index
    }

    pub fn add_factor(&mut self, factor: Factor) -> NodeIndex {
        let node_index = self.graph.add_node(Node::Factor(factor));
        // self.graph[node_index].set_node_index(node_index);
        // self.node_count.factors += 1;
        self.factor_indices.push(node_index);
        node_index
    }

    /// Add an edge between nodes `a` and `b` in the factorgraph.
    ///
    /// **invariants**:
    /// - Both `a` and `b` must already be in the factorgraph. Panics if any of the nodes does not exist.
    pub fn add_edge(&mut self, a: NodeIndex, b: NodeIndex) -> EdgeIndex {
        let dofs = 4;
        // TODO: explain why we send an empty message
        match self.graph[a] {
            Node::Factor(ref mut factor) => factor.send_message(b, Message::empty(dofs)),
            Node::Variable(ref mut variable) => variable.send_message(b, Message::empty(dofs)),
        }
        match self.graph[b] {
            Node::Factor(ref mut factor) => factor.send_message(a, Message::empty(dofs)),
            Node::Variable(ref mut variable) => variable.send_message(a, Message::empty(dofs)),
        }
        self.graph.add_edge(a, b, ())
    }

    /// Number of nodes in the factorgraph
    ///
    /// **Computes in O(1) time**
    pub fn len(&self) -> usize {
        self.graph.node_count()
    }

    /// Return an ordered interval of variables indices.
    /// The indices are ordered by the order in which they are inserted into the factorgraph.
    /// Returns `None`, if the end of the  **range** exceeds the number of variables in the factorgraph.
    pub fn variable_indices_ordered_by_creation(
        &self,
        range: Range<usize>,
    ) -> Option<Vec<NodeIndex>> {
        let within_range = range.end <= self.variable_indices.len();
        if within_range {
            Some(
                self.variable_indices
                    .iter()
                    .skip(range.start)
                    .take(range.end - range.start)
                    .copied()
                    .collect::<Vec<_>>(),
            )
        } else {
            None
        }
    }

    // pub fn factors(&self) -> impl Iterator<Item = Node> {}

    /// A count over the number of variables and factors in the factorgraph
    ///
    /// **Computes in O(1) time**
    pub fn node_count(&self) -> NodeCount {
        NodeCount {
            factors: self.factor_indices.len(),
            variables: self.variable_indices.len(),
        }
    }

    #[inline(always)]
    pub fn nth_variable_index(&self, index: usize) -> Option<NodeIndex> {
        self.variable_indices.get(index).copied()
    }

    pub fn nth_variable(&self, index: usize) -> Option<(NodeIndex, &Variable)> {
        let variable_index = self.nth_variable_index(index)?;
        let node = &self.graph[variable_index];
        let variable = node.as_variable()?;
        Some((variable_index, variable))
    }

    pub fn nth_variable_mut(&mut self, index: usize) -> Option<(NodeIndex, &mut Variable)> {
        let variable_index = self.nth_variable_index(index)?;
        let node = &mut self.graph[variable_index];
        let variable = node.as_variable_mut()?;
        Some((variable_index, variable))
    }

    pub fn first_variable(&self) -> Option<(NodeIndex, &Variable)> {
        self.nth_variable(0usize)
    }

    #[inline(always)]
    pub fn last_variable(&self) -> Option<(NodeIndex, &Variable)> {
        if self.variable_indices.is_empty() {
            None
        } else {
            self.nth_variable(self.variable_indices.len() - 1)
        }
    }

    #[inline(always)]
    pub fn last_variable_mut(&mut self) -> Option<(NodeIndex, &mut Variable)> {
        if self.variable_indices.is_empty() {
            None
        } else {
            self.nth_variable_mut(self.variable_indices.len() - 1)
        }
    }

    pub fn export_data(&self) -> (Vec<graphviz::Node>, Vec<graphviz::Edge>) {
        let nodes = self
            .graph
            .node_indices()
            .map(|node_index| {
                let node = &self.graph[node_index];
                graphviz::Node {
                    index: node_index.index(),
                    kind: match node {
                        Node::Factor(factor) => match factor.kind {
                            FactorKind::Dynamic(_) => graphviz::NodeKind::DynamicFactor,
                            FactorKind::Obstacle(_) => graphviz::NodeKind::ObstacleFactor,
                            FactorKind::Pose(_) => graphviz::NodeKind::PoseFactor,
                            FactorKind::InterRobot(inner) => {
                                graphviz::NodeKind::InterRobotFactor(inner.connection.clone())
                            }
                        },
                        Node::Variable(variable) => {
                            // let mean = variable.belief.mean();
                            let mean = &variable.mu;
                            graphviz::NodeKind::Variable {
                                x: mean[0] as f32,
                                y: mean[1] as f32,
                            }
                        }
                    },
                }
            })
            .collect::<Vec<_>>();

        let edges = self
            .graph
            .edge_indices()
            .filter_map(|edge_index| {
                self.graph
                    .edge_endpoints(edge_index)
                    .map(|(from, to)| graphviz::Edge {
                        from: from.index(),
                        to: to.index(),
                    })
            })
            .collect::<Vec<_>>();

        (nodes, edges)
    }

    /// Aggregate and marginalise over all adjacent variables, and send.
    /// Aggregation: product of all incoming messages
    pub fn factor_iteration(&mut self, robot_id: RobotId, mode: MessagePassingMode) {
        // TODO: use rayon .par_iter()
        for factor_index in self.graph.node_indices() {
            if let Node::Variable(_) = self.graph[factor_index] {
                continue;
            }

            let adjacent_variables = self.graph.neighbors(factor_index).collect::<Vec<_>>();
            let variable_messages = self.update_factor(factor_index, adjacent_variables);

            for (variable_index, message) in variable_messages {
                let variable = self.graph[variable_index]
                    .as_variable_mut()
                    .expect("A factor can only have variables as neighbors");
                variable.send_message(factor_index, message);
            }
        }
    }

    fn update_factor(
        &mut self,
        factor_index: NodeIndex,
        adjacent_variables: Vec<NodeIndex>,
    ) -> HashMap<NodeIndex, Message> {
        // TODO: do not hardcode
        let dofs = 4;

        let factor = self.graph[factor_index]
            .as_factor_mut()
            .expect("factor_index should point to a Factor in the graph");

        let empty_mean = Vector::<Float>::zeros(dofs);
        // Collect the means of the incoming messages from the connected variables
        for (i, &variable_index) in adjacent_variables.iter().enumerate() {
            let message = factor
                .read_message_from(variable_index)
                .expect("There should be a message from the variable");
            let mean = message.mean().unwrap_or_else(|| &empty_mean).clone();
            factor
                .state
                .linearisation_point
                .slice_mut(s![i * dofs..(i + 1) * dofs])
                .assign(&mean);
        }

        // *Depending on the problem*, we may need to skip computation of this factor.
        // eg. to avoid extra computation, factor may not be required if two connected variables are too far apart.
        // in which case send out a Zero Message.
        if factor.skip() {
            warn!("The factor {:?} is skipped", factor_index);
            let messages = adjacent_variables
                .iter()
                .map(|&variable_index| (variable_index, Message::empty(dofs)))
                .collect::<HashMap<_, _>>();

            return messages;
        }

        // TODO: avoid clone
        let _ = factor.measure(&factor.state.linearisation_point.clone());
        let jacobian = factor.jacobian(&factor.state.linearisation_point.clone());

        // eprintln!("jacobian =");
        // dbg!(factor.name());
        // pretty_print_matrix!(&jacobian);
        // jacobian.pretty_print();
        // eprintln!("factor.state.measurement_precision =");
        // factor.state.measurement_precision.pretty_print();
        // eprintln!("factor.state.linearisation_point =");
        // pretty_print::pre
        // pretty_print_vector!(&factor.state.linearisation_point);
        // factor.state.linearisation_point.pretty_print();

        let factor_lambda_potential = jacobian
            .t()
            .dot(&factor.state.measurement_precision)
            .dot(&jacobian);
        let factor_eta_potential = jacobian
            .t()
            .dot(&factor.state.measurement_precision)
            .dot(&(jacobian.dot(&factor.state.linearisation_point) + factor.residual()));

        factor.mark_as_initialized();

        // eprintln!("factor_eta_potential =");
        // pretty_print_vector!(&factor_eta_potential);
        // factor_eta_potential.pretty_print();
        // eprintln!("factor_lam_potential =");
        // pretty_print_matrix!(&factor_lam_potential);
        // factor_lam_potential.pretty_print();

        if factor_eta_potential.iter().all(|x| x.is_zero()) {
            // warn!("The factor {:?} has a zero potential", factor_index);
            let messages = adjacent_variables
                .iter()
                .map(|&variable_index| {
                    let message = Message::empty(dofs);
                    (variable_index, message)
                })
                .collect::<HashMap<_, _>>();

            return messages;
        }

        // update factor precision and information with incoming messages from connected variables.
        let mut marginalisation_idx = 0;
        let mut messages = HashMap::with_capacity(adjacent_variables.iter().len());

        let empty_precision = Matrix::<Float>::zeros((dofs, dofs));
        // For each variable, marginalise over the factor precision and information from all other variables except the current one
        for &variable_index in adjacent_variables.iter() {
            let mut factor_eta = factor_eta_potential.clone();
            let mut factor_lambda = factor_lambda_potential.clone();

            for (j, &other_variable_index) in adjacent_variables.iter().enumerate() {
                if other_variable_index != variable_index {
                    let message = factor
                        .read_message_from(other_variable_index)
                        .expect("There should be a message from the variable");

                    let message_mean = message.mean().unwrap_or_else(|| &empty_mean);
                    let message_precision = message
                        .precision_matrix()
                        .unwrap_or_else(|| &empty_precision);
                    factor_eta
                        .slice_mut(s![j * dofs..(j + 1) * dofs])
                        .add_assign(message_mean);
                    factor_lambda
                        .slice_mut(s![j * dofs..(j + 1) * dofs, j * dofs..(j + 1) * dofs])
                        .add_assign(message_precision);
                }
            }

            // // Marginalise the Factor Precision and Information to send to the relevant variable
            // let message = if message_is_empty {
            //     Message::empty(dofs)
            // } else {
            //     marginalise_factor_distance(factor_eta, factor_lam, dofs, marginalisation_idx)
            //         .unwrap()
            // };

            let message =
                marginalise_factor_distance(factor_eta, factor_lambda, dofs, marginalisation_idx)
                    .unwrap();
            messages.insert(variable_index, message);
            marginalisation_idx += dofs;
        }

        messages
    }

    /// Variable Iteration in Gaussian Belief Propagation (GBP).
    /// For each variable in the factorgraph:
    ///  - Messages are collected from the outboxes of each of the connected factors
    ///  - Variable belief is updated and outgoing message in the variable's outbox is created.
    ///
    ///  * Note: we deal with cases where the variable/factor iteration may need to be skipped:
    ///      - communications failure modes:
    ///          if interrobot_comms_active_ is false, variables and factors connected to
    ///          other robots should not take part in GBP iterations,
    ///      - message passing modes (INTERNAL within a robot's own factorgraph or EXTERNAL between a robot and other robots):
    ///          in which case the variable or factor may or may not need to take part in GBP depending on if it's connected to another robot
    pub fn variable_iteration(&mut self, robot_id: RobotId, mode: MessagePassingMode) {
        // TODO: send messages to connected interrobot factors in other robots factorgraphs

        for variable_index in self.graph.node_indices() {
            // if let Node::Factor(_) = self.graph[variable_index] {
            //     continue;
            // }

            let node = &mut self.graph[variable_index];
            if node.is_factor() {
                continue;
            }

            let variable = node
                .as_variable_mut()
                .expect("variable_index should point to a Variable in the graph");

            let factor_messages = variable.update_belief_and_create_responses();
            if factor_messages.is_empty() {
                error!(
                    "The variable {:?} did not receive any messages from its connected factors",
                    variable_index
                );
                continue;
            }
            // send the messages to the connected factors
            for (factor_index, message) in factor_messages {
                let factor = self.graph[factor_index]
                    .as_factor_mut()
                    .expect("The variable can only send messages to factors");
                factor.send_message(variable_index, message);
            }
        }
    }

    pub fn change_prior_of_variable(&mut self, variable_index: NodeIndex, new_mean: Vector<Float>) {
        let indices_of_adjacent_factors = self.graph.neighbors(variable_index).collect::<Vec<_>>();
        let variable = self.graph[variable_index]
            .as_variable_mut()
            .expect("variable_index should point to a Variable in the graph");

        let factor_messages = variable.change_prior(&new_mean, indices_of_adjacent_factors);

        for (factor_index, message) in factor_messages {
            let factor = self.graph[factor_index]
                .as_factor_mut()
                .expect("The variable can only send messages to factors");
            factor.send_message(variable_index, message);
        }
    }

    pub(crate) fn delete_interrobot_factor_connected_to(
        &mut self,
        other: RobotId,
    ) -> Result<(), &'static str> {
        let node_idx = self
        .graph
        .node_indices()
        // Use `find_map` for a more concise filter-and-map operation
        .find_map(|node_idx| {
            let node = &self.graph[node_idx];
            node.as_factor()
                .and_then(|factor| factor.kind.as_inter_robot())
                // Extract `id_of_robot_connected_with` directly
                .filter(|interrobot| interrobot.connection.id_of_robot_connected_with == other)
                .map(|_interrobot| node_idx)
        })
        .ok_or("not found")?;

        // Directly remove the node using the fallible method
        self.graph
            .remove_node(node_idx)
            .ok_or("the interrobot factor does not exist in the graph")?;

        Ok(())
    }

    // /// TODO: should probably not be a method on the graph, but on the robot, but whatever
    // pub(crate) fn delete_interrobot_factor_connected_to(
    //     &mut self,
    //     other: RobotId,
    // ) -> Result<(), &'static str> {
    //     let node_idx = self
    //         .graph
    //         .node_indices()
    //         .filter_map(|node_idx| {
    //             let node = &self.graph[node_idx];
    //             let Some(factor) = node.as_factor() else {
    //                 return None;
    //             };

    //             let Some(interrobot) = factor.kind.as_inter_robot() else {
    //                 return None;
    //             };

    //             Some((node_idx, interrobot))
    //         })
    //         .find(|(_, interrobot)| interrobot.id_of_robot_connected_with == other)
    //         .map(|(node_idx, _)| node_idx);

    //     let Some(node_idx) = node_idx else {
    //         return Err("not found");
    //     };

    //     self.graph.remove_node(node_idx).expect(
    //         "The node index was retrieved from the graph in the previous statement",
    //     );

    //     Ok(())

    //     // let node_idx = self
    //     //     .graph
    //     //     .raw_nodes()
    //     //     .iter()
    //     //     .filter_map(|node| node.weight.as_factor())
    //     //     .filter_map(|factor| factor.kind.as_inter_robot())
    //     //     .find(|&interrobot| interrobot.id_of_robot_connected_with == other)
    //     //     .ok_or("not found")?;
    // }

    // fn update_variable(
    //     &mut self,
    //     variable_index: NodeIndex,
    //     indices_of_adjacent_factors: Vec<NodeIndex>,
    // ) -> HashMap<NodeIndex, Message> {
    //     let adjacent_factors = self.graph.neighbors(variable_index);
    //     // // Update variable belief and create outgoing messages
    //     // variable.update_belief(&adjacent_factors, &mut self.graph);

    //     todo!()
    // }
}
