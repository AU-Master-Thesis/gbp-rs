use std::{
    collections::HashMap,
    num::NonZeroUsize,
    ops::{AddAssign, Sub},
};

use bevy::{
    log::{info, warn},
    render::texture::Image,
};
use gbp_linalg::{pretty_print_matrix, pretty_print_vector, Float, Matrix, Vector, VectorNorm};
use ndarray::{array, concatenate, s, Axis};
use petgraph::prelude::NodeIndex;
use typed_floats::StrictlyPositiveFinite;

use super::{
    factorgraph::{FactorGraphNode, MessagesFromFactors, MessagesToVariables, VariableId},
    message::Message,
    robot::RobotId,
};
use crate::{
    escape_codes::*,
    planner::{factorgraph::FactorId, marginalise_factor_distance::marginalise_factor_distance},
    pretty_print_message,
};

// TODO: make generic over f32 | f64
// TODO: hide the state parameter from the public API, by having the `Factor`
// struct expose similar methods that dispatch to the `FactorState` struct.
trait Model {
    /// The name of the factor. Used for debugging and visualization.
    fn name(&self) -> &'static str;

    fn jacobian_delta(&self) -> Float;

    /// Whether to skip this factor in the update step
    /// In gbpplanner, this is only used for the interrobot factor.
    /// The other factors are always included in the update step.
    fn skip(&mut self, state: &FactorState) -> bool;

    /// Whether the factor is linear or non-linear
    fn linear(&self) -> bool;

    #[must_use]
    #[inline]
    fn jacobian(&mut self, state: &FactorState, x: &Vector<Float>) -> Matrix<Float> {
        self.first_order_jacobian(state, x.clone())
    }

    /// Measurement function
    /// **Note**: This method takes a mutable reference to self, because the
    /// interrobot factor
    fn measure(&mut self, state: &FactorState, x: &Vector<Float>) -> Vector<Float>;

    fn first_order_jacobian(&mut self, state: &FactorState, mut x: Vector<Float>) -> Matrix<Float> {
        let h0 = self.measure(state, &x); // value at linearization point
        let mut jacobian = Matrix::<Float>::zeros((h0.len(), x.len()));

        for i in 0..x.len() {
            x[i] += self.jacobian_delta(); // perturb by delta
            let derivatives = (self.measure(state, &x) - &h0) / self.jacobian_delta();
            jacobian.column_mut(i).assign(&derivatives);
            x[i] -= self.jacobian_delta(); // reset the perturbation
        }

        jacobian
    }
}

#[derive(Debug, Clone, Copy)]
pub struct InterRobotConnection {
    pub id_of_robot_connected_with: RobotId,
    pub index_of_connected_variable_in_other_robots_factorgraph: NodeIndex,
}

impl InterRobotConnection {
    #[must_use]
    pub fn new(
        id_of_robot_connected_with: RobotId,
        index_of_connected_variable_in_other_robots_factorgraph: NodeIndex,
    ) -> Self {
        Self {
            id_of_robot_connected_with,
            index_of_connected_variable_in_other_robots_factorgraph,
        }
    }
}

// #[derive(Debug, Clone, Copy, PartialEq, Eq)]
// pub struct Skip(bool);

/// Interrobot factor: for avoidance of other robots
/// This factor results in a high energy or cost if two robots are planning to
/// be in the same position at the same timestep (collision). This factor is
/// created between variables of two robots. The factor has 0 energy if the
/// variables are further away than the safety distance.
#[derive(Debug, Clone)]
pub struct InterRobotFactor {
    safety_distance: Float,
    skip:            bool,
    pub connection:  InterRobotConnection,
}

#[derive(Debug, thiserror::Error)]
pub enum InterRobotFactorError {
    #[error("The robot radius must be positive, but it was {0}")]
    NegativeRobotRadius(Float),
}

impl InterRobotFactor {
    #[must_use]
    pub fn new(
        robot_radius: Float,
        connection: InterRobotConnection,
    ) -> Result<Self, InterRobotFactorError> {
        if robot_radius < 0.0 {
            return Err(InterRobotFactorError::NegativeRobotRadius(robot_radius));
        }
        let epsilon = 0.2 * robot_radius;

        Ok(Self {
            safety_distance: 2.0 * robot_radius + epsilon,
            skip: false,
            connection,
        })
    }
}

impl Model for InterRobotFactor {
    #[inline(always)]
    fn name(&self) -> &'static str {
        "InterRobotFactor"
    }

    fn jacobian(&mut self, state: &FactorState, x: &Vector<Float>) -> Matrix<Float> {
        let dofs = state.dofs.get();
        let mut jacobian = Matrix::<Float>::zeros((state.initial_measurement.len(), dofs * 2));
        let x_diff = {
            let offset = dofs / 2;
            let mut x_diff = x.slice(s![..offset]).sub(&x.slice(s![dofs..dofs + offset]));

            // NOTE: In gbplanner, they weight this by the robot id, why they do this is
            // unclear as a robot id should be unique, and not have any
            // semantics of distance/weight.
            for i in 0..offset {
                x_diff[i] += 1e-6 * self.connection.id_of_robot_connected_with.index() as Float;
                // Add a tiny random offset to avoid div/0 errors
            }
            x_diff
        };

        let radius = x_diff.euclidean_norm();
        if radius <= self.safety_distance {
            // TODO: why do we change the Jacobian if we are not outside the safety
            // distance?

            // J(0, seqN(0, n_dofs_ / 2)) = -1.f / safety_distance_ / r * X_diff;
            jacobian
                .slice_mut(s![0, ..dofs / 2])
                .assign(&(-1.0 / self.safety_distance / radius * &x_diff));

            // J(0, seqN(n_dofs_, n_dofs_ / 2)) = 1.f / safety_distance_ / r * X_diff;
            jacobian
                .slice_mut(s![0, dofs..dofs + (dofs / 2)])
                .assign(&(1.0 / self.safety_distance / radius * &x_diff));
        }
        jacobian
    }

    fn measure(&mut self, state: &FactorState, x: &Vector<Float>) -> Vector<Float> {
        let dofs = state.dofs.get();
        let mut h = Vector::<Float>::zeros(state.initial_measurement.len());
        let x_diff = {
            let offset = dofs / 2;
            let mut x_diff = x.slice(s![..offset]).sub(&x.slice(s![dofs..dofs + offset]));
            // NOTE: In gbplanner, they weight this by the robot id, why they do this is
            // unclear as a robot id should be unique, and not have any
            // semantics of distance/weight.
            for i in 0..offset {
                // Add a tiny random offset to avoid div/0 errors
                x_diff[i] += 1e-6 * self.connection.id_of_robot_connected_with.index() as Float;
            }
            x_diff
        };

        let radius = x_diff.euclidean_norm();
        let within_safety_distance = radius <= self.safety_distance;
        // match (self.skip, within_safety_distance) {
        //     (Skip(true), true) => {}
        //     (Skip(true), false) => {}
        //     (Skip(false), true) => {
        //         self.skip = Skip(true);
        //         info!("skip = true, radius = {}", radius);
        //     }
        //     (Skip(false), false) => {}
        // };
        if radius <= self.safety_distance {
            if self.skip {
                info!(
                    "within safety distance, radius = {}, setting self.skip to false",
                    radius
                );
            }
            self.skip = false;
            // gbpplanner: h(0) = 1.f*(1 - r/safety_distance_);
            // NOTE: in Eigen, indexing a matrix with a single index corresponds to indexing
            // the matrix as a flattened array in column-major order.
            // h[(0, 0)] = 1.0 * (1.0 - radius / self.safety_distance);
            h[0] = 1.0 * (1.0 - radius / self.safety_distance);
            // println!("h = {}", h);
        } else {
            if !self.skip {
                info!(
                    "outside safety distance, radius = {}, setting self.skip to true",
                    radius
                );
            }
            self.skip = true;
        }

        h
    }

    #[inline(always)]
    fn jacobian_delta(&self) -> Float {
        1e-2
    }

    fn skip(&mut self, state: &FactorState) -> bool {
        // this->skip_flag = ( (X_(seqN(0,n_dofs_/2)) - X_(seqN(n_dofs_,
        // n_dofs_/2))).squaredNorm() >= safety_distance_*safety_distance_ );
        let dofs = state.dofs.get();
        let offset = dofs / 2;

        // [..offset] is the position of the first variable
        // [dofs..dofs + offset] is the position of the other variable

        let difference_between_estimated_positions = state
            .linearisation_point
            .slice(s![..offset])
            .sub(&state.linearisation_point.slice(s![dofs..dofs + offset]));
        let squared_norm = difference_between_estimated_positions
            .mapv(|x| x.powi(2))
            .sum();

        let skip = squared_norm >= self.safety_distance.powi(2);
        // let skip = squared_norm >= Float::powi(self.safety_distance, 2);
        if self.skip != skip {
            // warn!(
            //     "skip = {}, squared_norm = {} safety_distance^2 = {}",
            //     skip,
            //     squared_norm,
            //     Float::powi(self.safety_distance, 2)
            // );
        }
        self.skip = skip;
        // self.skip = squared_norm >= Float::powi(self.safety_distance, 2);
        self.skip
    }

    #[inline(always)]
    fn linear(&self) -> bool {
        false
    }
}

/// Dynamic factor: constant velocity model
#[derive(Debug)]
pub struct DynamicFactor {
    cached_jacobian: Matrix<Float>,
}

// TODO: constrain within the limits that a differential drive robot can turn

impl DynamicFactor {
    #[must_use]
    pub fn new(state: &mut FactorState, delta_t: Float) -> Self {
        let dofs = state.dofs.get();
        let eye = Matrix::<Float>::eye(dofs / 2);
        let zeros = Matrix::<Float>::zeros((dofs / 2, dofs / 2));
        #[allow(clippy::similar_names)]
        let qc_inv = Float::powi(state.strength, -2) * &eye;
        // pretty_print_matrix!(&qc_inv);

        #[allow(clippy::similar_names)]
        let qi_inv = concatenate![
            Axis(0),
            concatenate![
                Axis(1),
                12.0 * Float::powi(delta_t, -3) * &qc_inv,
                -6.0 * Float::powi(delta_t, -2) * &qc_inv
            ],
            concatenate![
                Axis(1),
                -6.0 * Float::powi(delta_t, -2) * &qc_inv,
                (4.0 / delta_t) * &qc_inv
            ]
        ];
        debug_assert_eq!(qi_inv.shape(), &[dofs, dofs]);

        // pretty_print_matrix!(&qi_inv);

        // std::process::exit(1);

        state.measurement_precision = qi_inv;

        let cached_jacobian = concatenate![
            Axis(0),
            concatenate![Axis(1), eye, delta_t * &eye, -1.0 * &eye, zeros],
            concatenate![Axis(1), zeros, eye, zeros, -1.0 * &eye]
        ];
        debug_assert_eq!(cached_jacobian.shape(), &[dofs, dofs * 2]);

        // pretty_print_matrix!(&cached_jacobian);

        // std::process::exit(1);

        Self { cached_jacobian }
    }
}

impl Model for DynamicFactor {
    #[inline]
    fn name(&self) -> &'static str {
        "DynamicFactor"
    }

    #[inline]
    fn jacobian(&mut self, _state: &FactorState, _x: &Vector<Float>) -> Matrix<Float> {
        self.cached_jacobian.clone()
    }

    #[inline(always)]
    fn measure(&mut self, _state: &FactorState, x: &Vector<Float>) -> Vector<Float> {
        self.cached_jacobian.dot(x)
    }

    #[inline(always)]
    fn skip(&mut self, _state: &FactorState) -> bool {
        false
    }

    #[inline(always)]
    fn jacobian_delta(&self) -> Float {
        1e-8
    }

    #[inline(always)]
    fn linear(&self) -> bool {
        true
    }
}

#[derive(Debug)]
pub struct PoseFactor;

impl Model for PoseFactor {
    #[inline(always)]
    fn name(&self) -> &'static str {
        "PoseFactor"
    }

    /// Default jacobian is the first order taylor series jacobian
    fn jacobian(&mut self, state: &FactorState, x: &Vector<Float>) -> Matrix<Float> {
        self.first_order_jacobian(state, x.clone())
    }

    /// Default measurement function is the identity function
    #[inline(always)]
    fn measure(&mut self, _state: &FactorState, x: &Vector<Float>) -> Vector<Float> {
        x.clone()
    }

    #[inline(always)]
    fn skip(&mut self, _state: &FactorState) -> bool {
        false
    }

    #[inline(always)]
    fn jacobian_delta(&self) -> Float {
        1e-8
    }

    #[inline(always)]
    fn linear(&self) -> bool {
        false
    }
}

#[derive(Clone)]
pub struct ObstacleFactor {
    /// The signed distance field of the environment
    obstacle_sdf: &'static Image,
    /// Copy of the `WORLD_SZ` setting from **gbpplanner**, that we store a copy
    /// of here since `ObstacleFactor` needs this information to calculate
    /// `.jacobian_delta()` and `.measurement()`
    world_size:   Float,
}

impl std::fmt::Debug for ObstacleFactor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ObstacleFactor")
            // .field("obstacle_sdf", &self.obstacle_sdf)
            .field("world_size", &self.world_size)
            .finish()
    }
}

impl ObstacleFactor {
    /// Creates a new [`ObstacleFactor`].
    #[must_use]
    fn new(obstacle_sdf: &'static Image, world_size: Float) -> Self {
        Self {
            obstacle_sdf,
            world_size,
        }
    }
}

impl Model for ObstacleFactor {
    #[inline]
    fn name(&self) -> &'static str {
        "ObstacleFactor"
    }

    #[inline]
    fn jacobian(&mut self, state: &FactorState, x: &Vector<Float>) -> Matrix<Float> {
        // Same as PoseFactor
        // TODO: change to not clone x
        self.first_order_jacobian(state, x.clone())
    }

    fn measure(&mut self, _state: &FactorState, x: &Vector<Float>) -> Vector<Float> {
        // pretty_print_vector!(x);
        debug_assert!(x.len() >= 2, "x.len() = {}", x.len());
        // White areas are obstacles, so h(0) should return a 1 for these regions.
        let scale = self.obstacle_sdf.width() as Float / self.world_size;
        // let offset = (self.world_size / 2.0) as usize;
        let offset = self.world_size / 2.0;
        if (x[0] + offset) * scale > self.obstacle_sdf.width() as Float {
            // warn!(
            //     "x[0] + offset = {}, scale = {}, width = {}",
            //     (x[0] + offset) * scale,
            //     scale,
            //     self.obstacle_sdf.width()
            // );
            return array![0.0];
        }
        if (x[1] + offset) * scale > self.obstacle_sdf.height() as Float {
            // warn!(
            //     "x[1] + offset = {}, scale = {}, height = {}",
            //     (x[1] + offset) * scale,
            //     scale,
            //     self.obstacle_sdf.height()
            // );
            return array![0.0];
        }
        // dbg!(offset);
        let pixel_x = ((x[0] + offset) * scale) as u32;
        let pixel_y = ((x[1] + offset) * scale) as u32;
        // println!("pixel_x = {}, pixel_y = {}", pixel_x, pixel_y);
        // dbg!(pixel_x, pixel_y);
        // assert_eq!((self.obstacle_sdf.width() * self.obstacle_sdf.height() * 4) as
        // usize, self.obstacle_sdf.data.len()); multiply by 4 because the image
        // is in RGBA format, and we simply use th R channel to determine value,
        // as the image is grayscale
        // TODO: assert that the image's data is laid out in row-major order
        let linear_index = ((self.obstacle_sdf.width() * pixel_y + pixel_x) * 4) as usize;
        if linear_index >= self.obstacle_sdf.data.len() {
            warn!(
                "linear_index = {}, obstacle_sdf.data.len() = {}",
                linear_index,
                self.obstacle_sdf.data.len()
            );
            return array![0.0];
        }
        let red = self.obstacle_sdf.data[linear_index];
        // NOTE: do 1.0 - red to invert the value, as the obstacle sdf is white where
        // there are obstacles in gbpplanner, they do not do the inversion here,
        // but instead invert the entire image, when they load it from disk.
        let hsv_value = 1.0 - red as Float / 255.0;
        // let hsv_value = pixel as Float / 255.0;
        // if hsv_value <= 0.5 {
        //     println!("image(x={}, y={}).z {} (scale = {})", pixel_x, pixel_y,
        // hsv_value, scale); }
        // dbg!(hsv_value);

        // println!("hsv_value = {}", hsv_value);

        array![hsv_value]
    }

    #[inline(always)]
    fn jacobian_delta(&self) -> Float {
        self.world_size / self.obstacle_sdf.width() as Float
    }

    #[inline(always)]
    fn skip(&mut self, _state: &FactorState) -> bool {
        false
    }

    #[inline(always)]
    fn linear(&self) -> bool {
        false
    }
}

#[derive(Debug)]
pub enum FactorKind {
    Pose(PoseFactor),
    InterRobot(InterRobotFactor),
    Dynamic(DynamicFactor),
    Obstacle(ObstacleFactor),
}

impl FactorKind {
    /// Returns `true` if the factor kind is [`Obstacle`].
    ///
    /// [`Obstacle`]: FactorKind::Obstacle
    #[must_use]
    pub fn is_obstacle(&self) -> bool {
        matches!(self, Self::Obstacle(..))
    }

    /// Returns `true` if the factor kind is [`Dynamic`].
    ///
    /// [`Dynamic`]: FactorKind::Dynamic
    #[must_use]
    pub fn is_dynamic(&self) -> bool {
        matches!(self, Self::Dynamic(..))
    }

    /// Returns `true` if the factor kind is [`InterRobot`].
    ///
    /// [`InterRobot`]: FactorKind::InterRobot
    #[must_use]
    pub fn is_inter_robot(&self) -> bool {
        matches!(self, Self::InterRobot(..))
    }

    /// Returns `true` if the factor kind is [`Pose`].
    ///
    /// [`Pose`]: FactorKind::Pose
    #[must_use]
    pub fn is_pose(&self) -> bool {
        matches!(self, Self::Pose(..))
    }

    pub fn as_pose(&self) -> Option<&PoseFactor> {
        if let Self::Pose(v) = self {
            Some(v)
        } else {
            None
        }
    }

    pub fn as_inter_robot(&self) -> Option<&InterRobotFactor> {
        if let Self::InterRobot(v) = self {
            Some(v)
        } else {
            None
        }
    }

    pub fn as_dynamic(&self) -> Option<&DynamicFactor> {
        if let Self::Dynamic(v) = self {
            Some(v)
        } else {
            None
        }
    }

    pub fn as_obstacle(&self) -> Option<&ObstacleFactor> {
        if let Self::Obstacle(v) = self {
            Some(v)
        } else {
            None
        }
    }
}

impl Model for FactorKind {
    fn name(&self) -> &'static str {
        match self {
            FactorKind::Pose(f) => f.name(),
            FactorKind::InterRobot(f) => f.name(),
            FactorKind::Dynamic(f) => f.name(),
            FactorKind::Obstacle(f) => f.name(),
        }
    }

    fn jacobian(&mut self, state: &FactorState, x: &Vector<Float>) -> Matrix<Float> {
        match self {
            FactorKind::Pose(f) => f.jacobian(state, x),
            FactorKind::InterRobot(f) => f.jacobian(state, x),
            FactorKind::Dynamic(f) => f.jacobian(state, x),
            FactorKind::Obstacle(f) => f.jacobian(state, x),
        }
    }

    fn measure(&mut self, state: &FactorState, x: &Vector<Float>) -> Vector<Float> {
        match self {
            FactorKind::Pose(f) => f.measure(state, x),
            FactorKind::InterRobot(f) => f.measure(state, x),
            FactorKind::Dynamic(f) => f.measure(state, x),
            FactorKind::Obstacle(f) => f.measure(state, x),
        }
    }

    fn skip(&mut self, state: &FactorState) -> bool {
        match self {
            FactorKind::Pose(f) => f.skip(state),
            FactorKind::InterRobot(f) => f.skip(state),
            FactorKind::Dynamic(f) => f.skip(state),
            FactorKind::Obstacle(f) => f.skip(state),
        }
    }

    fn jacobian_delta(&self) -> Float {
        match self {
            FactorKind::Pose(f) => f.jacobian_delta(),
            FactorKind::InterRobot(f) => f.jacobian_delta(),
            FactorKind::Dynamic(f) => f.jacobian_delta(),
            FactorKind::Obstacle(f) => f.jacobian_delta(),
        }
    }

    fn linear(&self) -> bool {
        match self {
            FactorKind::Pose(f) => f.linear(),
            FactorKind::InterRobot(f) => f.linear(),
            FactorKind::Dynamic(f) => f.linear(),
            FactorKind::Obstacle(f) => f.linear(),
        }
    }
}

// TODO: make generic over f32 | f64
// <T: nalgebra::Scalar + Copy>
#[derive(Debug, Clone)]
pub struct FactorState {
    /// called `z_` in **gbpplanner**
    pub initial_measurement:   Vector<Float>,
    /// called `meas_model_lambda_` in **gbpplanner**
    pub measurement_precision: Matrix<Float>,
    /// Stored linearisation point
    /// called `X_` in **gbpplanner**, they use `Eigen::MatrixXd` instead
    pub linearisation_point:   Vector<Float>,
    /// Strength of the factor. Called `sigma` in gbpplanner.
    /// The factor precision $Lambda = sigma^-2 * Identify$
    pub strength:              Float,
    /// Number of degrees of freedom e.g. 4 [x, y, x', y']
    pub dofs:                  NonZeroUsize,

    /// Cached value of the factors jacobian function
    /// called `J_` in **gbpplanner**
    pub cached_jacobian: Matrix<Float>,

    /// Cached value of the factors jacobian function
    /// called `h_` in **gbpplanner**
    pub cached_measurement: Vector<Float>,
    /// Set to true after the first call to self.update()
    /// TODO: move to FactorState
    initialized:            bool,
}

impl FactorState {
    fn new(
        measurement: Vector<Float>,
        strength: Float,
        dofs: NonZeroUsize,
        neighbor_amount: usize,
    ) -> Self {
        // Initialise precision of the measurement function
        // this->meas_model_lambda_ = Eigen::MatrixXd::Identity(z_.rows(), z_.rows()) /
        // pow(sigma,2.);
        let measurement_precision =
            Matrix::<Float>::eye(measurement.len()) / Float::powi(strength, 2);

        Self {
            initial_measurement: measurement,
            measurement_precision,
            linearisation_point: Vector::<Float>::zeros(dofs.get() * neighbor_amount),
            strength,
            dofs,
            cached_jacobian: array![[]],
            cached_measurement: array![],
            initialized: false,
        }
    }
}

#[derive(Debug)]
pub struct Factor {
    /// Unique identifier that associates the variable with a factorgraph/robot.
    pub node_index: Option<NodeIndex>,
    /// State common between all factor kinds
    pub state:      FactorState,
    /// Variant storing the specialized behavior of each Factor kind.
    pub kind:       FactorKind,
    /// Mailbox for incoming message storage
    pub inbox:      MessagesFromFactors,
}

impl Factor {
    fn new(state: FactorState, kind: FactorKind) -> Self {
        Self {
            node_index: None,
            state,
            kind,
            inbox: MessagesFromFactors::new(),
        }
    }

    pub fn new_dynamic_factor(
        strength: Float,
        measurement: Vector<Float>,
        dofs: NonZeroUsize,
        delta_t: Float,
    ) -> Self {
        let mut state = FactorState::new(measurement, strength, dofs, 2); // Dynamic factors have 2 neighbors
        let dynamic_factor = DynamicFactor::new(&mut state, delta_t);
        let kind = FactorKind::Dynamic(dynamic_factor);
        Self::new(state, kind)
    }

    // TODO: need to store the id of the variable in the other robots factorgraph,
    // so we can visualize it with graphviz
    pub fn new_interrobot_factor(
        strength: Float,
        measurement: Vector<Float>,
        dofs: NonZeroUsize,
        safety_radius: StrictlyPositiveFinite<Float>,
        connection: InterRobotConnection,
    ) -> Result<Self, InterRobotFactorError> {
        let state = FactorState::new(measurement, strength, dofs, 2); // Interrobot factors have 2 neighbors
        let interrobot_factor = InterRobotFactor::new(safety_radius.get(), connection)?;
        let kind = FactorKind::InterRobot(interrobot_factor);

        Ok(Self::new(state, kind))
    }

    pub fn new_pose_factor() -> Self {
        todo!()
    }

    pub fn new_obstacle_factor(
        strength: Float,
        measurement: Vector<Float>,
        dofs: NonZeroUsize,
        obstacle_sdf: &'static Image,
        world_size: Float,
    ) -> Self {
        let state = FactorState::new(measurement, strength, dofs, 1); // Obstacle factors have 1 neighbor
                                                                      // let obstacle_factor = ObstacleFactor::new(obstacle_sdf, world_size);
        let obstacle_factor = ObstacleFactor::new(obstacle_sdf, world_size);
        let kind = FactorKind::Obstacle(obstacle_factor);
        Self::new(state, kind)
    }

    #[inline(always)]
    pub fn variant(&self) -> &'static str {
        self.kind.name()
    }

    #[inline(always)]
    fn jacobian(&mut self, x: &Vector<Float>) -> Matrix<Float> {
        self.kind.jacobian(&self.state, x)
    }

    fn measure(&mut self, x: &Vector<Float>) -> Vector<Float> {
        self.state.cached_measurement = self.kind.measure(&self.state, x);
        self.state.cached_measurement.clone()
    }

    #[inline(always)]
    fn skip(&mut self) -> bool {
        self.kind.skip(&self.state)
    }

    pub fn set_node_index(&mut self, node_index: NodeIndex) {
        if self.node_index.is_some() {
            panic!("The node index is already set");
        }
        self.node_index = Some(node_index);
    }

    pub fn get_node_index(&self) -> NodeIndex {
        if self.node_index.is_none() {
            panic!("The node index has not been set");
        }
        self.node_index.expect("I checked it was there 3 lines ago")
    }

    pub fn receive_message_from(&mut self, from: VariableId, message: Message) {
        if message.is_empty() {
            // warn!("received an empty message from {:?}", from);
        }
        let _ = self.inbox.insert(from, message);
    }

    #[inline(always)]
    pub fn read_message_from(&mut self, from: VariableId) -> Option<&Message> {
        self.inbox.get(&from)
    }

    fn residual(&self) -> Vector<Float> {
        &self.state.initial_measurement - &self.state.cached_measurement
    }

    pub fn update(&mut self) -> MessagesToVariables {
        let dofs = 4;
        debug_assert_eq!(
            self.state.linearisation_point.len(),
            dofs * self.inbox.len()
        );
        let zero_mean = Vector::<Float>::zeros(dofs);
        for (i, (_, message)) in self.inbox.iter().enumerate() {
            let mean = message.mean().unwrap_or(&zero_mean);
            self.state
                .linearisation_point
                .slice_mut(s![i * dofs..(i + 1) * dofs])
                .assign(mean);
        }

        if self.skip() {
            let messages = self
                .inbox
                .iter()
                .map(|(variable_id, _)| (*variable_id, Message::empty(dofs)))
                .collect::<HashMap<_, _>>();
            return messages;
        }

        let _ = self.measure(&self.state.linearisation_point.clone());
        let jacobian = self.jacobian(&self.state.linearisation_point.clone());

        let factor_lambda_potential = jacobian
            .t()
            .dot(&self.state.measurement_precision)
            .dot(&jacobian);
        let factor_eta_potential = jacobian
            .t()
            .dot(&self.state.measurement_precision)
            .dot(&(jacobian.dot(&self.state.linearisation_point) + self.residual()));

        // pretty_print_vector!(&factor_eta_potential);

        self.state.initialized = true;

        let mut marginalisation_idx = 0;
        let mut messages = MessagesToVariables::with_capacity(self.inbox.len());

        let zero_precision = Matrix::<Float>::zeros((dofs, dofs));

        let color_code = match self.kind {
            FactorKind::Pose(_) => MAGENTA,
            FactorKind::InterRobot(_) => GREEN,
            FactorKind::Dynamic(_) => BLUE,
            FactorKind::Obstacle(_) => RED,
        };
        // println!(
        //     "{}{}{} UPDATE",
        //     color_code,
        //     self.node_index.unwrap().index(),
        //     RESET
        // );
        for variable_id in self.inbox.keys() {
            let mut factor_eta = factor_eta_potential.clone();
            let mut factor_lambda = factor_lambda_potential.clone();

            for (j, (other_variable_id, other_message)) in self.inbox.iter().enumerate() {
                // Do not aggregate data from the variable we're sending to
                if other_variable_id == variable_id {
                    continue;
                }

                let message_eta = other_message
                    .information_vector()
                    .expect("it better be there");

                // println!("{}{}{} eta = ", YELLOW, variable_id.global_id(), RESET);
                // pretty_print_vector!(message_eta);

                let message_precision = other_message.precision_matrix().unwrap_or(&zero_precision);

                factor_eta
                    .slice_mut(s![j * dofs..(j + 1) * dofs])
                    .add_assign(message_eta);
                factor_lambda
                    .slice_mut(s![j * dofs..(j + 1) * dofs, j * dofs..(j + 1) * dofs])
                    .add_assign(message_precision);
            }

            // pretty_print_vector!(&factor_eta);

            let message =
                marginalise_factor_distance(factor_eta, factor_lambda, dofs, marginalisation_idx)
                    .expect("marginalise_factor_distance should not fail");
            messages.insert(*variable_id, message);
            marginalisation_idx += dofs;
        }

        // messages.iter().for_each(|(variable_id, message)| {
        //     pretty_print_message!(
        //         match self.kind {
        //             FactorKind::Pose(_) => FactorId::new_pose(
        //                 variable_id.get_factor_graph_id(),
        //                 self.node_index.unwrap().into()
        //             ),
        //             FactorKind::InterRobot(_) => FactorId::new_interrobot(
        //                 variable_id.get_factor_graph_id(),
        //                 self.node_index.unwrap().into()
        //             ),
        //             FactorKind::Dynamic(_) => FactorId::new_dynamic(
        //                 variable_id.get_factor_graph_id(),
        //                 self.node_index.unwrap().into()
        //             ),
        //             FactorKind::Obstacle(_) => FactorId::new_obstacle(
        //                 variable_id.get_factor_graph_id(),
        //                 self.node_index.unwrap().into()
        //             ),
        //         },
        //         variable_id,
        //         self.kind.name()
        //     );
        //     pretty_print_vector!(message.information_vector().unwrap());
        //     pretty_print_matrix!(message.precision_matrix().unwrap());
        //     pretty_print_vector!(message.mean().unwrap());
        // });

        messages
    }
}

impl FactorGraphNode for Factor {
    fn remove_connection_to(
        &mut self,
        factorgraph_id: super::factorgraph::FactorGraphId,
    ) -> Result<(), super::factorgraph::RemoveConnectionToError> {
        let connections_before = self.inbox.len();
        self.inbox
            .retain(|variable_id, v| variable_id.factorgraph_id != factorgraph_id);
        let connections_after = self.inbox.len();

        let no_connections_removed = connections_before == connections_after;
        if no_connections_removed {
            Err(super::factorgraph::RemoveConnectionToError)
        } else {
            Ok(())
        }
    }
}
