use std::collections::{BTreeSet, VecDeque};

use crate::config::Config;
use crate::utils::get_variable_timesteps;

use super::factor::{Factor, InterRobotConnection};
use super::factorgraph::{FactorGraph, FactorIndex, MessagePassingMode, VariableIndex};
// use super::multivariate_normal::MultivariateNormal;
use super::variable::Variable;
use super::NodeIndex;
use bevy::prelude::*;
use gbp_linalg::{prelude::*, pretty_print_matrix};
use gbp_multivariate_normal::MultivariateNormal;
use ndarray::{array, concatenate, s, Axis};
use std::collections::HashMap;

pub struct RobotPlugin;

pub type RobotId = Entity;

// /// Sigma for Unary pose factor on current and horizon states
// /// from **gbpplanner** `Globals.h`
// const SIGMA_POSE_FIXED: f32 = 1e-6;

#[derive(Resource)]
pub struct VariableTimestepsResource {
    pub timesteps: Vec<u32>,
}

impl FromWorld for VariableTimestepsResource {
    fn from_world(world: &mut World) -> Self {
        let config = world.resource::<Config>();
        let lookahead_horizon = config.robot.planning_horizon / config.simulation.t0;

        #[allow(clippy::cast_possible_truncation)]
        Self {
            timesteps: get_variable_timesteps(
                lookahead_horizon as u32,
                config.gbp.lookahead_multiple as u32,
            ),
        }
    }
}

impl Plugin for RobotPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<VariableTimestepsResource>()
            .add_systems(
                FixedUpdate,
                (
                    update_robot_neighbours_system,
                    delete_interrobot_factors_system,
                    create_interrobot_factors_system,
                    update_failed_comms_system,
                    iterate_gbp_internal_system,
                    iterate_gbp_external_system,
                    update_prior_of_horizon_state_system,
                    update_prior_of_current_state_system,
                )
                    .chain()
                    .run_if(time_is_not_paused),
            );

        info!("built RobotPlugin, added:");
        eprintln!(" - Resource: VariableTimestepsResource");
        eprintln!(" - system:Update");
        eprintln!("   - update_robot_neighbours_system");
        eprintln!("   - delete_interrobot_factors_system");
        eprintln!("   - create_interrobot_factors_system");
        eprintln!("   - update_failed_comms_system");
        eprintln!("   - iterate_gbp_internal_system");
        eprintln!("   - iterate_gbp_external_system");
        eprintln!("   - update_prior_of_horizon_state_system");
        eprintln!("   - update_prior_of_current_state_system");
    }
}

#[derive(Debug, thiserror::Error)]
pub enum RobotInitError {
    #[error("No waypoints were provided")]
    NoWaypoints,
    #[error("No variable timesteps were provided")]
    NoVariableTimesteps,
}

/// A component to encapsulate the idea of a radius.
#[derive(Component, Debug)]
pub struct Radius(pub f32);

/// A waypoint is a position that the robot should move to.
#[allow(clippy::similar_names)]
#[derive(Component, Debug)]
pub struct Waypoints(pub VecDeque<Vec4>);
// pub struct Waypoints(pub VecDeque<Vec2>);

/// A robot's state, consisting of other robots within communication range,
/// and other robots that are connected via inter-robot factors.
#[derive(Component, Debug)]
pub struct RobotState {
    /// List of robot ids that are within the communication radius of this robot.
    /// called `neighbours_` in **gbpplanner**.
    /// TODO: maybe change to a BTreeSet
    // ids_of_robots_within_comms_range: Vec<RobotId>,
    pub ids_of_robots_within_comms_range: BTreeSet<RobotId>,
    /// List of robot ids that are currently connected via inter-robot factors to this robot
    /// called `connected_r_ids_` in **gbpplanner**.
    pub ids_of_robots_connected_with: BTreeSet<RobotId>,
    // pub ids_of_robots_connected_with: Vec<RobotId>,
    /// Flag for whether this factorgraph/robot communicates with other robots
    pub interrobot_comms_active: bool,
}

impl RobotState {
    pub fn new() -> Self {
        Self {
            ids_of_robots_within_comms_range: BTreeSet::new(),
            ids_of_robots_connected_with: BTreeSet::new(),
            interrobot_comms_active: true,
        }
    }
}

#[derive(Bundle, Debug)]
pub struct RobotBundle {
    /// The factor graph that the robot is part of, and uses to perform GBP message passing.
    pub factorgraph: FactorGraph,
    /// Radius of the robot.
    /// If the robot is not a perfect circle, then set radius to be the smallest circle that fully encompass the shape of the robot.
    /// **constraint**: > 0.0
    pub radius: Radius, // TODO: create new type that guarantees this constraint
    /// The current state of the robot
    pub state: RobotState,
    // pub transform: Transform,
    /// Waypoints used to instruct the robot to move to a specific position.
    /// A VecDeque is used to allow for efficient pop_front operations, and push_back operations.
    pub waypoints: Waypoints,
    // NOTE: Using the **Bevy** entity id as the robot id
    // pub id: RobotId,
    // NOTE: These are accessible as **Bevy** resources
    // obstacle_sdf: Option<Rc<image::RgbImage>>,
    // settings: &'a RobotSettings,
    // id_generator: Rc<RefCell<IdGenerator>>,
}

impl RobotBundle {
    #[must_use = "Constructor responsible for creating the robots factorgraph"]
    pub fn new(
        mut waypoints: VecDeque<Vec4>,
        variable_timesteps: &[u32],
        config: &Config,
        obstacle_sdf: &'static Image,
    ) -> Result<Self, RobotInitError> {
        if waypoints.is_empty() {
            return Err(RobotInitError::NoWaypoints);
        }

        if variable_timesteps.is_empty() {
            return Err(RobotInitError::NoVariableTimesteps);
        }

        // dbg!(&waypoints);

        let start = waypoints
            .pop_front()
            .expect("Waypoints has at least one element");
        let goal = waypoints
            .front()
            .expect("Waypoints has at least two elements");

        // Initialise the horizon in the direction of the goal, at a distance T_HORIZON * MAX_SPEED from the start.
        let start2goal = *goal - start;
        let horizon = start
            + f32::min(
                start2goal.length(),
                config.robot.planning_horizon * config.robot.max_speed,
            ) * start2goal.normalize();

        let ndofs = 4; // [x, y, x', y']

        let mut factorgraph = FactorGraph::new();

        let last_variable_timestep = *variable_timesteps
            .last()
            .expect("Know that variable_timesteps has at least one element");

        let n_variables = variable_timesteps.len();
        let mut variable_node_indices = Vec::with_capacity(n_variables);
        for (i, &variable_timestep) in variable_timesteps.iter().enumerate().take(n_variables) {
            // Set initial mean and covariance of variable interpolated between start and horizon
            let mean = start
                + (horizon - start) * (variable_timestep as f32 / last_variable_timestep as f32);

            let sigma = if i == 0 || i == n_variables - 1 {
                // Start and Horizon state variables should be 'fixed' during optimisation at a timestep
                // SIGMA_POSE_FIXED
                1e30
            } else {
                // 4e9
                Float::INFINITY
            };

            let precision_matrix = Matrix::<Float>::from_diag_elem(ndofs, sigma);
            pretty_print_matrix!(&precision_matrix);

            // let covariance = Matrix::<Float>::from_diag_elem(ndofs, sigma);
            // dbg!(&covariance);
            //
            // pretty_print_matrix!(&covariance);

            let mean = array![
                mean.x as Float,
                mean.y as Float,
                mean.z as Float,
                mean.w as Float
            ];
            // let information_vector = precision_matrix.dot(&mean);
            // let prior = MultivariateNormal::from_information_and_precision(
            //     information_vector,
            //     precision_matrix,
            // )
            // .expect("the precision matrix is nonsingular");

            // let prior = MultivariateNormal::from_mean_and_covariance(
            //     // array![mean.x as Float, mean.y as Float, 0.0, 0.0], // initial velocity (x', y') is zero
            //     array![
            //         mean.x as Float,
            //         mean.y as Float,
            //         mean.z as Float,
            //         mean.w as Float
            //     ], // initial velocity (x', y') is zero
            //     covariance,
            // )
            // .expect("the covariance is nonsingular");

            // dbg!(&prior);

            // let variable = Variable::new(prior, ndofs);
            let variable = Variable::new(mean, precision_matrix, ndofs);
            let variable_index = factorgraph.add_variable(variable);
            variable_node_indices.push(variable_index);
        }

        // Create Dynamics factors between variables
        for i in 0..variable_timesteps.len() - 1 {
            // T0 is the timestep between the current state and the first planned state.
            let delta_t =
                config.simulation.t0 * (variable_timesteps[i + 1] - variable_timesteps[i]) as f32;
            // dbg!(delta_t);
            // println!("delta_t = {} i = {}", delta_t, i);
            // dbg!(&variable_timesteps);
            let measurement = Vector::<Float>::zeros(config.robot.dofs);
            let dynamic_factor = Factor::new_dynamic_factor(
                config.gbp.sigma_factor_dynamics as Float,
                measurement,
                config.robot.dofs,
                delta_t as Float,
            );

            let factor_node_index = factorgraph.add_factor(dynamic_factor);
            let _ = factorgraph.add_edge(
                VariableIndex(variable_node_indices[i]),
                FactorIndex(factor_node_index),
            );
            let _ = factorgraph.add_edge(
                VariableIndex(variable_node_indices[i + 1]),
                FactorIndex(factor_node_index),
            );
        }

        // Create Obstacle factors for all variables excluding start, excluding horizon

        #[allow(clippy::needless_range_loop)]
        for i in 1..variable_timesteps.len() - 1 {
            let obstacle_factor = Factor::new_obstacle_factor(
                config.gbp.sigma_factor_obstacle as Float,
                array![0.0],
                config.robot.dofs,
                obstacle_sdf,
                config.simulation.world_size as Float,
            );

            let factor_node_index = factorgraph.add_factor(obstacle_factor);
            let _ = factorgraph.add_edge(
                VariableIndex(variable_node_indices[i]),
                FactorIndex(factor_node_index),
            );
        }

        // std::process::exit(0);

        // dbg!(&factorgraph);

        Ok(Self {
            factorgraph,
            radius: Radius(config.robot.radius),
            state: RobotState::new(),
            waypoints: Waypoints(waypoints),
        })
    }
}

/// Called `Simulator::calculateRobotNeighbours` in **gbpplanner**
fn update_robot_neighbours_system(
    // query: Query<(Entity, &Transform, &mut RobotState)>,
    robots: Query<(Entity, &Transform), With<RobotState>>,
    mut states: Query<(Entity, &Transform, &mut RobotState)>,
    config: Res<Config>,
) {
    // TODO: use kdtree to speed up, and to have something in the report
    for (entity_id, transform, mut robotstate) in states.iter_mut() {
        // TODO: maybe use clear() instead
        // unsafe {
        //     state.ids_of_robots_within_comms_range.set_len(0);
        // }
        // TODO: reuse memory of ids_of_robots_within_comms_range
        robotstate.ids_of_robots_within_comms_range = robots
            .iter()
            .filter_map(|(other_entity_id, other_transform)| {
                // Do not compute the distance to self
                if other_entity_id == entity_id
                    || config.robot.communication.radius
                        < transform.translation.distance(other_transform.translation)
                {
                    None
                } else {
                    Some(other_entity_id)
                }
            })
            .collect();
        // .for_each(|other_entity_id| {
        //     state.ids_of_robots_within_comms_range.push(other_entity_id)
        // });
    }
}

// FIXME: more that one interrobot is created in `create_interrobot_factors`
fn delete_interrobot_factors_system(mut query: Query<(Entity, &mut FactorGraph, &mut RobotState)>) {
    // the set of robots connected with will (possibly) be mutated
    // the robots factorgraph will (possibly) be mutated
    // the other robot with an interrobot factor connected will be mutated

    let mut interrobot_factors_to_delete: HashMap<Entity, Entity> = HashMap::new();
    for (entity, _, mut robotstate) in query.iter_mut() {
        let ids_of_robots_connected_with_outside_comms_range: BTreeSet<_> = robotstate
            .ids_of_robots_connected_with
            .difference(&robotstate.ids_of_robots_within_comms_range)
            .cloned()
            .collect();

        interrobot_factors_to_delete.extend(
            ids_of_robots_connected_with_outside_comms_range
                .iter()
                .map(|id| (entity, *id)),
        );

        let robotstate = robotstate.as_mut();
        for id in ids_of_robots_connected_with_outside_comms_range {
            robotstate.ids_of_robots_connected_with.remove(&id);
        }
    }

    for (a, b) in interrobot_factors_to_delete {
        // Will delete both interrobot factors as,
        // (a, b)
        // (a, c)
        // (b, a)
        // (c, a)
        // (b, d)
        // (d, b)
        // etc.
        // deletes a's interrobot factor connecting to b, a -> b
        // deletes b's interrobot factor connecting to a, b -> a
        // TODO: use par_iter_mut
        for (entity, mut graph, _) in query.iter_mut() {
            if entity != a {
                continue;
            }

            if let Err(err) = graph.as_mut().delete_interrobot_factor_connected_to(b) {
                error!(
                    "Could not delete interrobot factor between {:?} -> {:?}, with error msg: {}",
                    a, b, err
                );
            }
        }
    }
}

fn create_interrobot_factors_system(
    mut query: Query<(Entity, &mut FactorGraph, &mut RobotState)>,
    config: Res<Config>,
    variable_timesteps: Res<VariableTimestepsResource>,
) {
    // a mapping between a robot and the other robots it should create a interrobot factor to
    // e.g:
    // {a -> [b, c, d], b -> [a, c], c -> [a, b], d -> [c]}
    let new_connections_to_establish: HashMap<RobotId, Vec<RobotId>> = query
        .iter()
        .map(|(entity, _, robotstate)| {
            let new_connections = robotstate
                .ids_of_robots_within_comms_range
                .difference(&robotstate.ids_of_robots_connected_with)
                .cloned()
                .collect::<Vec<_>>();

            (entity, new_connections)
        })
        .collect();

    let number_of_variables = variable_timesteps.timesteps.len();

    let variable_indices_of_each_factorgraph: HashMap<RobotId, Vec<NodeIndex>> = query
        .iter()
        .map(|(robot_id, factorgraph, _)| {
            let variable_indices = factorgraph
                .variable_indices_ordered_by_creation(1..number_of_variables)
                .expect("the factorgraph has up to `n_variables` variables");
            (robot_id, variable_indices)
        })
        .collect();

    for (robot_id, mut factorgraph, mut robotstate) in query.iter_mut() {
        for other_robot_id in new_connections_to_establish
            .get(&robot_id)
            .expect("the key is in the map")
        {
            let other_variable_indices = variable_indices_of_each_factorgraph
                .get(other_robot_id)
                .expect("the key is in the map");
            for i in 1..number_of_variables {
                // TODO: do not hardcode
                let dofs = 4;
                let z = Vector::<Float>::zeros(dofs);
                let eps = 0.2 * config.robot.radius;
                let safety_radius = 2.0 * config.robot.radius + eps;
                let connection = InterRobotConnection {
                    id_of_robot_connected_with: *other_robot_id,
                    index_of_connected_variable_in_other_robots_factorgraph: other_variable_indices
                        [i - 1],
                };
                // TODO: notify the variable that it is connected to another robot
                let interrobot_factor = Factor::new_interrobot_factor(
                    config.gbp.sigma_factor_interrobot as Float,
                    z,
                    dofs,
                    safety_radius as Float,
                    connection,
                );
                let factor_index = factorgraph.add_factor(interrobot_factor);
                let variable_index = factorgraph
                    .nth_variable_index(i)
                    .expect("there should be an i'th variable");
                factorgraph.add_edge(VariableIndex(variable_index), FactorIndex(factor_index));
            }

            robotstate
                .ids_of_robots_connected_with
                .insert(*other_robot_id);
        }
    }
}

/// Called `Simulator::setCommsFailure` in **gbpplanner**
fn update_failed_comms_system(mut query: Query<&mut RobotState>, config: Res<Config>) {
    for mut state in query.iter_mut() {
        state.interrobot_comms_active =
            config.robot.communication.failure_rate > rand::random::<f32>();
    }
}

macro_rules! iterate_gbp_impl {
    ($name:ident, $mode:expr) => {
        fn $name(
            mut query: Query<(Entity, &mut FactorGraph), With<RobotState>>,
            // config: Res<Config>,
        ) {
            // query
            //     .par_iter_mut()
            //     .for_each(|(robot_id, mut factorgraph)| {
            //         factorgraph.variable_iteration(robot_id, $mode);
            //     });
            // query
            //     .par_iter_mut()
            //     .for_each(|(robot_id, mut factorgraph)| {
            //         factorgraph.factor_iteration(robot_id, $mode);
            //     });
            for (robot_id, mut factorgraph) in query.iter_mut() {
                factorgraph.factor_iteration(robot_id, $mode);
            }
            for (robot_id, mut factorgraph) in query.iter_mut() {
                factorgraph.variable_iteration(robot_id, $mode);
            }
        }
    };
}

fn iterate_gbp_internal_system(mut query: Query<(Entity, &mut FactorGraph), With<RobotState>>) {
    let mode = MessagePassingMode::Internal;

    for (robot_id, mut factorgraph) in query.iter_mut() {
        factorgraph.factor_iteration(robot_id, mode);
    }
    for (robot_id, mut factorgraph) in query.iter_mut() {
        factorgraph.variable_iteration(robot_id, mode);
    }
}

fn iterate_gbp_external_system(mut query: Query<(Entity, &mut FactorGraph), With<RobotState>>) {
    let mode = MessagePassingMode::External;

    for (robot_id, mut factorgraph) in query.iter_mut() {
        factorgraph.factor_iteration(robot_id, mode);
    }
    for (robot_id, mut factorgraph) in query.iter_mut() {
        factorgraph.variable_iteration(robot_id, mode);
    }
}

// pub fn iterate_gbp_internal_system(
//     mut query: Query<(Entity, &mut FactorGraph), With<RobotState>>,
//     config: Res<Config>,
// ) {
//     for (robot_id, mut factorgraph) in query.iter_mut() {
//         factorgraph.variable_iteration(robot_id, MessagePassingMode::Internal);
//     }
//
//     for (robot_id, mut factorgraph) in query.iter_mut() {
//         factorgraph.factor_iteration(robot_id, MessagePassingMode::Internal);
//     }
//
//     // query
//     //     .par_iter_mut()
//     //     .for_each(|(robot_id, mut factorgraph)| {
//     //         factorgraph.variable_iteration(robot_id, MessagePassingMode::Internal);
//     //     });
//     // query
//     //     .par_iter_mut()
//     //     .for_each(|(robot_id, mut factorgraph)| {
//     //         factorgraph.factor_iteration(robot_id, MessagePassingMode::Internal);
//     //     });
// }

// iterate_gbp_impl!(iterate_gbp_internal_system, MessagePassingMode::Internal);
// iterate_gbp_impl!(iterate_gbp_external_system, MessagePassingMode::External);

// fn iterate_gbp_system(
//     mut query: Query<(Entity, &mut FactorGraph), With<RobotState>>,
//     config: Res<Config>,
// ) {

//     query.par_iter_mut().for_each(|(robot_id, mut factorgraph)| {
//         factorgraph.factor_iteration(robot_id, MessagePassingMode::Internal);
//     });
//     query.par_iter_mut().for_each(|(robot_id, mut factorgraph)| {
//         factorgraph.variable_iteration(robot_id, MessagePassingMode::Internal);
//     });
//     // for (robot_id, mut factorgraph) in query.par_iter_mut() {
//     //     factorgraph.factor_iteration(robot_id, MessagePassingMode::Internal);
//     // }
//     // for (robot_id, mut factorgraph) in query.iter_mut() {
//     //     factorgraph.variable_iteration(robot_id, MessagePassingMode::Internal);
//     // }
// }

// fn iterate_gbp(query: Query<&mut FactorGraph>, config: Res<Config>) {}

/// Called `Robot::updateHorizon` in **gbpplanner**
fn update_prior_of_horizon_state_system(
    mut query: Query<(Entity, &mut FactorGraph, &mut Waypoints), With<RobotState>>,
    config: Res<Config>,
    time: Res<Time>,
) {
    let delta_t = time.delta_seconds();
    for (entity, mut factorgraph, mut waypoints) in query.iter_mut() {
        let Some(current_waypoint) = waypoints
            .0
            .front()
            .map(|wp| array![wp.x as Float, wp.y as Float])
        else {
            warn_once!("robot {:?}, has reached its final waypoint", entity);
            continue;
        };

        let (index, new_mean, horizon2goal_dist) = {
            let (index, horizon_variable) = factorgraph
                .last_variable()
                .expect("factorgraph has a horizon variable");

            // let index = horizon_variable.node_index.unwrap();

            // let mean_of_horizon_variable = horizon_variable.belief.mean();
            let mean_of_horizon_variable = &horizon_variable.mu;
            debug_assert_eq!(mean_of_horizon_variable.len(), 4);
            // dbg!(&current_waypoint);
            // dbg!(&mean_of_horizon_variable);
            let estimated_position = mean_of_horizon_variable.slice(s![..2]); // the mean is a 4x1 vector with [x, y, x', y']
            dbg!(&estimated_position);
            dbg!(&current_waypoint);
            let horizon2goal_dir = current_waypoint - estimated_position;
            let horizon2goal_dist = horizon2goal_dir.euclidean_norm();

            let new_velocity = Float::min(config.robot.max_speed as Float, horizon2goal_dist)
                * horizon2goal_dir.normalized();
            let new_position = estimated_position.into_owned() + (&new_velocity * delta_t as Float);

            // Update horizon state with new pos and vel
            // horizon->mu_ << new_pos, new_vel;
            // horizon->change_variable_prior(horizon->mu_);
            let new_mean = concatenate![Axis(0), new_position, new_velocity];
            debug_assert_eq!(new_mean.len(), 4);
            (index, new_mean, horizon2goal_dist)
        };

        println!(
            "index = {:?}, horizon2goal_dist = {:?}",
            index, horizon2goal_dist
        );

        // TODO: cache the mean ...
        // horizon_variable.belief.mean()

        // TODO: create a separate method on Variable so we do not have to abuse the interface, and call it with a empty

        factorgraph.change_prior_of_variable(index, new_mean);
        // vector, and get an empty HashMap as a return value.
        // let _ = horizon_variable.change_prior(new_mean, vec![]);

        println!(
            "horizon2goal_dist = {:?}, config.robot.radius = {:?}",
            horizon2goal_dist, config.robot.radius
        );
        // NOTE: this is weird, we think
        let horizon_has_reached_waypoint = horizon2goal_dist < config.robot.radius as Float;
        if horizon_has_reached_waypoint && !waypoints.0.is_empty() {
            info!("robot {:?}, has reached its waypoint", entity);
            waypoints.0.pop_front();
        }
    }
}

/// run criteria if time is not paused
fn time_is_not_paused(time: Res<Time<Virtual>>) -> bool {
    !time.is_paused()
}

/// Called `Robot::updateCurrent` in **gbpplanner**
fn update_prior_of_current_state_system(
    mut query: Query<(&mut FactorGraph, &mut Transform), With<RobotState>>,
    config: Res<Config>,
    time: Res<Time>,
) {
    let scale = time.delta_seconds() / config.simulation.t0;

    for (mut factorgraph, mut transform) in query.iter_mut() {
        let (current_variable_index, mean_of_current_variable, increment) = {
            let (current_index, current_variable) = factorgraph
                .nth_variable(0)
                .expect("factorgraph should have a current variable");
            let (_, next_variable) = factorgraph
                .nth_variable(1)
                .expect("factorgraph should have a next variable");

            // let mean_of_current_variable = current_variable.belief.mean().clone();
            let mean_of_current_variable = current_variable.mu.clone();
            let increment = scale as Float * (&next_variable.mu - &mean_of_current_variable);

            (current_index, mean_of_current_variable, increment)
        };

        // dbg!(&increment);
        // TODO: create a separate method on Variable so we do not have to abuse the interface, and call it with a empty
        // vector, and get an empty HashMap as a return value.
        factorgraph.change_prior_of_variable(
            current_variable_index,
            mean_of_current_variable + &increment,
        );
        #[allow(clippy::cast_possible_truncation)]
        let increment = Vec3::new(increment[0] as f32, 0.0, increment[1] as f32);
        transform.translation += increment;
    }
}
