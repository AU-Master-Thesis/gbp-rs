use bevy::{
    prelude::*,
    render::{mesh::PrimitiveTopology, render_asset::RenderAssetUsages},
};
use itertools::Itertools;

use crate::{
    asset_loader::SceneAssets,
    config::{self, Config},
    theme::{CatppuccinTheme, ColorFromCatppuccinColourExt},
    ui,
};

use super::{robot::Waypoints, RobotId, RobotState};

/// A **Bevy** `Plugin` for visualising aspects of the planner
/// Includes visualising parts of the factor graph
pub struct VisualiserPlugin;

impl Plugin for VisualiserPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Update,
            (
                init_waypoints,
                update_waypoints,
                show_or_hide_waypoints,
                init_factorgraphs,
                update_factorgraphs,
                show_or_hide_factorgraphs,
                draw_lines,
                // init_communication_graph, // TODO: when [`Path`] is no updatable
                draw_communication_graph,
                // show_or_hide_communication_graph, // TODO: when [`Path`] is no updatable
            ),
        );
    }
}

/// A **Bevy** `Component` for keeping track of a robot
/// Keeps track of the `RobotId` and `Vec2` position
#[derive(Component)]
pub struct RobotTracker {
    pub robot_id: RobotId,
    pub variable_id: usize,
    pub order: usize,
}

impl RobotTracker {
    pub fn new(robot_id: RobotId) -> Self {
        Self {
            robot_id,
            variable_id: 0,
            order: 0,
        }
    }

    pub fn with_variable_id(mut self, id: usize) -> Self {
        self.variable_id = id;
        self
    }

    pub fn with_order(mut self, order: usize) -> Self {
        self.order = order;
        self
    }
}

/// A **Bevy** [`Component`] to mark an entity as a visualised waypoint
#[derive(Component)]
pub struct WaypointVisualiser;

/// A **Bevy** [`Component`] to mark an entity as a visualised factor graph
#[derive(Component)]
pub struct VariableVisualiser;

/// A **Bevy** [`Component`] to mark an entity as a visualised communication graph
#[derive(Component)]
pub struct CommunicationGraphVisualiser;

/// A **Bevy** [`Component`] to mark a robot that it has a corresponding `WaypointVis` entity
/// Useful for easy exclusion in queries
#[derive(Component)]
pub struct HasWaypointVisualiser;

/// A **Bevy** [`Component`] to mark a robot that it has a corresponding `FactorGraphVis` entity
/// Useful for easy exclusion in queries
#[derive(Component)]
pub struct HasFactorGraphVisualiser;

/// A **Bevy** marker [`Component`] for lines
/// Generally used to identify previously spawned lines,
/// so they can be updated or removed
#[derive(Component)]
pub struct Line;

/// A **Bevy** marker [`Component`] for a line segment
/// Generally used to identify previously spawned line segments,
/// so they can be updated or removed
#[derive(Component)]
pub struct LineSegment;

/// A **Bevy** [`Component`] for drawing a path or line
/// Contains a list of points and a width used to construct a mesh
#[derive(Debug, Clone)]
struct Path {
    points: Vec<Vec3>,
    width: f32,
}

impl Path {
    pub fn new(points: Vec<Vec3>) -> Self {
        Self { points, width: 0.1 }
    }

    pub fn with_width(mut self, width: f32) -> Self {
        self.width = width;
        self
    }
}

impl From<Path> for Mesh {
    fn from(line: Path) -> Self {
        let vertices = line.points.clone();
        let width = line.width;

        let mut left_vertices = Vec::<Vec3>::with_capacity(vertices.len());
        let mut right_vertices = Vec::<Vec3>::with_capacity(vertices.len());

        // add the first offset
        let (a, b) = (vertices[0], vertices[1]);
        let ab = (b - a).normalize();
        let n = Vec3::new(ab.z, ab.y, -ab.x);
        let left = a - n * width / 2.0;
        let right = a + n * width / 2.0;
        left_vertices.push(left);
        right_vertices.push(right);

        for window in vertices.windows(3) {
            let (a, b, c) = (window[0], window[1], window[2]);
            let ab = (b - a).normalize();
            let bc = (c - b).normalize();

            let angle = (std::f32::consts::PI - ab.dot(bc).acos()) / 2.0;
            let kinked_width = width / angle.sin();

            let n = {
                let sum = (ab + bc).normalize();
                Vec3::new(sum.z, sum.y, -sum.x)
            };
            let left = b - n * kinked_width / 2.0;
            let right = b + n * kinked_width / 2.0;

            left_vertices.push(left);
            right_vertices.push(right);
        }

        // add the last offset
        let (a, b) = (vertices[vertices.len() - 2], vertices[vertices.len() - 1]);
        let ab = (b - a).normalize();
        let n = Vec3::new(ab.z, ab.y, -ab.x);
        let left = b - n * width / 2.0;
        let right = b + n * width / 2.0;
        left_vertices.push(left);
        right_vertices.push(right);

        // collect all vertices
        let vertices: Vec<Vec3> = left_vertices
            .iter()
            .zip(right_vertices.iter())
            .flat_map(|(l, r)| [*r, *l])
            .collect();

        Mesh::new(
            PrimitiveTopology::TriangleStrip,
            RenderAssetUsages::MAIN_WORLD  | RenderAssetUsages::RENDER_WORLD
        )
        // Add the vertices positions as an attribute
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, vertices)
    }
}

/// A **Bevy** [`Update`] system
/// Initialises each new [`FactorGraph`] component to have a matching [`PbrBundle`] and [`FactorGraphVisualiser`] component
/// I.e. if the [`FactorGraph`] component already has a [`FactorGraphVisualiser`], it will be ignored
fn init_factorgraphs(
    mut commands: Commands,
    query: Query<(Entity, &super::FactorGraph), Without<HasFactorGraphVisualiser>>,
    scene_assets: Res<SceneAssets>,
    config: Res<Config>,
) {
    for (entity, factorgraph) in query.iter() {
        // Mark the robot with `HasFactorGraphVisualiser` to exclude next time
        commands.entity(entity).insert(HasFactorGraphVisualiser);

        factorgraph
            // .variables_ordered()
            .variables()
            .enumerate()
            .for_each(|(i, (index, v))| {
                let mean = v.belief.mean();
                let transform = Vec3::new(
                    mean[0] as f32,
                    config.visualisation.height.objects,
                    mean[1] as f32,
                );

                // info!("{:?}: Initialising variable at {:?}", entity, transform);

                // Spawn a `FactorGraphVisualiser` component with a corresponding `PbrBundle`
                commands.spawn((
                    RobotTracker::new(entity)
                        .with_variable_id(index.index())
                        .with_order(i),
                    VariableVisualiser,
                    PbrBundle {
                        mesh: scene_assets.meshes.variable.clone(),
                        material: scene_assets.materials.variable.clone(),
                        transform: Transform::from_translation(transform),
                        ..Default::default()
                    },
                ));
            });
    }
}

/// A **Bevy** [`Update`] system
/// Updates the [`Transform`]s of all [`FactorGraphVisualiser`] entities
/// Done by cross-referencing with the [`FactorGraph`] components
/// that have matching [`Entity`] with the `RobotTracker.robot_id`
/// and variables in the [`FactorGraph`] that have matching `RobotTracker.variable_id`
fn update_factorgraphs(
    mut tracker_query: Query<(&RobotTracker, &mut Transform), With<VariableVisualiser>>,
    factorgraph_query: Query<(Entity, &super::FactorGraph)>,
    config: Res<Config>,
) {
    // Update the `RobotTracker` components
    for (tracker, mut transform) in tracker_query.iter_mut() {
        for (entity, factorgraph) in factorgraph_query.iter() {
            // continue if we're not looking at the right robot
            if tracker.robot_id != entity {
                continue;
            }

            // else look through the variables
            for (index, v) in factorgraph.variables() {
                // continue if we're not looking at the right variable
                if index.index() != tracker.variable_id {
                    continue;
                }

                // info!("{:?}: Updating variable to {:?}", entity, v.belief.mean());

                // else update the transform
                let mean = v.belief.mean();
                transform.translation = Vec3::new(
                    mean[0] as f32,
                    config.visualisation.height.objects,
                    mean[1] as f32,
                );
            }
        }
    }
}

/// A **Bevy** [`Update`] system
/// Reads [`DrawSettingEvent`], where if `DrawSettingEvent.setting == DrawSetting::PredictedTrajectories`
/// the boolean `DrawSettingEvent.value` will be used to set the visibility of the [`VariableVisualiser`] entities
fn show_or_hide_factorgraphs(
    mut query: Query<(&VariableVisualiser, &mut Visibility)>,
    mut draw_setting_event: EventReader<ui::DrawSettingsEvent>,
) {
    for event in draw_setting_event.read() {
        if matches!(event.setting, config::DrawSetting::PredictedTrajectories) {
            for (_, mut visibility) in query.iter_mut() {
                if event.value {
                    *visibility = Visibility::Visible;
                } else {
                    *visibility = Visibility::Hidden;
                }
            }
        }
    }
}

/// A **Bevy** [`Update`] system
/// Draws lines between all variables in each factor graph
///
/// Despawns old lines, and spawns new lines
///
/// Queries variables by [`RobotTracker`] with the [`FactorGraphVisualiser`] component
/// as initialised by the `init_factorgraphs` system
/// -> Will return if this query is empty
fn draw_lines(
    mut commands: Commands,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut meshes: ResMut<Assets<Mesh>>,
    // should_i_draw_lines: Res<ShouldIDrawLines>,
    config: Res<Config>,
    query_variables: Query<(&RobotTracker, &Transform), With<VariableVisualiser>>,
    query_previous_lines: Query<Entity, With<Line>>,
    factorgraph_query: Query<Entity, With<super::FactorGraph>>,
    catppuccin_theme: Res<CatppuccinTheme>,
) {
    // If there are no variables visualised yet by the `init_factorgraphs` system, return
    if query_variables.iter().count() == 0 {
        return;
    }

    // Remove previous lines
    // TODO: Update lines instead of removing and re-adding
    for entity in query_previous_lines.iter() {
        commands.entity(entity).despawn();
    }

    // If we're not supposed to draw lines, return
    if !config.visualisation.draw.predicted_trajectories {
        return;
    }

    let line_material = materials.add(Color::from_catppuccin_colour(
        catppuccin_theme.flavour.text(),
    ));

    for entity in factorgraph_query.iter() {
        let positions = query_variables
            .iter()
            .filter(|(tracker, _)| tracker.robot_id == entity)
            .sorted_by(|(a, _), (b, _)| a.order.cmp(&b.order))
            .rev()
            .map(|(_, t)| t.translation)
            .collect::<Vec<Vec3>>();

        let line = Path::new(positions).with_width(0.2);

        commands.spawn((
            PbrBundle {
                mesh: meshes.add(Mesh::from(line)),
                material: line_material.clone(),
                ..Default::default()
            },
            Line,
        ));
    }
}

/// A **Bevy** [`Update`] system
/// Initialises each new [`Waypoints`] component to have a matching [`PbrBundle`] and [`RobotTracker`] component
/// I.e. if the [`Waypoints`] component already has a [`RobotTracker`], it will be ignored
fn init_waypoints(
    mut commands: Commands,
    query: Query<(Entity, &Waypoints), Without<HasWaypointVisualiser>>,
    scene_assets: Res<SceneAssets>,
    config: Res<Config>,
) {
    for (entity, waypoints) in query.iter() {
        // Mark the robot with `RobotHasTracker` to exclude next time
        commands.entity(entity).insert(HasWaypointVisualiser);

        if let Some(next_waypoint) = waypoints.0.front() {
            // info!("Next waypoint: {:?}", next_waypoint);

            let transform = Transform::from_translation(Vec3::new(
                next_waypoint.x,
                config.visualisation.height.objects,
                next_waypoint.y,
            ));
            info!("{:?}: Initialising waypoints at {:?}", entity, transform);

            // Spawn a `RobotTracker` component with a corresponding `PbrBundle`
            commands.spawn((
                WaypointVisualiser,
                RobotTracker::new(entity),
                PbrBundle {
                    mesh: scene_assets.meshes.waypoint.clone(),
                    material: scene_assets.materials.waypoint.clone(),
                    transform,
                    ..Default::default()
                },
            ));
        } else {
            info!("No waypoints for robot {:?}", entity);
        }
    }
}

/// A **Bevy** [`Update`] system
/// Updates the [`Transform`]s of all [`WaypointVisualiser`] entities
/// Done by cross-referencing [`Entity`] with the `RobotTracker.robot_id`
fn update_waypoints(
    mut tracker_query: Query<(&RobotTracker, &mut Transform), With<WaypointVisualiser>>,
    robots_query: Query<(Entity, &Waypoints)>,
    config: Res<Config>,
) {
    // Update the `RobotTracker` components
    for (tracker, mut transform) in tracker_query.iter_mut() {
        for (entity, waypoints) in robots_query.iter() {
            if let Some(next_waypoint) = waypoints.0.front() {
                if tracker.robot_id == entity {
                    // Update the `Transform` component
                    // to match the `Waypoints` component

                    // info!("{:?}: Updating waypoints to {:?}", entity, next_waypoint);
                    transform.translation = Vec3::new(
                        next_waypoint.x,
                        config.visualisation.height.objects,
                        next_waypoint.y,
                    );
                }
            } else {
                // info!("Robot {:?} has no more waypoints", tracker.robot_id);
            }
        }
    }
}

/// A **Bevy** [`Update`] system
/// Reads [`DrawSettingEvent`], where if `DrawSettingEvent.setting == DrawSetting::Waypoints`
/// the boolean `DrawSettingEvent.value` will be used to set the visibility of the [`WaypointVisualiser`] entities
fn show_or_hide_waypoints(
    mut query: Query<(&WaypointVisualiser, &mut Visibility)>,
    mut draw_setting_event: EventReader<ui::DrawSettingsEvent>,
) {
    for event in draw_setting_event.read() {
        if matches!(event.setting, config::DrawSetting::Waypoints) {
            for (_, mut visibility) in query.iter_mut() {
                if event.value {
                    *visibility = Visibility::Visible;
                } else {
                    *visibility = Visibility::Hidden;
                }
            }
        }
    }
}

/// A **Bevy** [`Update`] system
/// Draws the communication graph with a [`Path`], through a [`PbrBundle`] and [`CommunicationGraphVisualiser`] component
///
/// Draws a line segment between each robot and its neighbours
/// A robot is a neighbour if it is within the communication range `config.communication.range`
///
/// However, if the robot's comms are off `RobotState.interrobot_comms_active == false`, it will not draw a line segment
fn draw_communication_graph(
    mut commands: Commands,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut meshes: ResMut<Assets<Mesh>>,
    query_previous_line_segments: Query<Entity, With<LineSegment>>,
    robots_query: Query<(Entity, &RobotState, &Transform)>,
    config: Res<Config>,
    catppuccin_theme: Res<CatppuccinTheme>,
) {
    // If there are no robots, return
    if robots_query.iter().count() == 0 {
        return;
    }

    // Remove previous lines
    query_previous_line_segments.iter().for_each(|entity| {
        commands.entity(entity).despawn();
    });

    // If we're not supposed to draw the communication graph, return
    if !config.visualisation.draw.communication_graph {
        return;
    }

    let line_material = materials.add(Color::from_catppuccin_colour(
        catppuccin_theme.flavour.teal(),
    ));

    // TODO: Don't double-draw lines from and to the same two robots
    for (robot_id, robot_state, transform) in robots_query.iter() {
        if !robot_state.interrobot_comms_active {
            continue;
        }

        // Find all neighbour transforms within the communication range
        // but filter out all robots that do not have comms on
        let neighbours = robots_query
            .iter()
            .filter(|(other_robot_id, other_robot_state, _)| {
                robot_id != *other_robot_id && !other_robot_state.interrobot_comms_active
            })
            .filter_map(|(_, _, other_transform)| {
                let distance = transform.translation.distance(other_transform.translation);
                if distance < config.robot.communication.radius {
                    Some(other_transform.translation)
                } else {
                    None
                }
            })
            .collect::<Vec<Vec3>>();

        if neighbours.is_empty() {
            continue;
        }

        // Make a line for each neighbour
        for neighbour_transform in neighbours {
            let line = Path::new(vec![transform.translation, neighbour_transform]).with_width(0.2);
            commands.spawn((
                PbrBundle {
                    mesh: meshes.add(Mesh::from(line)),
                    material: line_material.clone(),
                    transform: Transform::default(),
                    ..Default::default()
                },
                LineSegment,
            ));
        }
    }
}
