use bevy::prelude::*;
use bevy_mod_picking::prelude::*;
use gbp_config::{Config, DrawSetting};
use itertools::Itertools;

use super::RobotTracker;
use crate::{
    // asset_loader::SceneAssets,
    asset_loader::Meshes,
    bevy_utils::run_conditions::event_exists,
    factorgraph::{factor::Factor, factorgraph::VariableIndex, prelude::FactorGraph},
    input::DrawSettingsEvent,
    planner::{
        robot::{Radius, RobotDespawned, RobotSpawned},
        RobotConnections,
    },
    simulation_loader::{self, EndSimulation},
    theme::{CatppuccinTheme, ColorAssociation, ColorFromCatppuccinColourExt},
};

pub struct FactorGraphVisualiserPlugin;

impl Plugin for FactorGraphVisualiserPlugin {
    fn build(&self, app: &mut App) {
        app
            .add_event::<VariableClickedOn>()
            .add_systems(
            Update,
            (
                update_factorgraph_visualizers,
                show_or_hide_factorgraphs.run_if(event_exists::<DrawSettingsEvent>),
                draw_lines_between_variables.run_if(enabled),
                remove_rendered_factorgraph_when_robot_despawns,
                remove_rendered_factorgraphs.run_if(on_event::<EndSimulation>()),
                on_variable_clicked
            ),
        )
        // NOTE: this needs to be on the 'PostUpdate' schedule, otherwise there are timing issues between the creation of the robots factorgraph to reading its state.
        .add_systems(PostUpdate, create_factorgraph_visualizer);
    }
}

fn remove_rendered_factorgraph_when_robot_despawns(
    mut commands: Commands,
    query: Query<(Entity, &RobotTracker)>,
    mut evr_robot_despawned: EventReader<RobotDespawned>,
) {
    for RobotDespawned(robot_id) in evr_robot_despawned.read() {
        for (entity, tracker) in &query {
            if tracker.robot_id == *robot_id {
                if let Some(mut entitycommands) = commands.get_entity(entity) {
                    entitycommands.despawn();
                }
            }
        }
    }
}

fn remove_rendered_factorgraphs(
    mut commands: Commands,
    query: Query<Entity, With<VariableVisualiser>>,
) {
    for entity in &query {
        info!("despawning factorgraph visualiser: {:?}", entity);
        commands.entity(entity).despawn();
    }
}

/// A **Bevy** [`Component`] to mark an entity as a visualised _factor graph_
#[derive(Component)]
pub struct VariableVisualiser;

// #[derive(Component)]
// pub struct DynamicFactorVisualiser;

#[derive(Event)]
struct VariableClickedOn(pub Entity);

impl From<ListenerInput<Pointer<Click>>> for VariableClickedOn {
    #[inline]
    fn from(value: ListenerInput<Pointer<Click>>) -> Self {
        Self(value.target)
    }
}

fn on_variable_clicked(
    mut evr_variable_clicked_on: EventReader<VariableClickedOn>,
    q_robottracker: Query<&RobotTracker, With<VariableVisualiser>>,
    q_factorgraph: Query<&FactorGraph, With<RobotConnections>>,
    config: Res<Config>,
) {
    use colored::Colorize;

    for VariableClickedOn(entity) in evr_variable_clicked_on.read() {
        let Ok(tracker) = q_robottracker.get(*entity) else {
            error!("the clicked variable mesh is not associated with any existing factorgraph!");
            return;
        };

        let Ok(factorgraph) = q_factorgraph.get(tracker.robot_id) else {
            error!("the clicked variable mesh is not associated with any existing factorgraph!");
            return;
        };

        let Some((node_index, variable)) = factorgraph.nth_variable(tracker.variable_index) else {
            error!("the clicked variable mesh is not associated with any existing factorgraph!");
            return;
        };

        let Some(neighbours) = factorgraph.variable_neighbours(VariableIndex(*node_index)) else {
            error!("the clicked variable mesh is not associated with any existing factorgraph!");
            return;
        };

        let hr = "=".repeat(80);
        println!("{}: {}", "variable".magenta(), node_index.index());
        println!("  {}: {}", "factorgraph".cyan(), factorgraph.id().index());
        if config.debug.on_variable_clicked.variable {
            println!("  {}:", "belief".cyan());
            // println!("    {}:", "mean".red());
            // gbp_linalg::pretty_format_vector!()
            for line in
                gbp_linalg::pretty_format_vector!("mean", &variable.belief.mean, None).lines()
            {
                println!("    {}", line);
            }

            // println!(
            //    "    {}:",
            //    "variance".red(),
            //    // variable.belief.precision_matrix
            //);

            for line in gbp_linalg::pretty_format_matrix!(
                "precision",
                &variable.belief.precision_matrix,
                None
            )
            .lines()
            {
                println!("    {}", line);
            }

            // println!(
            //    "    {}:",
            //    "information vector".red(),
            //    // variable.belief.information_vector
            //);

            for line in gbp_linalg::pretty_format_vector!(
                "information",
                &variable.belief.information_vector,
                None
            )
            .lines()
            {
                println!("    {}", line);
            }
            // TODO: do not use debug print here
            // println!("{:#?}", variable);
        }

        if config.debug.on_variable_clicked.inbox {
            println!("  {}:", "inbox".cyan());
            dbg!(&variable.inbox);
        }

        println!("  {}:", "connected factors".cyan());
        for (i, neighbour) in neighbours.enumerate() {
            // println!("name: {}", neighbour.kind.name());
            use std::fmt::Write;

            use colored::Colorize;

            use crate::factorgraph::factor::FactorKind::{Dynamic, InterRobot, Obstacle, Tracking};
            // println!("factor[{i}]: {}", <neighbour.kind as &dyn Factor>::name());
            let [r, g, b] = neighbour.kind.color();
            println!("   - {}: {}", "neighbour_ix".red(), i);
            println!(
                "     {}: {}",
                "kind",
                neighbour.kind.name().truecolor(r, g, b)
            );
            // println!("factor[{i}]: {}", neighbour.kind.name().truecolor(r, g, b));

            let mut content = String::new();
            match neighbour.kind {
                InterRobot(ref interrobot) if config.debug.on_variable_clicked.interrobot => {
                    write!(&mut content, "{}", interrobot).unwrap();
                    // println!("{}", interrobot)
                }
                Dynamic(ref dynanic) if config.debug.on_variable_clicked.dynamic => {
                    write!(&mut content, "{}", dynanic).unwrap();
                    // println!("{}", dynanic)
                }
                Obstacle(ref obstacle) if config.debug.on_variable_clicked.obstacle => {
                    write!(&mut content, "{}", obstacle).unwrap();
                    // println!("{}", obstacle)
                }
                Tracking(ref tracking) if config.debug.on_variable_clicked.tracking => {
                    write!(&mut content, "{}", tracking).unwrap();
                    // println!("{}", tracking)
                }
                _ => {}
            }

            for line in content.lines() {
                println!("       {}", line);
            }
            if config.debug.on_variable_clicked.inbox {
                println!("  {}:", "inbox".cyan());
                dbg!(&neighbour.inbox);
            }

            println!("{}", hr);
        }
    }
}

/// A **Bevy** [`Update`] system
/// Initialises each new [`FactorGraph`] component to have a matching
/// [`PbrBundle`] and [`FactorGraphVisualiser`] component I.e. if the
/// [`FactorGraph`] component already has a [`FactorGraphVisualiser`], it will
/// be ignored
fn create_factorgraph_visualizer(
    mut commands: Commands,
    mut spawn_robot_event: EventReader<RobotSpawned>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    query: Query<(Entity, &FactorGraph, &ColorAssociation, &Radius), With<RobotConnections>>,
    config: Res<Config>,
    // scene_assets: Res<SceneAssets>,
    meshes: Res<Meshes>,
    theme: Res<CatppuccinTheme>,
) {
    for RobotSpawned(robot_id) in spawn_robot_event.read() {
        let Ok((_, factorgraph, color_association, radius)) = query.get(*robot_id) else {
            error!(
                "should not happen, a factorgraph should be attached to the newly spawned robot"
            );
            continue;
        };

        // let Some((_, factorgraph, color_association, radius)) =
        //    query.iter().find(|(entity, _, _)| entity == robot_id)
        // else {
        //    error!(
        //        "should not happen, a factorgraph should be attached to the newly
        // spawned robot"    );
        //    continue;
        //};

        for (i, (index, variable)) in factorgraph.variables().enumerate() {
            let [x, y] = variable.estimated_position();
            #[allow(clippy::cast_possible_truncation)]
            // let transform = Vec3::new(x as f32, -config.visualisation.height.objects, y as f32);
            let transform = Vec3::new(x as f32, -radius.0, y as f32);
            let robottracker = RobotTracker {
                robot_id: *robot_id,
                variable_index: index.index(),
                order: i,
            };

            debug!(
                "initialising factor graph visualiser: {:?} with tf: {:?}",
                robottracker, transform
            );
            commands.spawn((
                robottracker,
                simulation_loader::Reloadable,
                VariableVisualiser,
                PickableBundle::default(),
                On::<Pointer<Click>>::send_event::<VariableClickedOn>(),
                PbrBundle {
                    mesh: meshes.variable.clone(),
                    material: materials.add(StandardMaterial {
                        base_color: Color::from_catppuccin_colour(
                            theme.get_display_colour(&color_association.name),
                        ),
                        ..Default::default()
                    }),
                    transform: Transform::from_translation(transform),
                    visibility: if config.visualisation.draw.predicted_trajectories {
                        Visibility::Visible
                    } else {
                        Visibility::Hidden
                    },
                    ..Default::default()
                },
            ));
        }
    }
}

/// A **Bevy** [`Update`] system
/// Updates the [`Transform`]s of all [`FactorGraphVisualiser`] entities
/// Done by cross-referencing with the [`FactorGraph`] components
/// that have matching [`Entity`] with the `RobotTracker.robot_id`
/// and variables in the [`FactorGraph`] that have matching
/// `RobotTracker.variable_id`
#[allow(clippy::cast_possible_truncation)]
fn update_factorgraph_visualizers(
    mut query_tracker: Query<(&RobotTracker, &mut Transform), With<VariableVisualiser>>,
    query_factorgraph: Query<(Entity, &FactorGraph)>,
    config: Res<Config>,
) {
    // Update the `RobotTracker` components
    for (tracker, mut transform) in &mut query_tracker {
        for (entity, factorgraph) in query_factorgraph.iter() {
            // continue if we're not looking at the right robot
            let not_the_right_robot = tracker.robot_id != entity;
            if not_the_right_robot {
                continue;
            }

            // else look through the variables
            for (index, v) in factorgraph.variables() {
                // continue if we're not looking at the right variable
                if usize::from(index) != tracker.variable_index {
                    continue;
                }

                if !v.finite_covariance() {
                    continue;
                }

                // else update the transform
                let [x, y] = v.estimated_position();
                transform.translation =
                    Vec3::new(x as f32, -config.visualisation.height.objects, y as f32);
            }
        }
    }
}

/// A **Bevy** [`Update`] system
/// Reads [`DrawSettingEvent`], where if `DrawSettingEvent.setting ==
/// DrawSetting::PredictedTrajectories` the boolean `DrawSettingEvent.value`
/// will be used to set the visibility of the [`VariableVisualiser`] entities
fn show_or_hide_factorgraphs(
    mut query: Query<&mut Visibility, With<VariableVisualiser>>,
    mut draw_setting_event: EventReader<DrawSettingsEvent>,
) {
    for event in draw_setting_event.read() {
        if matches!(event.setting, DrawSetting::PredictedTrajectories) {
            for mut visibility in &mut query {
                *visibility = if event.draw {
                    Visibility::Visible
                } else {
                    Visibility::Hidden
                };
            }
        }
    }
}

#[inline]
fn enabled(config: Res<Config>) -> bool {
    config.visualisation.draw.predicted_trajectories
}

/// A **Bevy** [`Update`] system
/// Draws lines between all variables in each factor graph
///
/// Queries variables by [`RobotTracker`] with the [`FactorGraphVisualiser`]
/// component as initialised by the `init_factorgraphs` system
/// -> Will return if this query is empty
fn draw_lines_between_variables(
    mut gizmos: Gizmos,
    query_variables: Query<(&RobotTracker, &Transform), With<VariableVisualiser>>,
    query_factorgraphs: Query<(Entity, &ColorAssociation), With<FactorGraph>>,
    theme: Res<CatppuccinTheme>,
) {
    // let color = Color::from_catppuccin_colour(catppuccin_theme.text());

    for (entity, color_association) in &query_factorgraphs {
        // PERF: reuse the same vector, as all factorgraphs have the same variables
        let positions = query_variables
            .iter()
            .filter(|(tracker, _)| tracker.robot_id == entity)
            .sorted_by(|(a, _), (b, _)| a.order.cmp(&b.order))
            .rev()
            .map(|(_, t)| t.translation)
            .collect::<Vec<Vec3>>();

        let color =
            Color::from_catppuccin_colour(theme.get_display_colour(&color_association.name));
        for window in positions.windows(2) {
            let start = window[0];
            let end = window[1];
            gizmos.line(start, end, color);
        }
    }
}
