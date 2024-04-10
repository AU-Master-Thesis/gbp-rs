use bevy::prelude::*;
use bevy_mod_picking::prelude::*;
use gbp_linalg::{pretty_print_matrix, pretty_print_vector};
use itertools::Itertools;

use super::{super::FactorGraph, RobotTracker};
use crate::{
    asset_loader::SceneAssets,
    config::{Config, DrawSetting},
    // factorgraph,
    planner::{
        robot::{DespawnRobotEvent, SpawnRobotEvent},
        RobotState,
    },
    pretty_print_title,
    theme::{self, CatppuccinTheme, ColorAssociation, ColorFromCatppuccinColourExt},
    ui::DrawSettingsEvent,
};

pub struct FactorGraphVisualiserPlugin;

impl Plugin for FactorGraphVisualiserPlugin {
    fn build(&self, app: &mut App) {
        app
            .add_event::<VariableClickEvent>()
            .add_systems(
            Update,
            (
                // init_factorgraphs,
                update_factorgraphs,
                show_or_hide_factorgraphs,
                draw_lines_between_variables.run_if(draw_predicted_trajectories_is_enabled),
                remove_rendered_factorgraph_when_robot_despawns,
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
    mut despawn_robot_event: EventReader<DespawnRobotEvent>,
) {
    for DespawnRobotEvent(robot_id) in despawn_robot_event.read() {
        // info!(
        //     "received DespawnRobotEvent({:?}), despawning the robots factorgraph
        // visualizer \      entities",
        //     robot_id
        // );
        for (entity, tracker) in query.iter() {
            if tracker.robot_id == *robot_id {
                if let Some(mut entitycommands) = commands.get_entity(entity) {
                    entitycommands.despawn();
                } else {
                }
            }
        }
    }
}

/// A **Bevy** [`Component`] to mark an entity as a visualised _factor graph_
#[derive(Component)]
pub struct VariableVisualiser;

#[derive(Component)]
pub struct DynamicFactorVisualiser;

#[derive(Event)]
struct VariableClickEvent(pub Entity);

impl VariableClickEvent {
    #[inline]
    pub fn target(&self) -> Entity {
        self.0
    }
}

impl From<ListenerInput<Pointer<Click>>> for VariableClickEvent {
    #[inline]
    fn from(value: ListenerInput<Pointer<Click>>) -> Self {
        Self(value.target)
    }
}

fn on_variable_clicked(
    mut variable_click_event: EventReader<VariableClickEvent>,
    query_robottracker: Query<&RobotTracker, With<VariableVisualiser>>,
    query_factorgraph: Query<&FactorGraph, With<RobotState>>,
) {
    for VariableClickEvent(entity) in variable_click_event.read() {
        let Some(variable) = query_robottracker
            .get(*entity)
            .inspect(|tracker| {
                info!(
                    "clicked variable mesh: {:?}, associated with robot: {:?}",
                    entity, tracker
                );
            })
            .and_then(|tracker| {
                query_factorgraph
                    .get(tracker.robot_id)
                    .map(|graph| (tracker.variable_index, graph))
            })
            .ok()
            .and_then(|(variable_index, factorgraph)| factorgraph.nth_variable(variable_index))
            .map(|(_, variable)| variable)
        else {
            error!("the clicked variable mesh is not associated with any existing factorgraph!");
            return;
        };

        pretty_print_vector!(&variable.belief.eta);
        pretty_print_matrix!(&variable.belief.lambda);
        pretty_print_vector!(&variable.belief.mu);
    }
}

/// A **Bevy** [`Update`] system
/// Initialises each new [`FactorGraph`] component to have a matching
/// [`PbrBundle`] and [`FactorGraphVisualiser`] component I.e. if the
/// [`FactorGraph`] component already has a [`FactorGraphVisualiser`], it will
/// be ignored
fn create_factorgraph_visualizer(
    mut commands: Commands,
    mut spawn_robot_event: EventReader<SpawnRobotEvent>,
    mut matierals: ResMut<Assets<StandardMaterial>>,
    query: Query<(Entity, &FactorGraph, &ColorAssociation), With<RobotState>>,
    config: Res<Config>,
    scene_assets: Res<SceneAssets>,
    theme: Res<CatppuccinTheme>,
) {
    for SpawnRobotEvent(robot_id) in spawn_robot_event.read() {
        let Some((_, factorgraph, color_association)) =
            query.iter().find(|(entity, _, _)| entity == robot_id)
        else {
            error!(
                "should not happen, a factorgraph should be attached to the newly spawned robot"
            );
            continue;
        };

        for (i, (index, variable)) in factorgraph.variables().enumerate() {
            let [x, y] = variable.estimated_position();
            let transform = Vec3::new(x as f32, config.visualisation.height.objects, y as f32);
            let robottracker = RobotTracker {
                robot_id: *robot_id,
                variable_index: index.into(),
                order: i,
            };

            debug!(
                "initialising factor graph visualiser: {:?} with tf: {:?}",
                robottracker, transform
            );
            commands.spawn((
                robottracker,
                VariableVisualiser,
                PickableBundle::default(),
                On::<Pointer<Click>>::send_event::<VariableClickEvent>(),
                PbrBundle {
                    mesh: scene_assets.meshes.variable.clone(),
                    // material: scene_assets.materials.variable.clone(),
                    material: matierals.add(StandardMaterial {
                        base_color: Color::from_catppuccin_colour(
                            theme.get_display_colour(&color_association.name),
                        ),
                        ..Default::default()
                    }),
                    transform: Transform::from_translation(transform),
                    visibility: match config.visualisation.draw.predicted_trajectories {
                        true => Visibility::Visible,
                        false => Visibility::Hidden,
                    },
                    ..Default::default()
                },
            ));
        }

        // for ((_, v1), (_, v2)) in factorgraph.variables().tuple_windows() {
        //
        // }
    }
}

/// A **Bevy** [`Update`] system
/// Updates the [`Transform`]s of all [`FactorGraphVisualiser`] entities
/// Done by cross-referencing with the [`FactorGraph`] components
/// that have matching [`Entity`] with the `RobotTracker.robot_id`
/// and variables in the [`FactorGraph`] that have matching
/// `RobotTracker.variable_id`
fn update_factorgraphs(
    mut query_tracker: Query<(&RobotTracker, &mut Transform), With<VariableVisualiser>>,
    query_factorgraph: Query<(Entity, &FactorGraph)>,
    config: Res<Config>,
) {
    // Update the `RobotTracker` components
    for (tracker, mut transform) in query_tracker.iter_mut() {
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
                    Vec3::new(x as f32, config.visualisation.height.objects, y as f32);
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
            for mut visibility in query.iter_mut() {
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
fn draw_predicted_trajectories_is_enabled(config: Res<Config>) -> bool {
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

    for (entity, color_association) in query_factorgraphs.iter() {
        // PERF: reuse the same vector, as all factorgraphs have the same variables
        let positions = query_variables
            .iter()
            .filter(|(tracker, _)| tracker.robot_id == entity)
            .sorted_by(|(a, _), (b, _)| a.order.cmp(&b.order))
            .rev()
            .map(|(_, t)| t.translation)
            .collect::<Vec<Vec3>>();

        for window in positions.windows(2) {
            let start = window[0];
            let end = window[1];
            gizmos.line(
                start,
                end,
                Color::from_catppuccin_colour(theme.get_display_colour(&color_association.name)),
            );
        }
    }
}
