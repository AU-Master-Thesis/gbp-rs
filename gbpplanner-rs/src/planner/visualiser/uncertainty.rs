use super::{super::FactorGraph, RobotTracker, Z_FIGHTING_OFFSET};
use crate::{asset_loader::SceneAssets, config::Config};

use bevy::prelude::*;

pub struct UncertaintyVisualiserPlugin;

impl Plugin for UncertaintyVisualiserPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Update,
            (
                init_uncertainty,
                update_uncertainty,
                show_or_hide_uncertainty,
            ),
        );
    }
}

/// A **Bevy** [`Component`] to mark an entity as visualised _uncertainty_ gaussian
#[derive(Component)]
pub struct UncertaintyVisualiser;

/// A **Bevy** [`Component`] to mark a robot that it has a corresponding `UncertaintyVis` entity
/// Useful for easy _exclusion_ in queries
#[derive(Component)]
pub struct HasUncertaintyVisualiser;

/// A **Bevy** [`Update`] system
/// Initialises each new [`FactorGraph`] components to have a matching 2D circle and [`UncertaintyVisualiser`] component
/// I.e. if the [`FactorGraph`] component already has a [`HasUncertaintyVisualiser`], it will be ignored
fn init_uncertainty(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    query: Query<(Entity, &FactorGraph), Without<HasUncertaintyVisualiser>>,
    scene_assets: Res<SceneAssets>,
    config: Res<Config>,
) {
    query.iter().for_each(|(entity, factorgraph)| {
        // Mark the robot with `HasUncertaintyVisualiser` to exclude next time
        commands.entity(entity).insert(HasUncertaintyVisualiser);

        factorgraph.variables().for_each(|v| {
            let mean = v.belief.mean();
            let transform = Vec3::new(
                mean[0] as f32,
                config.visualisation.height.objects - 2.0 * Z_FIGHTING_OFFSET, // just under the lines (z-fighting prevention)
                mean[1] as f32,
            );

            let covariance = v.belief.covariance();

            let mut attenable = true;
            let mesh = meshes.add(Ellipse::new(
                // pick `x` from the covariance diagonal, but cap it at 10.0
                if covariance.diag()[0] > 20.0 {
                    attenable = false;
                    config.visualisation.uncertainty.max_radius
                } else {
                    covariance.diag()[0] as f32
                },
                // pick `y` from the covariance diagonal, but cap it at 10.0
                if covariance.diag()[1] > 20.0 {
                    attenable = false;
                    config.visualisation.uncertainty.max_radius
                } else {
                    covariance.diag()[1] as f32
                },
            ));

            info!(
                "{:?}: Initialising uncertainty at {:?}, with covariance {:?}",
                entity, transform, covariance
            );

            // Spawn a `UncertaintyVisualiser` component with a corresponding 2D circle
            commands.spawn((
                RobotTracker::new(entity).with_variable_id(v.get_node_index().index()),
                UncertaintyVisualiser,
                PbrBundle {
                    mesh,
                    material: if attenable {
                        scene_assets.materials.uncertainty.clone()
                    } else {
                        scene_assets.materials.uncertainty_unattenable.clone()
                    },
                    transform: Transform::from_translation(transform)
                        .with_rotation(Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2)),
                    visibility: if config.visualisation.draw.uncertainty {
                        Visibility::Visible
                    } else {
                        Visibility::Hidden
                    },
                    ..Default::default()
                },
            ));
        });
    });
}

/// A **Bevy** [`Update`] system
/// Updates the [`Transform`]s of all [`UncertaintyVisualiser`] entities
/// Update the shape and potentially the material of the [`UncertaintyVisualiser`] entities, depending on how the covariance has changed
///
/// Done by cross-referencing with the [`FactorGraph`] components
/// that have matching [`Entity`] with the `RobotTracker.robot_id`
/// and variables in the [`FactorGraph`] that have matching `RobotTracker.variable_id`
fn update_uncertainty(
    mut tracker_query: Query<
        (
            &RobotTracker,
            &mut Transform,
            &mut Handle<Mesh>,
            &mut Handle<StandardMaterial>,
        ),
        With<UncertaintyVisualiser>,
    >,
    factorgraph_query: Query<(Entity, &FactorGraph)>,
    mut meshes: ResMut<Assets<Mesh>>,
    config: Res<Config>,
    scene_assets: Res<SceneAssets>,
) {
    // Update the `RobotTracker` components
    for (tracker, mut transform, mut mesh, mut material) in tracker_query.iter_mut() {
        for (entity, factorgraph) in factorgraph_query.iter() {
            // continue if we're not looking at the right robot
            if tracker.robot_id != entity {
                continue;
            }

            // else look through the variables
            for v in factorgraph.variables() {
                // continue if we're not looking at the right variable
                if v.get_node_index().index() != tracker.variable_id {
                    continue;
                }

                let mean = v.belief.mean();
                let covariance = v.belief.covariance();

                let mut attenable = true;
                let new_mesh = meshes.add(Ellipse::new(
                    // pick `x` from the covariance diagonal, but cap it at 10.0
                    if covariance.diag()[0] > 20.0 {
                        attenable = false;
                        config.visualisation.uncertainty.max_radius
                    } else {
                        covariance.diag()[0] as f32
                    },
                    // pick `y` from the covariance diagonal, but cap it at 10.0
                    if covariance.diag()[1] > 20.0 {
                        attenable = false;
                        config.visualisation.uncertainty.max_radius
                    } else {
                        covariance.diag()[1] as f32
                    },
                ));

                // info!("{:?}: Updating uncertainty at {:?}, with covariance {:?}", entity, transform, covariance);

                // else update the transform
                *transform = Transform::from_translation(Vec3::new(
                    mean[0] as f32,
                    config.visualisation.height.objects - 2.0 * Z_FIGHTING_OFFSET, // just under the lines (z-fighting prevention)
                    mean[1] as f32,
                ))
                .with_rotation(Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2));

                // update the mesh and material
                *mesh = new_mesh;
                *material = if attenable {
                    scene_assets.materials.uncertainty.clone()
                } else {
                    scene_assets.materials.uncertainty_unattenable.clone()
                };
            }
        }
    }
}

/// A **Bevy** [`Update`] system
/// Reads [`DrawSettingEvent`], where if `DrawSettingEvent.setting == DrawSetting::Uncertainty`
/// the boolean `DrawSettingEvent.value` will be used to set the visibility of the [`UncertaintyVisualiser`] entities
fn show_or_hide_uncertainty(
    mut query: Query<(&UncertaintyVisualiser, &mut Visibility)>,
    mut draw_setting_event: EventReader<crate::ui::DrawSettingsEvent>,
) {
    for event in draw_setting_event.read() {
        if matches!(event.setting, crate::config::DrawSetting::Uncertainty) {
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
