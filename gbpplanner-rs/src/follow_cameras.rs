use ::bevy::prelude::*;

use crate::movement::{AngularVelocity, Local, Orbit, OrbitMovementBundle, Velocity};

pub struct FollowCamerasPlugin;

impl Plugin for FollowCamerasPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(PostStartup, add_follow_cameras)
            .add_systems(Update, move_cameras);
    }
}

#[derive(Component)]
pub struct PID {
    pub p: f32,
    pub i: f32,
    pub d: f32,
}

impl Default for PID {
    fn default() -> Self {
        Self {
            p: 1.0,
            i: 0.0,
            d: 0.0,
        }
    }
}

#[derive(Component)]
pub struct FollowCameraMe;

#[derive(Component)]
pub struct FollowCameraSettings {
    // pub smoothing: f32, // 0.0 = infinite smoothing, 1.0 = no smoothing
    pub target: Entity,
    pub offset: Vec3,
    pub pid: PID,
    // pub p_heading: f32,
}

impl FollowCameraSettings {
    pub fn new(target: Entity) -> Self {
        Self {
            // smoothing: 0.5,
            target,
            offset: Vec3::new(0.0, 5.0, 10.0),
            pid: PID {
                p: 6.0,
                ..Default::default()
            },
        }
    }
}

#[derive(Bundle)]
pub struct FollowCameraBundle {
    pub settings: FollowCameraSettings,
    pub movement: OrbitMovementBundle,
    pub velocity: Velocity,
    pub camera: Camera3dBundle,
}

impl FollowCameraBundle {
    fn new(entity: Entity) -> Self {
        let position = Vec3::new(0.0, 5.0, 10.0).normalize() * 10.0;
        Self {
            settings: FollowCameraSettings::new(entity),
            movement: OrbitMovementBundle::default(),
            velocity: Velocity::new(Vec3::ZERO),
            camera: Camera3dBundle {
                transform: Transform::from_translation(position)
                    .looking_at(Vec3::ZERO, Vec3::Y),
                camera: Camera {
                    is_active: false,
                    ..Default::default()
                },
                ..Default::default()
            },
        }
    }
}

fn add_follow_cameras(
    mut commands: Commands,
    query: Query<Entity, With<FollowCameraMe>>,
) {
    for entity in query.iter() {
        info!("Adding follow camera for entity: {:?}", entity);
        commands.spawn((FollowCameraBundle::new(entity), Local));
    }
}

fn move_cameras(
    // time: Res<Time>,
    mut query_cameras: Query<
        (
            &Transform,
            &FollowCameraSettings,
            &mut AngularVelocity,
            &mut Velocity,
            &mut Orbit,
        ),
        With<Camera>,
    >,
    query_targets: Query<
        (Entity, &Transform, &Velocity),
        (With<FollowCameraMe>, Without<Camera>),
    >,
) {
    for (
        camera_transform,
        follow_settings,
        mut angular_velocity,
        mut camera_velocity,
        mut orbit,
    ) in query_cameras.iter_mut()
    {
        for (target_entity, target_transform, target_velocity) in query_targets.iter() {
            if target_entity == follow_settings.target {
                // let target_position = target_transform.translation
                //     + target_transform.right() * follow_settings.offset.x
                //     + target_transform.forward() * follow_settings.offset.z
                //     + target_transform.up() * follow_settings.offset.y;

                // // camera_transform.translation = target_position
                // //     * follow_camera_bundle.smoothing
                // //     + camera_transform.translation
                // //         * (1.0 - follow_camera_bundle.smoothing);

                // let delta = target_position - camera_transform.translation;
                // let distance = delta.length();

                // if distance < std::f32::EPSILON {
                //     continue;
                // }

                // let speed = distance * follow_settings.pid.p;
                // let direction = delta.normalize_or_zero();
                // velocity.value = direction * speed;
                camera_velocity.value = target_velocity.value;

                let (target_yaw, ..) = target_transform.rotation.to_euler(EulerRot::YXZ);
                let (camera_yaw, ..) = camera_transform.rotation.to_euler(EulerRot::YXZ);
                let delta_yaw = target_yaw - camera_yaw;

                let mut delta_yaw = delta_yaw;
                if delta_yaw > std::f32::consts::PI {
                    delta_yaw -= std::f32::consts::PI * 2.0;
                } else if delta_yaw < -std::f32::consts::PI {
                    delta_yaw += std::f32::consts::PI * 2.0;
                }

                let speed = delta_yaw * follow_settings.pid.p;
                angular_velocity.value.x = speed;

                orbit.origin = target_transform.translation;
                // camera_transform.look_at(target_transform.translation, Vec3::Y);
            }
        }
    }
}
