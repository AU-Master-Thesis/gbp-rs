use bevy::{
    prelude::*,
    render::{mesh::Indices, render_resource::PrimitiveTopology},
};
use bevy_infinite_grid::{InfiniteGridBundle, InfiniteGridPlugin, InfiniteGridSettings};
use catppuccin::Flavour;

use crate::asset_loader::SceneAssets;

pub struct EnvironmentPlugin;

impl Plugin for EnvironmentPlugin {
    fn build(&self, app: &mut App) {
        let (r, g, b) = Flavour::Macchiato.base().into();
        app.insert_resource(ClearColor(Color::rgb_u8(r, g, b)))
            .insert_resource(AmbientLight {
                color: Color::default(),
                brightness: 0.5,
            })
            .add_state::<HeightMapState>()
            .add_plugins(InfiniteGridPlugin)
            .add_systems(Startup, (infinite_grid, lighting));
        // .add_systems(Update, obstacles.run_if(environment_png_is_loaded));
    }
}

fn infinite_grid(mut commands: Commands) {
    commands.spawn(InfiniteGridBundle {
        settings: InfiniteGridSettings {
            // shadow_color: None,
            major_line_color: {
                let (r, g, b) = Flavour::Macchiato.crust().into();
                Color::rgba_u8(r, g, b, (0.5 * 255.0) as u8)
            },
            minor_line_color: {
                let (r, g, b) = Flavour::Macchiato.crust().into();
                Color::rgba_u8(r, g, b, (0.25 * 255.0) as u8)
            },
            x_axis_color: {
                let (r, g, b) = Flavour::Macchiato.maroon().into();
                Color::rgba_u8(r, g, b, (0.1 * 255.0) as u8)
            },
            z_axis_color: {
                let (r, g, b) = Flavour::Macchiato.blue().into();
                Color::rgba_u8(r, g, b, (0.1 * 255.0) as u8)
            },
            ..default()
        },
        ..default()
    });
}

fn lighting(mut commands: Commands) {
    commands.spawn(DirectionalLightBundle {
        transform: Transform::from_translation(Vec3::X * 15.0 + Vec3::Z * 20.0)
            .looking_at(Vec3::ZERO, Vec3::Z),
        ..default()
    });
}

#[derive(Debug, Clone, Copy, Default, Eq, PartialEq, Hash, States)]
pub enum HeightMapState {
    #[default]
    Waiting,
    Generated,
}

fn environment_png_is_loaded(
    state: Res<State<HeightMapState>>,
    scene_assets: Res<SceneAssets>,
    image_assets: Res<Assets<Image>>,
) -> bool {
    if let Some(_) = image_assets.get(scene_assets.obstacle_image_raw.clone()) {
        return match state.get() {
            HeightMapState::Waiting => true,
            _ => false,
        };
    }
    false
}

fn obstacles(
    mut commands: Commands,
    scene_assets: Res<SceneAssets>,
    image_assets: Res<Assets<Image>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut next_state: ResMut<NextState<HeightMapState>>,
) {
    if let Some(image) = image_assets.get(scene_assets.obstacle_image_raw.clone()) {
        next_state.set(HeightMapState::Generated);

        let width = image.texture_descriptor.size.width as usize;
        let height = image.texture_descriptor.size.height as usize;
        let bytes_per_pixel =
            image.texture_descriptor.format.block_dimensions().0 as usize;
        let channels = 4;

        let vertices_count = width * height;
        let triangle_count = (width - 1) * (height - 1) * 6;
        let extent = 100.0;
        let intensity = 0.5;

        info!("image.texture_descriptor.size.width: {}", width);
        info!("image.texture_descriptor.size.height: {}", height);
        info!("image.data.len(): {}", image.data.len());
        info!("bytes_per_pixel: {}", bytes_per_pixel);
        info!(
            "image.data.len() / bytes_per_pixel: {}",
            image.data.len() / bytes_per_pixel
        );
        info!("vertices_count: {}", vertices_count);
        info!("triangle_count: {}", triangle_count);

        let mut heightmap = Vec::<f32>::with_capacity(vertices_count);
        for w in 0..width {
            for h in 0..height {
                // heightmap.push((w + h) as f32);
                // heightmap.push(0.0);
                heightmap
                    .push(1.0 - image.data[(w * height + h) * channels] as f32 / 255.0);
            }
        }

        info!("heightmap.len(): {}", heightmap.len());

        // Defining vertices.
        let mut positions: Vec<[f32; 3]> = Vec::with_capacity(vertices_count);
        let mut uvs: Vec<[f32; 2]> = Vec::with_capacity(vertices_count);

        for d in 0..width {
            for w in 0..height {
                let (w_f32, d_f32) = (w as f32, d as f32);

                let pos = [
                    (w_f32 - width as f32 / 2.) * extent as f32 / width as f32,
                    heightmap[d * width + w] * intensity,
                    (d_f32 - height as f32 / 2.) * extent as f32 / height as f32,
                ];
                positions.push(pos);
                uvs.push([w_f32 / width as f32, d_f32 / height as f32]);
            }
        }

        // Defining triangles.
        let mut triangles: Vec<u32> = Vec::with_capacity(triangle_count);

        for d in 0..(height - 2) as u32 {
            for w in 0..(width - 2) as u32 {
                // First tringle
                triangles.push((d * (width as u32 + 1)) + w);
                triangles.push(((d + 1) * (width as u32 + 1)) + w);
                triangles.push(((d + 1) * (width as u32 + 1)) + w + 1);
                // Second triangle
                triangles.push((d * (width as u32 + 1)) + w);
                triangles.push(((d + 1) * (width as u32 + 1)) + w + 1);
                triangles.push((d * (width as u32 + 1)) + w + 1);
            }
        }

        let mut mesh = Mesh::new(PrimitiveTopology::TriangleList);
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
        // mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
        mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
        mesh.set_indices(Some(Indices::U32(triangles)));
        mesh.duplicate_vertices();
        mesh.compute_flat_normals();

        let material_handle = materials.add(StandardMaterial {
            base_color_texture: Some(scene_assets.obstacle_image_raw.clone()),
            // base_color: Color::rgb(0.5, 0.5, 0.85),
            ..default()
        });

        commands.spawn(PbrBundle {
            mesh: meshes.add(mesh),
            material: material_handle,
            ..default()
        });
    }
}
