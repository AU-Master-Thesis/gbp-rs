use bevy::prelude::*;
use strum_macros::EnumIter;

mod camera;
mod general;
mod moveable_object;
mod ui;

pub use self::camera::CameraAction;
pub use self::general::GeneralAction;
pub use self::moveable_object::MoveableObjectAction;
pub use self::ui::UiAction;

use self::{
    camera::CameraInputPlugin, general::GeneralInputPlugin,
    moveable_object::MoveableObjectInputPlugin, ui::UiInputPlugin,
};

#[derive(Debug, EnumIter)]
pub enum InputAction {
    General(GeneralAction),
    Camera(CameraAction),
    MoveableObject(MoveableObjectAction),
    Ui(UiAction),
    Undefined,
}

impl Default for InputAction {
    fn default() -> Self {
        Self::Undefined
    }
}

impl ToString for InputAction {
    fn to_string(&self) -> String {
        match self {
            Self::Camera(_) => "Camera".to_string(),
            Self::General(_) => "General".to_string(),
            Self::MoveableObject(_) => "Moveable Object".to_string(),
            Self::Ui(_) => "UI".to_string(),
            Self::Undefined => "Undefined".to_string(),
        }
    }
}

pub struct InputPlugin;

impl Plugin for InputPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            CameraInputPlugin,
            MoveableObjectInputPlugin,
            GeneralInputPlugin,
            UiInputPlugin,
        ));
    }
}
