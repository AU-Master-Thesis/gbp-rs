use std::iter::Scan;

use bevy::{
    input::{gamepad, keyboard::KeyboardInput},
    prelude::*,
    window::WindowTheme,
};
use bevy_egui::{
    egui::{self, Color32, RichText, Visuals},
    EguiContexts, EguiPlugin,
};
use catppuccin::Flavour;
use color_eyre::owo_colors::OwoColorize;
use leafwing_input_manager::{
    axislike::{
        AxisType, DualAxis, MouseMotionAxisType, MouseWheelAxisType, SingleAxis, VirtualAxis,
        VirtualDPad,
    },
    buttonlike::{MouseMotionDirection, MouseWheelDirection},
    input_map::InputMap,
    user_input::{InputKind, Modifier, UserInput},
};
use strum::IntoEnumIterator;

use crate::theme::{CatppuccinTheme, CatppuccinThemeExt};
use crate::{
    input::{CameraAction, GeneralAction, InputAction, MoveableObjectAction, UiAction},
    theme::FromCatppuccinColourExt,
};
use std::fmt;

//  _     _ _______ _______  ______
//  |     | |______ |______ |_____/
//  |_____| ______| |______ |    \_
//
//  _____ __   _ _______ _______  ______ _______ _______ _______ _______
//    |   | \  |    |    |______ |_____/ |______ |_____| |       |______
//  __|__ |  \_|    |    |______ |    \_ |       |     | |_____  |______
//

pub struct EguiInterfacePlugin;

impl Plugin for EguiInterfacePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<OccupiedScreenSpace>()
            .init_resource::<UiState>()
            .add_plugins(EguiPlugin)
            .add_systems(Startup, configure_visuals_system)
            .add_systems(Update, ui_binding_panel);
    }
}

pub trait ToDisplayString {
    fn to_display_string(&self) -> String;
}

impl ToDisplayString for UserInput {
    fn to_display_string(&self) -> String {
        match self {
            UserInput::Single(input) => input.to_display_string(),
            UserInput::VirtualDPad(virtual_dpad) => virtual_dpad.to_display_string(),
            UserInput::VirtualAxis(virtual_axis) => virtual_axis.to_display_string(),
            UserInput::Chord(chord) => chord
                .iter()
                .map(|x| x.to_display_string())
                .collect::<Vec<String>>()
                .join(" + "),
        }
    }
}

impl ToDisplayString for VirtualDPad {
    fn to_display_string(&self) -> String {
        format!(
            "{}{}{}{}",
            self.up.to_display_string(),
            self.left.to_display_string(),
            self.down.to_display_string(),
            self.right.to_display_string()
        )
    }
}

impl ToDisplayString for VirtualAxis {
    fn to_display_string(&self) -> String {
        format!(
            "{}{}",
            self.positive.to_display_string(),
            self.negative.to_display_string()
        )
    }
}

impl ToDisplayString for InputKind {
    fn to_display_string(&self) -> String {
        match self {
            InputKind::GamepadButton(gamepad_button) => gamepad_button.to_display_string(),
            InputKind::SingleAxis(single_axis) => single_axis.to_display_string(),
            InputKind::DualAxis(dual_axis) => dual_axis.to_display_string(),
            InputKind::PhysicalKey(key_code) => key_code.to_display_string(),
            // InputKind::KeyLocation(key_location) => key_location.to_display_string(),
            InputKind::Modifier(modifier) => modifier.to_display_string(),
            InputKind::Mouse(mouse) => mouse.to_display_string(),
            InputKind::MouseWheel(mouse_wheel_direction) => {
                mouse_wheel_direction.to_display_string()
            }
            InputKind::MouseMotion(mouse_motion) => mouse_motion.to_display_string(),
            _ => "Unknown".to_string(),
        }
    }
}

impl ToDisplayString for MouseMotionDirection {
    fn to_display_string(&self) -> String {
        match self {
            MouseMotionDirection::Up => "Mouse Move Up".to_string(),
            MouseMotionDirection::Down => "Mouse Move Down".to_string(),
            MouseMotionDirection::Left => "Mouse Move Left".to_string(),
            MouseMotionDirection::Right => "Mouse Move Right".to_string(),
        }
    }
}

impl ToDisplayString for MouseWheelDirection {
    fn to_display_string(&self) -> String {
        match self {
            MouseWheelDirection::Up => "Mouse Wheel Up".to_string(),
            MouseWheelDirection::Down => "Mouse Wheel Down".to_string(),
            MouseWheelDirection::Left => "Mouse Wheel Left".to_string(),
            MouseWheelDirection::Right => "Mouse Wheel Right".to_string(),
        }
    }
}

impl ToDisplayString for MouseButton {
    fn to_display_string(&self) -> String {
        match self {
            MouseButton::Left => "Left".to_string(),
            MouseButton::Right => "Right".to_string(),
            MouseButton::Middle => "Middle".to_string(),
            MouseButton::Other(x) => format!("Mouse {}", x).to_string(),
            _ => unreachable!(),
        }
    }
}

impl ToDisplayString for Modifier {
    fn to_display_string(&self) -> String {
        match self {
            Modifier::Alt => "Alt".to_string(),
            Modifier::Control => "Control".to_string(),
            Modifier::Shift => "Shift".to_string(),
            Modifier::Super => "Super".to_string(),
        }
    }
}

// impl ToDisplayString for ScanCode {
//     fn to_display_string(&self) -> String {
//         match self {
//             ScanCode(17) => "W".to_string(),
//             ScanCode(30) => "A".to_string(),
//             ScanCode(31) => "S".to_string(),
//             ScanCode(32) => "D".to_string(),
//             _ => format!("{:?}", self),
//         }
//     }
// }
// pub enum KeyCode {
//     /// This variant is used when the key cannot be translated to any other variant.
//     ///
//     /// The native keycode is provided (if available) so you're able to more reliably match
//     /// key-press and key-release events by hashing the [`KeyCode`]. It is also possible to use
//     /// this for keybinds for non-standard keys, but such keybinds are tied to a given platform.
//     Unidentified(NativeKeyCode),
//     /// <kbd>`</kbd> on a US keyboard. This is also called a backtick or grave.
//     /// This is the <kbd>半角</kbd>/<kbd>全角</kbd>/<kbd>漢字</kbd>
//     /// (hankaku/zenkaku/kanji) key on Japanese keyboards
//     Backquote,
//     /// Used for both the US <kbd>\\</kbd> (on the 101-key layout) and also for the key
//     /// located between the <kbd>"</kbd> and <kbd>Enter</kbd> keys on row C of the 102-,
//     /// 104- and 106-key layouts.
//     /// Labeled <kbd>#</kbd> on a UK (102) keyboard.
//     Backslash,
//     /// <kbd>[</kbd> on a US keyboard.
//     BracketLeft,
//     /// <kbd>]</kbd> on a US keyboard.
//     BracketRight,
//     /// <kbd>,</kbd> on a US keyboard.
//     Comma,
//     /// <kbd>0</kbd> on a US keyboard.
//     Digit0,
//     /// <kbd>1</kbd> on a US keyboard.
//     Digit1,
//     /// <kbd>2</kbd> on a US keyboard.
//     Digit2,
//     /// <kbd>3</kbd> on a US keyboard.
//     Digit3,
//     /// <kbd>4</kbd> on a US keyboard.
//     Digit4,
//     /// <kbd>5</kbd> on a US keyboard.
//     Digit5,
//     /// <kbd>6</kbd> on a US keyboard.
//     Digit6,
//     /// <kbd>7</kbd> on a US keyboard.
//     Digit7,
//     /// <kbd>8</kbd> on a US keyboard.
//     Digit8,
//     /// <kbd>9</kbd> on a US keyboard.
//     Digit9,
//     /// <kbd>=</kbd> on a US keyboard.
//     Equal,
//     /// Located between the left <kbd>Shift</kbd> and <kbd>Z</kbd> keys.
//     /// Labeled <kbd>\\</kbd> on a UK keyboard.
//     IntlBackslash,
//     /// Located between the <kbd>/</kbd> and right <kbd>Shift</kbd> keys.
//     /// Labeled <kbd>\\</kbd> (ro) on a Japanese keyboard.
//     IntlRo,
//     /// Located between the <kbd>=</kbd> and <kbd>Backspace</kbd> keys.
//     /// Labeled <kbd>¥</kbd> (yen) on a Japanese keyboard. <kbd>\\</kbd> on a
//     /// Russian keyboard.
//     IntlYen,
//     /// <kbd>a</kbd> on a US keyboard.
//     /// Labeled <kbd>q</kbd> on an AZERTY (e.g., French) keyboard.
//     KeyA,
//     /// <kbd>b</kbd> on a US keyboard.
//     KeyB,
//     /// <kbd>c</kbd> on a US keyboard.
//     KeyC,
//     /// <kbd>d</kbd> on a US keyboard.
//     KeyD,
//     /// <kbd>e</kbd> on a US keyboard.
//     KeyE,
//     /// <kbd>f</kbd> on a US keyboard.
//     KeyF,
//     /// <kbd>g</kbd> on a US keyboard.
//     KeyG,
//     /// <kbd>h</kbd> on a US keyboard.
//     KeyH,
//     /// <kbd>i</kbd> on a US keyboard.
//     KeyI,
//     /// <kbd>j</kbd> on a US keyboard.
//     KeyJ,
//     /// <kbd>k</kbd> on a US keyboard.
//     KeyK,
//     /// <kbd>l</kbd> on a US keyboard.
//     KeyL,
//     /// <kbd>m</kbd> on a US keyboard.
//     KeyM,
//     /// <kbd>n</kbd> on a US keyboard.
//     KeyN,
//     /// <kbd>o</kbd> on a US keyboard.
//     KeyO,
//     /// <kbd>p</kbd> on a US keyboard.
//     KeyP,
//     /// <kbd>q</kbd> on a US keyboard.
//     /// Labeled <kbd>a</kbd> on an AZERTY (e.g., French) keyboard.
//     KeyQ,
//     /// <kbd>r</kbd> on a US keyboard.
//     KeyR,
//     /// <kbd>s</kbd> on a US keyboard.
//     KeyS,
//     /// <kbd>t</kbd> on a US keyboard.
//     KeyT,
//     /// <kbd>u</kbd> on a US keyboard.
//     KeyU,
//     /// <kbd>v</kbd> on a US keyboard.
//     KeyV,
//     /// <kbd>w</kbd> on a US keyboard.
//     /// Labeled <kbd>z</kbd> on an AZERTY (e.g., French) keyboard.
//     KeyW,
//     /// <kbd>x</kbd> on a US keyboard.
//     KeyX,
//     /// <kbd>y</kbd> on a US keyboard.
//     /// Labeled <kbd>z</kbd> on a QWERTZ (e.g., German) keyboard.
//     KeyY,
//     /// <kbd>z</kbd> on a US keyboard.
//     /// Labeled <kbd>w</kbd> on an AZERTY (e.g., French) keyboard, and <kbd>y</kbd> on a
//     /// QWERTZ (e.g., German) keyboard.
//     KeyZ,
//     /// <kbd>-</kbd> on a US keyboard.
//     Minus,
//     /// <kbd>.</kbd> on a US keyboard.
//     Period,
//     /// <kbd>'</kbd> on a US keyboard.
//     Quote,
//     /// <kbd>;</kbd> on a US keyboard.
//     Semicolon,
//     /// <kbd>/</kbd> on a US keyboard.
//     Slash,
//     /// <kbd>Alt</kbd>, <kbd>Option</kbd>, or <kbd>⌥</kbd>.
//     AltLeft,
//     /// <kbd>Alt</kbd>, <kbd>Option</kbd>, or <kbd>⌥</kbd>.
//     /// This is labeled <kbd>AltGr</kbd> on many keyboard layouts.
//     AltRight,
//     /// <kbd>Backspace</kbd> or <kbd>⌫</kbd>.
//     /// Labeled <kbd>Delete</kbd> on Apple keyboards.
//     Backspace,
//     /// <kbd>CapsLock</kbd> or <kbd>⇪</kbd>
//     CapsLock,
//     /// The application context menu key, which is typically found between the right
//     /// <kbd>Super</kbd> key and the right <kbd>Control</kbd> key.
//     ContextMenu,
//     /// <kbd>Control</kbd> or <kbd>⌃</kbd>
//     ControlLeft,
//     /// <kbd>Control</kbd> or <kbd>⌃</kbd>
//     ControlRight,
//     /// <kbd>Enter</kbd> or <kbd>↵</kbd>. Labeled <kbd>Return</kbd> on Apple keyboards.
//     Enter,
//     /// The Windows, <kbd>⌘</kbd>, <kbd>Command</kbd>, or other OS symbol key.
//     SuperLeft,
//     /// The Windows, <kbd>⌘</kbd>, <kbd>Command</kbd>, or other OS symbol key.
//     SuperRight,
//     /// <kbd>Shift</kbd> or <kbd>⇧</kbd>
//     ShiftLeft,
//     /// <kbd>Shift</kbd> or <kbd>⇧</kbd>
//     ShiftRight,
//     /// <kbd> </kbd> (space)
//     Space,
//     /// <kbd>Tab</kbd> or <kbd>⇥</kbd>
//     Tab,
//     /// Japanese: <kbd>変</kbd> (henkan)
//     Convert,
//     /// Japanese: <kbd>カタカナ</kbd>/<kbd>ひらがな</kbd>/<kbd>ローマ字</kbd> (katakana/hiragana/romaji)
//     KanaMode,
//     /// Korean: HangulMode <kbd>한/영</kbd> (han/yeong)
//     ///
//     /// Japanese (Mac keyboard): <kbd>か</kbd> (kana)
//     Lang1,
//     /// Korean: Hanja <kbd>한</kbd> (hanja)
//     ///
//     /// Japanese (Mac keyboard): <kbd>英</kbd> (eisu)
//     Lang2,
//     /// Japanese (word-processing keyboard): Katakana
//     Lang3,
//     /// Japanese (word-processing keyboard): Hiragana
//     Lang4,
//     /// Japanese (word-processing keyboard): Zenkaku/Hankaku
//     Lang5,
//     /// Japanese: <kbd>無変換</kbd> (muhenkan)
//     NonConvert,
//     /// <kbd>⌦</kbd>. The forward delete key.
//     /// Note that on Apple keyboards, the key labelled <kbd>Delete</kbd> on the main part of
//     /// the keyboard is encoded as [`Backspace`].
//     ///
//     /// [`Backspace`]: Self::Backspace
//     Delete,
//     /// <kbd>Page Down</kbd>, <kbd>End</kbd>, or <kbd>↘</kbd>
//     End,
//     /// <kbd>Help</kbd>. Not present on standard PC keyboards.
//     Help,
//     /// <kbd>Home</kbd> or <kbd>↖</kbd>
//     Home,
//     /// <kbd>Insert</kbd> or <kbd>Ins</kbd>. Not present on Apple keyboards.
//     Insert,
//     /// <kbd>Page Down</kbd>, <kbd>PgDn</kbd>, or <kbd>⇟</kbd>
//     PageDown,
//     /// <kbd>Page Up</kbd>, <kbd>PgUp</kbd>, or <kbd>⇞</kbd>
//     PageUp,
//     /// <kbd>↓</kbd>
//     ArrowDown,
//     /// <kbd>←</kbd>
//     ArrowLeft,
//     /// <kbd>→</kbd>
//     ArrowRight,
//     /// <kbd>↑</kbd>
//     ArrowUp,
//     /// On the Mac, this is used for the numpad <kbd>Clear</kbd> key.
//     NumLock,
//     /// <kbd>0 Ins</kbd> on a keyboard. <kbd>0</kbd> on a phone or remote control
//     Numpad0,
//     /// <kbd>1 End</kbd> on a keyboard. <kbd>1</kbd> or <kbd>1 QZ</kbd> on a phone or remote control
// impl fmt::Display for YourEnum {
//     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//         let display_string = match self {
//             YourEnum::Numpad1 => "Numpad1",
//             YourEnum::Numpad2 => "Numpad2",
//             YourEnum::Numpad3 => "Numpad3",
//             // ... add more cases for other enum variants ...
//             YourEnum::F35 => "F35",
//         };
//         write!(f, "{}", display_string)
//     }
// }

impl ToDisplayString for KeyCode {
    fn to_display_string(&self) -> String {
        format!("{:?}", self)
    }
}

impl ToDisplayString for DualAxis {
    fn to_display_string(&self) -> String {
        match (self.x.axis_type, self.y.axis_type) {
            (
                AxisType::Gamepad(GamepadAxisType::LeftStickX),
                AxisType::Gamepad(GamepadAxisType::LeftStickY),
            ) => "Left Stick".to_string(),
            (
                AxisType::Gamepad(GamepadAxisType::LeftStickY),
                AxisType::Gamepad(GamepadAxisType::LeftStickX),
            ) => "Left Stick".to_string(),
            (
                AxisType::Gamepad(GamepadAxisType::RightStickX),
                AxisType::Gamepad(GamepadAxisType::RightStickY),
            ) => "Right Stick".to_string(),
            (
                AxisType::Gamepad(GamepadAxisType::RightStickY),
                AxisType::Gamepad(GamepadAxisType::RightStickX),
            ) => "Right Stick".to_string(),
            // TODO: add more cases for `MouseWheel` and `MouseMotion`
            _ => "Not yet implemented".to_string(),
        }
    }
}

impl ToDisplayString for GamepadButtonType {
    fn to_display_string(&self) -> String {
        match self {
            GamepadButtonType::South => "󰸴".to_string(), // Cross/A
            GamepadButtonType::East => "󰸷".to_string(),  // Circle/B
            GamepadButtonType::North => "󰸸".to_string(), // Triangle/Y
            GamepadButtonType::West => "󰸵".to_string(),  // Square/X
            GamepadButtonType::C => "C".to_string(),
            GamepadButtonType::Z => "Z".to_string(),
            GamepadButtonType::LeftTrigger => "Left Bumper".to_string(),
            GamepadButtonType::RightTrigger => "Right Bumper".to_string(),
            GamepadButtonType::LeftTrigger2 => "Left Trigger".to_string(),
            GamepadButtonType::RightTrigger2 => "Right Trigger".to_string(),
            GamepadButtonType::Select => "Select".to_string(),
            GamepadButtonType::Start => "Start".to_string(),
            GamepadButtonType::Mode => "Mode".to_string(),
            GamepadButtonType::LeftThumb => "Left Trigger Press".to_string(),
            GamepadButtonType::RightThumb => "Right Trigger Press".to_string(),
            GamepadButtonType::DPadUp => "󰹁".to_string(), // DPad Up
            GamepadButtonType::DPadDown => "󰸽".to_string(), // DPad Down
            GamepadButtonType::DPadLeft => "󰸾".to_string(), // DPad Left
            GamepadButtonType::DPadRight => "󰹀".to_string(), // DPad Right
            GamepadButtonType::Other(x) => format!("Gamepad {}", x).to_string(),
            // _ => "Unknown".to_string(),
        }
    }
}

impl ToDisplayString for SingleAxis {
    fn to_display_string(&self) -> String {
        match self.axis_type {
            AxisType::Gamepad(gamepad_axis) => gamepad_axis.to_display_string(),
            AxisType::MouseWheel(mouse_wheel_direction) => {
                mouse_wheel_direction.to_display_string()
            }
            AxisType::MouseMotion(mouse_motion) => mouse_motion.to_display_string(),
        }
    }
}

impl ToDisplayString for GamepadAxisType {
    fn to_display_string(&self) -> String {
        match self {
            GamepadAxisType::LeftStickX => "Left Stick X".to_string(),
            GamepadAxisType::LeftStickY => "Left Stick Y".to_string(),
            GamepadAxisType::LeftZ => "Left Stick Down".to_string(),
            GamepadAxisType::RightStickX => "Right Stick X".to_string(),
            GamepadAxisType::RightStickY => "Right Stick Y".to_string(),
            GamepadAxisType::RightZ => "Right Stick Down".to_string(),
            GamepadAxisType::Other(x) => format!("Gamepad {}", x).to_string(),
            // _ => "Unknown".to_string(),
        }
    }
}

impl ToDisplayString for MouseWheelAxisType {
    fn to_display_string(&self) -> String {
        match self {
            MouseWheelAxisType::X => "Horizontal".to_string(),
            MouseWheelAxisType::Y => "Vertical".to_string(),
        }
    }
}

impl ToDisplayString for MouseMotionAxisType {
    fn to_display_string(&self) -> String {
        match self {
            MouseMotionAxisType::X => "Horizontal".to_string(),
            MouseMotionAxisType::Y => "Vertical".to_string(),
        }
    }
}

/// Resource to store the occupied screen space by each `egui` panel
#[derive(Default, Resource)]
struct OccupiedScreenSpace {
    left: f32,
}

/// UI state to represent which `equi` panels are open
#[derive(Default, Resource)]
pub struct UiState {
    pub left_panel: bool,
}

/// `Setup` **Bevy** sytem to initialise the `egui` visuals
/// This is where the **default** for `egui` is set
fn configure_visuals_system(
    mut contexts: EguiContexts,
    catppuccin: Res<CatppuccinTheme>,
    windows: Query<&Window>,
) {
    let window = windows.single();
    contexts.ctx_mut().set_visuals(match window.window_theme {
        Some(WindowTheme::Dark) => Visuals::catppuccin_dark(),
        _ => Visuals::catppuccin_light(),
    });

    let mut fonts = egui::FontDefinitions::default();

    // TODO: somehow use the **Bevy** asset loader through `scene_assets` to load the font
    // instead of a relative path
    fonts.font_data.insert(
        "JetBrainsMonoNerdFont-Regular".to_owned(),
        egui::FontData::from_static(include_bytes!(
            "../../assets/fonts/JetBrainsMonoNerdFont-Regular.ttf"
        )),
    );

    // Put JetBrainsMono first (highest priority) for proportional text:
    fonts
        .families
        .entry(egui::FontFamily::Proportional)
        .or_default()
        .insert(0, "JetBrainsMonoNerdFont-Regular".to_owned());

    // Put JetBrainsMono first (highest priority) for proportional text:
    fonts
        .families
        .entry(egui::FontFamily::Monospace)
        .or_default()
        .insert(0, "JetBrainsMonoNerdFont-Regular".to_owned());

    contexts.ctx_mut().set_fonts(fonts);
}

// #[derive(Default, Debug)]
// pub struct Binding(usize);

#[derive(Debug, Default)]
pub struct ChangingBinding {
    pub action: InputAction,
    pub binding: usize,
}

/// `Update` **Bevy** system to render the `egui` UI
/// Uses the `UiState` to understand which panels are open and should be rendered
fn ui_binding_panel(
    mut contexts: EguiContexts,
    mut occupied_screen_space: ResMut<OccupiedScreenSpace>,
    ui_state: ResMut<UiState>,
    mut query_camera_action: Query<&mut InputMap<CameraAction>>,
    mut query_general_action: Query<&mut InputMap<GeneralAction>>,
    mut query_moveable_object_action: Query<&mut InputMap<MoveableObjectAction>>,
    mut query_ui_action: Query<&mut InputMap<UiAction>>,
    catppuccin: Res<CatppuccinTheme>,
    mut currently_changing: Local<ChangingBinding>,
    mut keyboard_events: EventReader<KeyboardInput>,
) {
    let ctx = contexts.ctx_mut();

    // info!("Currently changing: {:?}", currently_changing);

    let left_panel = egui::SidePanel::left("left_panel")
        .default_width(300.0)
        .resizable(true)
        .show_animated(ctx, ui_state.left_panel, |ui| {
            egui::Grid::new("cool_grid")
                .num_columns(3)
                .striped(true)
                .spacing((10.0, 10.0))
                .show(ui, |ui| {
                    let size = 15.0; // pt
                    ui.label(
                        RichText::new("Binding")
                            .size(size)
                            .color(Color32::from_catppuccin_colour(catppuccin.flavour.green())),
                    );
                    ui.label(
                        RichText::new("1")
                            .size(size)
                            .color(Color32::from_catppuccin_colour(catppuccin.flavour.green())),
                    );
                    ui.label(
                        RichText::new("2")
                            .size(size)
                            .color(Color32::from_catppuccin_colour(catppuccin.flavour.green())),
                    );
                    ui.end_row();

                    // go through all InputAction variants, and make a title for each
                    // then nested go through each inner variant and make a button for each
                    for action in InputAction::iter() {
                        if matches!(action, InputAction::Undefined) {
                            continue;
                        }
                        ui.label(
                            RichText::new(action.to_string())
                                .italics()
                                .color(Color32::from_catppuccin_colour(
                                    catppuccin.flavour.lavender(),
                                ))
                                .size(size),
                        );

                        ui.end_row();
                        match action {
                            InputAction::MoveableObject(_) => {
                                let map = query_moveable_object_action.single();
                                for inner_action in map.iter() {
                                    ui.label(inner_action.0.to_string());

                                    inner_action.1.iter().enumerate().for_each(|(i, x)| {
                                        let button_response =
                                            ui.button(RichText::new(x.to_display_string()));
                                        if button_response.clicked() {
                                            // button_response.highlight();
                                            *currently_changing = ChangingBinding {
                                                action: InputAction::MoveableObject(
                                                    *inner_action.0,
                                                ),
                                                binding: i,
                                            };
                                        }
                                    });

                                    ui.end_row();
                                }
                            }
                            InputAction::General(_) => {
                                let map = query_general_action.single();
                                for inner_action in map.iter() {
                                    ui.label(inner_action.0.to_string());

                                    inner_action.1.iter().enumerate().for_each(|(i, x)| {
                                        let button_response =
                                            ui.button(RichText::new(x.to_display_string()));
                                        if button_response.clicked() {
                                            // button_response.highlight();
                                            *currently_changing = ChangingBinding {
                                                action: InputAction::General(*inner_action.0),
                                                binding: i,
                                            };
                                        }
                                    });

                                    ui.end_row();
                                }
                            }
                            InputAction::Camera(_) => {
                                let map = query_camera_action.iter().next().unwrap();
                                for inner_action in map.iter() {
                                    ui.label(inner_action.0.to_string());

                                    inner_action.1.iter().enumerate().for_each(|(i, x)| {
                                        let button_response =
                                            ui.button(RichText::new(x.to_display_string()));
                                        if button_response.clicked() {
                                            // button_response.highlight();
                                            *currently_changing = ChangingBinding {
                                                action: InputAction::Camera(*inner_action.0),
                                                binding: i,
                                            };
                                        }
                                    });

                                    ui.end_row();
                                }
                            }
                            InputAction::Ui(_) => {
                                let map = query_ui_action.single();
                                for inner_action in map.iter() {
                                    ui.label(inner_action.0.to_string());

                                    inner_action.1.iter().enumerate().for_each(|(i, x)| {
                                        let button_response =
                                            ui.button(RichText::new(x.to_display_string()));
                                        if button_response.clicked() {
                                            // button_response.highlight();
                                            // remove the button's text
                                            // button_response.text = "".to_string();
                                            *currently_changing = ChangingBinding {
                                                action: InputAction::Ui(*inner_action.0),
                                                binding: i,
                                            };
                                        }
                                    });

                                    ui.end_row();
                                }
                            }
                            _ => { /* do nothing */ }
                        }
                    }
                });

            ui.allocate_rect(ui.available_rect_before_wrap(), egui::Sense::hover());
        });

    // check for any input at all (keyboard, mouse, gamepad, etc.)
    // if there is, then rebind the map
    for event in keyboard_events.read() {
        let key_code = event.key_code;
        match currently_changing.action {
            InputAction::Camera(action) => {
                let mut map = query_camera_action.single_mut();
                map.remove_at(&action, currently_changing.binding);
                map.insert(action, UserInput::Single(InputKind::PhysicalKey(key_code)));
            }
            InputAction::General(action) => {
                let mut map = query_general_action.single_mut();
                map.remove_at(&action, currently_changing.binding);
                map.insert(action, UserInput::Single(InputKind::PhysicalKey(key_code)));
            }
            InputAction::MoveableObject(action) => {
                let mut map = query_moveable_object_action.single_mut();
                map.remove_at(&action, currently_changing.binding);
                map.insert(action, UserInput::Single(InputKind::PhysicalKey(key_code)));
            }
            InputAction::Ui(action) => {
                let mut map = query_ui_action.single_mut();
                map.remove_at(&action, currently_changing.binding);
                map.insert(action, UserInput::Single(InputKind::PhysicalKey(key_code)));
            }
            _ => { /* do nothing */ }
        }
        *currently_changing = ChangingBinding::default();
    }

    occupied_screen_space.left = left_panel
        .map(|ref inner| inner.response.rect.width())
        .unwrap_or(0.0);

    // occupied_screen_space.left = if left_panel.is_some() {
    //     left_panel.unwrap().response.rect.width()
    // } else {
    //     0.0
    // };
    // occupied_screen_space.left = if left_panel.is_some() {
    //     left_panel.unwrap().response.rect.width()
    // } else {
    //     0.0
    // };
}
