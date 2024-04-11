use std::{
    num::{NonZeroU64, NonZeroUsize},
    ops::Deref,
    time::Duration,
};

use bevy::{
    diagnostic::{Diagnostic, DiagnosticPath, Diagnostics, RegisterDiagnostic},
    prelude::*,
    time::common_conditions::*,
};
use typed_floats::StrictlyPositiveFinite;
use units::sample_rate::SampleRate;

use crate::planner::{FactorGraph, RobotState};

// /// Newtype representing a sample rate in seconds.
// /// The newtype wraps a `std::time::Duration` to ensure the invariant that
// the Duration is /// never zero time.
// pub struct SampleRate(Duration);

// impl SampleRate {
//     #[inline]
//     pub const fn from_hz(hz: NonZeroUsize) -> SampleRate {
//         Self(Duration::div_f32(``, ), ))
//         Self(Duration::from_secs(1.0 / hz.get() as f64))
//     }

//     // /// delay in seconds
//     // #[inline]
//     // pub fn from_delay(delay: StrictlyPositiveFinite) -> SampleRate {
//     //     Self(Duration::from_secs(delay.into()))
//     // }

//     /// delay in milliseconds
//     #[inline]
//     pub const fn from_millis(delay: NonZeroU64) -> SampleRate {
//         Self(Duration::from_millis(delay.into()))
//     }

//     /// Takes ownership of `Self` and returns the inner `std::time::Duration`
// type     #[inline]
//     pub fn as_duration(self) -> Duration {
//         self.0
//     }
// }

// impl std::ops::Deref for SampleRate {
//     type Target = Duration;

//     fn deref(&self) -> &Self::Target {
//         &self.0
//     }
// }

#[derive(Default)]
pub struct RobotDiagnosticsPlugin {
    pub config: RobotDiagnosticsConfig,
}

pub struct RobotDiagnosticsConfig {
    pub count_robots_sample_rate:    Option<SampleRate>,
    pub count_variables_and_factors: Option<SampleRate>,
    pub messages_sent_sample_rate:   Option<SampleRate>,
}

impl Default for RobotDiagnosticsConfig {
    fn default() -> Self {
        Self {
            count_robots_sample_rate:    None,
            count_variables_and_factors: Some(SampleRate::from_hz(2.try_into().unwrap())),
            messages_sent_sample_rate:   Some(SampleRate::from_hz(2.try_into().unwrap())),
        }
    }
}

impl Plugin for RobotDiagnosticsPlugin {
    fn build(&self, app: &mut App) {
        app.register_diagnostic(Diagnostic::new(Self::ROBOT_COUNT))
            .register_diagnostic(Diagnostic::new(Self::VARIABLE_COUNT))
            .register_diagnostic(Diagnostic::new(Self::FACTOR_COUNT))
            .register_diagnostic(Diagnostic::new(Self::MESSAGES_SENT_COUNT));

        let sample_schedule = PostUpdate;

        if let Some(duration) = self
            .config
            .count_robots_sample_rate
            .map(SampleRate::as_duration)
        {
            info!(
                "creating system to count number of robots every {:?}",
                duration
            );
            app.add_systems(PostUpdate, Self::count_robots.run_if(on_timer(duration)));
        } else {
            info!("creating system to count number of robots every `Update`");
            app.add_systems(PostUpdate, Self::count_robots);
        }

        if let Some(duration) = self
            .config
            .count_variables_and_factors
            .map(SampleRate::as_duration)
        {
            info!(
                "creating system to count number of variables and factors every {:?}",
                duration
            );
            app.add_systems(
                PostUpdate,
                Self::count_variables_and_factors.run_if(on_timer(duration)),
            );
        } else {
            app.add_systems(PostUpdate, Self::count_variables_and_factors);
        }

        if let Some(duration) = self
            .config
            .messages_sent_sample_rate
            .map(SampleRate::as_duration)
        {
            info!(
                "creating system to count number of messages sent every {:?}",
                duration
            );
            app.add_systems(
                PostUpdate,
                Self::count_messages_sent.run_if(on_timer(duration)),
            );
        } else {
            app.add_systems(PostUpdate, Self::count_messages_sent);
        }

        // .add_systems(
        //     Update,
        //     (
        //         Self::count_robots,
        //         Self::count_variables_and_factors
        //
        // .run_if(repeating_after_delay(self.config.
        // count_variables_and_factors.)),
        // .run_if(repeating_after_delay(Duration::from_millis(500))),
        //         Self::count_messages_sent
        //
        // .run_if(repeating_after_delay(Duration::from_millis(500))),
        //     ),
        // );
    }
}

impl RobotDiagnosticsPlugin {
    pub const FACTOR_COUNT: DiagnosticPath = DiagnosticPath::const_new("factor_count");
    pub const MESSAGES_SENT_COUNT: DiagnosticPath =
        DiagnosticPath::const_new("messages_sent_count");
    pub const ROBOT_COUNT: DiagnosticPath = DiagnosticPath::const_new("robot_count");
    pub const VARIABLE_COUNT: DiagnosticPath = DiagnosticPath::const_new("variable_count");

    fn count_robots(mut diagnostics: Diagnostics, query: Query<(), With<RobotState>>) {
        diagnostics.add_measurement(&Self::ROBOT_COUNT, || query.iter().count() as f64);
    }

    fn count_variables_and_factors(
        mut diagnostics: Diagnostics,
        query: Query<&FactorGraph, With<RobotState>>,
    ) {
        diagnostics.add_measurement(&Self::VARIABLE_COUNT, || {
            query
                .iter()
                .map(|factorgraph| factorgraph.node_count().variables)
                .sum::<usize>() as f64
            // query.par_iter().for_each(|factorgraph| )
        });
        diagnostics.add_measurement(&Self::FACTOR_COUNT, || {
            query
                .iter()
                .map(|factorgraph| factorgraph.node_count().factors)
                .sum::<usize>() as f64
        });
    }

    fn count_messages_sent(
        mut diagnostics: Diagnostics,
        mut query: Query<&mut FactorGraph, With<RobotState>>,
        mut messages_sent_in_total: Local<usize>,
    ) {
        diagnostics.add_measurement(&Self::MESSAGES_SENT_COUNT, || {
            let messages_sent = query
                .iter_mut()
                .map(|mut factorgraph| factorgraph.messages_sent())
                .sum::<usize>();

            *messages_sent_in_total += messages_sent;
            *messages_sent_in_total as f64
        })
    }
}