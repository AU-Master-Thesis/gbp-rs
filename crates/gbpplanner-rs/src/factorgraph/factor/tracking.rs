//! Obstacle factor

use std::{
    borrow::Cow,
    cell::{Cell, RefCell},
    sync::Mutex,
};

use bevy::math::Vec2;
use gbp_linalg::prelude::*;
use ndarray::{array, s};

use super::{Factor, FactorState};

#[derive(Debug)]
pub struct TrackingFactor {
    /// Reference to the tracking path (Likely from RRT)
    tracking_path: Vec<Vec2>,

    /// Most recent measurement
    last_measurement: Mutex<Cell<Vec2>>,
}

impl TrackingFactor {
    /// An obstacle factor has a single edge to another variable
    pub const NEIGHBORS: usize = 1;

    /// Creates a new [`TrackingFactor`].
    pub fn new(tracking_path: Vec<Vec2>) -> Self {
        assert!(
            tracking_path.len() >= 2,
            "Tracking path must have at least 2 points"
        );
        Self {
            tracking_path,
            last_measurement: Mutex::new(Cell::new(Vec2::ZERO)),
        }
    }

    /// Get the last measurement
    pub fn last_measurement(&self) -> Vec2 {
        self.last_measurement.lock().unwrap().get()
    }
}

impl Factor for TrackingFactor {
    #[inline]
    fn name(&self) -> &'static str {
        "TrackingFactor"
    }

    #[inline]
    fn jacobian(&self, state: &FactorState, x: &Vector<Float>) -> Cow<'_, Matrix<Float>> {
        // Same as PoseFactor
        // TODO: change to not clone x
        Cow::Owned(self.first_order_jacobian(state, x.clone()))
    }

    fn measure(&self, _state: &FactorState, x: &Vector<Float>) -> Vector<Float> {
        // 1. Window pairs of rrt path
        // 1.1. Find the line defined by the two points

        let (projected_point, distance, line) = self
            .tracking_path
            .windows(2)
            .map(|window| {
                let p2 = array![window[1].x as Float, window[1].y as Float];
                let p1 = array![window[0].x as Float, window[0].y as Float];

                let line = &p2 - &p1;
                let normal = array![-line[1], line[0]].normalized();

                // (p1, p2, line, normal)

                let x_pos = x.slice(s![0..2]).to_owned();

                // project x_pos onto the line
                let projected = &p1 + (&x_pos - &p1).dot(&line) / &line.dot(&line) * &line;
                let distance = (&x_pos - &projected).euclidean_norm();

                (projected, distance, line)
            })
            .min_by(|(_, a, _), (_, b, _)| a.partial_cmp(b).unwrap())
            .unwrap();

        // current speed is the magnitude of the velocity x[2..4]
        let speed = x.slice(s![2..4]).euclidean_norm();

        // let future_point = projected_point + speed * lines[min_index].2.normalized();
        let future_point = projected_point + 2.0 * line.normalized();
        self.last_measurement
            .lock()
            .unwrap()
            .set(Vec2::new(future_point[0] as f32, future_point[1] as f32));
        future_point
    }

    #[inline(always)]
    fn jacobian_delta(&self) -> Float {
        // Same as DynamicFactor for now
        // TODO: Tune this
        // NOTE: Maybe this should be influenced by the distance from variable to the
        // measurement
        1e-8
    }

    #[inline(always)]
    fn skip(&self, _state: &FactorState) -> bool {
        false
    }

    #[inline(always)]
    fn linear(&self) -> bool {
        false
    }

    #[inline(always)]
    fn neighbours(&self) -> usize {
        Self::NEIGHBORS
    }
}
