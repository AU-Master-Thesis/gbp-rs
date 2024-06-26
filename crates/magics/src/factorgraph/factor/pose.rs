//! Pose factor

use std::borrow::Cow;

use gbp_linalg::prelude::*;

use super::{Factor, FactorState, Measurement};

#[derive(Debug)]
pub struct PoseFactor;

impl PoseFactor {
    pub const NEIGHBORS: usize = 1;
}

impl std::fmt::Display for PoseFactor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "")
    }
}

impl Factor for PoseFactor {
    #[inline(always)]
    fn name(&self) -> &'static str {
        "PoseFactor"
    }

    #[inline]
    fn color(&self) -> [u8; 3] {
        //#c6a0f6
        [198, 160, 246]
    }

    #[inline(always)]
    fn jacobian_delta(&self) -> Float {
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

    /// Default jacobian is the first order taylor series jacobian
    #[inline]
    fn jacobian(&self, state: &FactorState, x: &Vector<Float>) -> Cow<'_, Matrix<Float>> {
        Cow::Owned(self.first_order_jacobian(state, x.clone()))
    }

    /// Default measurement function is the identity function
    #[inline(always)]
    // fn measure(&self, _state: &FactorState, x: &Vector<Float>) -> Vector<Float> {
    fn measure(&self, _state: &FactorState, x: &Vector<Float>) -> Measurement {
        Measurement::new(x.clone())
    }

    #[inline(always)]
    fn neighbours(&self) -> usize {
        Self::NEIGHBORS
    }
}
