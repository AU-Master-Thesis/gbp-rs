//! A small collection of extension traits and types for ndarray.

pub mod pretty_print;

/// `use gbp_linalg::prelude::*` to import all the common symbols from this
/// crate
pub mod prelude {
    // pub use ndarray::{array, concatenate, s, Axis};

    pub use super::{
        pretty_print::*, Float, GbpFloat, Matrix, MatrixView, NdarrayVectorExt, Vector, VectorNorm,
        VectorView,
    };
}

/// Marker trait for floating point types used in GBP.
/// - `ndarray::NdFloat` is a trait for floating point types that can be used
///   with ndarray.
/// It is implemented for f32 and f64.
/// - Copy, is to make some of the methods more ergonomic to use.
/// - `std::iter::Sum` is required by the `det()` method by
///   `ndarray_inverse::Inverse::det()`
/// which we use in `gbp_multivariate_normal::MultivariateNormal` to calculate
/// the determinant of the precision matrix.
pub trait GbpFloat: ndarray::NdFloat + Copy + std::iter::Sum {}

impl GbpFloat for f32 {}
impl GbpFloat for f64 {}

/// The precision of the floating point type used in GBP.
pub type Float = f64;

// only available on nightly :(
// pub type Vector<T> = ndarray::Array1<T: Scalar>;
// pub type Matrix<T> = ndarray::Array2<T: Scalar>;

/// Type alias for `ndarray::Array1<T>`
pub type Vector<T> = ndarray::Array1<T>;
/// Type alias for `ndarray::Array2<T>`
pub type Matrix<T> = ndarray::Array2<T>;
/// Type alias for `ndarray::ArrayView1<T>`
pub type VectorView<'a, T> = ndarray::ArrayView1<'a, T>;
/// Type alias for `ndarray::ArrayView2<T>`
pub type MatrixView<'a, T> = ndarray::ArrayView2<'a, T>;

/// Trait for vector norms
pub trait VectorNorm {
    /// The scalar type of the vector
    type Scalar: GbpFloat;
    /// Calculate the Euclidean norm of the vector
    fn euclidean_norm(&self) -> Self::Scalar;
    /// Calculate the L1 norm of the vector
    fn l1_norm(&self) -> Self::Scalar;

    /// Calculate the L2 norm of the vector
    #[inline(always)]
    fn l2_norm(&self) -> Self::Scalar {
        self.euclidean_norm()
    }
}

macro_rules! vector_norm_trait_impl {
    ($float:ty) => {
        impl VectorNorm for Vector<$float> {
            type Scalar = $float;

            fn euclidean_norm(&self) -> Self::Scalar {
                <$float>::sqrt(self.fold(0.0, |acc, x| acc + x * x))
            }

            #[inline(always)]
            fn l1_norm(&self) -> Self::Scalar {
                self.fold(0.0, |acc, x| acc + x.abs())
            }
        }

        impl VectorNorm for VectorView<'_, $float> {
            type Scalar = $float;

            fn euclidean_norm(&self) -> Self::Scalar {
                <$float>::sqrt(self.fold(0.0, |acc, x| acc + x * x))
            }

            #[inline(always)]
            fn l1_norm(&self) -> Self::Scalar {
                self.fold(0.0, |acc, x| acc + x.abs())
            }
        }
    };
}

vector_norm_trait_impl!(f32);
vector_norm_trait_impl!(f64);

/// Extension trait for `ndarray::Array1<T>`
pub trait NdarrayVectorExt: Clone + VectorNorm {
    /// The scalar type of the vector
    type Scalar: GbpFloat;
    /// Normalize the vector in place.
    fn normalize(&mut self);
    /// Return a normalized copy of the vector.
    #[must_use]
    fn normalized(&self) -> Self {
        let mut copy = self.clone();
        copy.normalize();
        copy
    }
}

macro_rules! ndarray_vector_ext_trait_impl {
    ($float:ty) => {
        impl NdarrayVectorExt for Vector<$float> {
            type Scalar = $float;

            /// Normalize the vector in place.
            fn normalize(&mut self) {
                let mag = self.euclidean_norm();
                if mag == 0.0 || mag.is_infinite() {
                    return;
                    // panic!("Cannot normalize a vector with zero magnitude or infinite
                    // magnitude.");
                }
                for i in 0..self.len() {
                    self[i] /= mag;
                }
                // self.map_mut(|x| *x = *x / mag);
            }
        }

        // impl NdarrayVectorExt for VectorView<'_, $float> {
        //     type Scalar = $float;

        //     /// Normalize the vector in place.
        //     fn normalize(&mut self) {
        //         let mag = self.euclidean_norm();
        //         if mag == 0.0 || mag.is_infinite() {
        //             return;
        //             // panic!("Cannot normalize a vector with zero magnitude or
        // infinite             // magnitude.");
        //         }
        //         for i in 0..self.len() {
        //             self[i] /= mag;
        //         }
        //         // self.map_mut(|x| *x = *x / mag);
        //     }
        // }
    };
}

// TODO: write test cases for impls
ndarray_vector_ext_trait_impl!(f32);
ndarray_vector_ext_trait_impl!(f64);

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use arbtest::arbtest;
    use ndarray::array;
    use paste::paste;
    use pretty_assertions::assert_eq;

    use super::*;

    macro_rules! test_vector_norm {
        ($name:ident: $ty:ty) => {
            paste! {
                #[test]
                fn [<$name _vec2_arbitrary_values>]() {
                    // Test with a 2D vector of arbitrary values
                    arbtest(|u| {
                        let a: $ty = u.arbitrary()?;
                        let b: $ty = u.arbitrary()?;
                        // eprintln!("a: {}, b: {}", a, b);
                        if a.is_nan() || b.is_nan() {
                            // To lazy to handle NaNs ¯\_(ツ)_/¯
                            return Ok(());
                        }

                        let v: Vector<$ty> = array![a, b];
                        assert_relative_eq!(<$ty>::sqrt(a.mul_add(a, b * b)), v.euclidean_norm());
                        assert_relative_eq!(v.l1_norm(), a.abs() + b.abs());
                        assert_relative_eq!(v.l2_norm(), v.euclidean_norm());
                        // assert_eq!(v.l2_norm(), v.euclidean_norm());
                        Ok(())
                    });
                }

                #[test]
                fn [<$name _arbitrary_length_vector>]() {
                    // Test with a vector of arbitrary length
                    arbtest(|u| {
                        let v: Vec<$ty> = u.arbitrary()?;
                        if v.iter().any(|x| x.is_nan()) {
                            // To lazy to handle NaNs ¯\_(ツ)_/¯
                            return Ok(());
                        }
                        let v: Vector<$ty> = Vector::from(v);

                        assert_relative_eq!(v.euclidean_norm(), v.dot(&v).sqrt());
                        assert_relative_eq!(v.l1_norm(), v.iter().map(|x| x.abs()).sum());
                        assert_relative_eq!(v.l2_norm(), v.euclidean_norm());
                        Ok(())
                    });
                }
            }
        };
    }

    test_vector_norm!(vector_norm_f32: f32);
    test_vector_norm!(vector_norm_f64: f64);

    macro_rules! test_vector_normalize {
        ($name:ident: $ty:ty) => {
            paste! {
                #[test]
                fn [<$name _vec3_arbitrary_values>]() {
                    // Test with a 3D vector of arbitrary values
                    arbtest(|u| {
                        let a: $ty = u.arbitrary()?;
                        let b: $ty = u.arbitrary()?;
                        let c: $ty = u.arbitrary()?;

                        let mut v: Vector<$ty> = array![a, b, c];
                        if v.iter().any(|x| x.is_nan()) {
                            // To lazy to handle NaNs ¯\_(ツ)_/¯
                            return Ok(());
                        }
                        // let mag = <$ty>::sqrt(a * a + b * b + c * c);
                        let mag = <$ty>::sqrt(c.mul_add(c, a.mul_add(a, b * b)));
                        if mag == 0.0 || mag.is_infinite() {
                            // To lazy to handle the edge cases ¯\_(ツ)_/¯
                            return Ok(());
                        }

                        // eprintln!("BEFORE: v: {:#?}, .mag = {}", v, mag);

                        let vn = v.normalized();

                        // eprintln!("AFTER: v: {:#?}, .mag = {}", v, v.euclidean_norm());
                        assert_eq!(vn, array![a / mag, b / mag, c / mag]);

                        v.normalize();
                        assert_eq!(v, array![a / mag, b / mag, c / mag]);

                        Ok(())
                    });
                }

                #[test]
                fn [<$name _of_arbitrary_length_vector>]() {

                    let float_eq = |a: $ty, b: $ty| {
                        let epsilon = <$ty>::EPSILON;
                        (a - b).abs() < epsilon
                    };

                    // Test with vector of arbitrary length
                    arbtest(|u| {
                        let v: Vec<$ty> = u.arbitrary()?;

                        if v.iter().any(|x| x.is_nan()) {
                            // To lazy to handle NaNs ¯\_(ツ)_/¯
                            return Ok(());
                        }

                        let mut v: Vector<$ty> = Vector::from(v);
                        let mag = v.euclidean_norm();
                        if mag.is_infinite() || mag == 0.0 {
                            // To lazy to handle NaNs ¯\_(ツ)_/¯
                            return Ok(());
                        }
                        // eprintln!("BEFORE: v: {:#?} .len() = {}, .mag = {}", v, v.len(), mag);

                        let vn = v.normalized();
                        let expected = &v / mag;
                        // compare each element of the normalized vector with the expected values
                        assert!(vn.iter().zip(expected.iter()).all(|(a, b)| float_eq(*a, *b)));

                        v.normalize();

                        // eprintln!("AFTER: v: {:#?}, .mag = {}", v, v.euclidean_norm());

                        // compare each element of the normalized vector with the expected values
                        assert!(v.iter().zip(expected.iter()).all(|(a, b)| float_eq(*a, *b)));
                        // test the magnitude of the normalized vector, should be 1.0 or close to ...
                        assert_relative_eq!(v.euclidean_norm(), 1.0, epsilon = 1e-1, max_relative = 0.15);
                        Ok(())
                    }).budget_ms(100);
                }
            }
        };
    }

    test_vector_normalize!(vector_normalize_f32: f32);
    test_vector_normalize!(vector_normalize_f64: f64);
}
