// use crate::{multivariate_normal::MultivariateNormal, robot::RobotId};

// use nalgebra::{DMatrix, DVector};

// #[derive(Debug)]
// pub struct Message {
//     header: MessageHeader,
//     payload: MultivariateNormal,
// }

// #[derive(Debug)]
// pub struct MessageHeader {
//     sender: RobotId,
//     receiver: RobotId,
// }

// Data structure for a message that is passed in GBP
// this consists of an information vector, precision matrix, and a mean vector.
// Traditionally GBP does not require the sending of the last parameter mu (the mean), as it
// can be calculated from the eta and lambda. We include it here for computational efficiency.

// #[derive(Debug)]
// pub struct MessagePayload(MultivariateNormal);

// impl MessagePayload {
//     pub fn zeros(dofs: usize) -> Self {
//         Self(MultivariateNormal::zeros(dofs))
//     }

//     pub fn new(information_vector: DVector<f64>, precision_matrix: DMatrix<f64>) -> Self {
//         Self(MultivariateNormal {
//             information_vector,
//             precision_matrix,
//         })
//     }

//     pub fn zeroize(&mut self) {
//         self.0.information_vector.fill(0.0);
//         self.0.precision_matrix.fill(0.0);
//     }
// }

// impl std::ops::Add for MessagePayload {
//     type Output = Self;

//     fn add(self, other: Self) -> Self {
//         Self(self.0 + other.0)
//     }
// }

// impl std::ops::AddAssign for MessagePayload {
//     fn add_assign(&mut self, other: Self) {
//         self.0 += other.0;
//     }
// }

// impl std::ops::Sub for MessagePayload {
//     type Output = Self;

//     fn sub(self, other: Self) -> Self {
//         Self(self.0 - other.0)
//     }
// }

// impl std::ops::SubAssign for MessagePayload {
//     fn sub_assign(&mut self, other: Self) {
//         self.0 -= other.0;
//     }
// }