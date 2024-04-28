use crate::network::{ state::{Vec2D, Network}, utils};

pub struct AdamHyperParams {
    pub beta_1: f64,
    pub beta_2: f64,
    pub epsilon: f64
}

#[derive(Clone)]
pub struct Moment {
    pub weights: Vec<Vec2D>,
    pub biases: Vec2D
}

impl Moment {
    pub fn new(composition: &[usize]) -> Self {
        Self {
            weights: utils::zeros_3d_vec(composition),
            biases: utils::zeros_2d_vec(composition, 1)
        }
    }
}

impl Network {   
    pub fn compute_moments(
        moment_1: &mut f64, moment_2: &mut f64, optimizer: &AdamHyperParams, gradient: &f64, iteration: &i32
    ) -> (f64, f64) {
        *moment_1 = optimizer.beta_1 * *moment_1 + (1.0 - optimizer.beta_1) * gradient;
        *moment_2 = optimizer.beta_2 * *moment_2 + (1.0 - optimizer.beta_2) * gradient.powi(2);

        (
            *moment_1 / (1.0 - optimizer.beta_1.powi(*iteration)),
            *moment_2 / (1.0 - optimizer.beta_2.powi(*iteration))
        )
    }
}

pub struct Adam {
    pub iteration: i32,
    pub moment_1: Moment,
    pub moment_2: Moment
}

impl Adam {
    pub fn new(composition: &[usize]) -> Self {
        Self {
            iteration: 0,
            moment_1: Moment::new(composition),
            moment_2: Moment::new(composition)
        }
    }
}
