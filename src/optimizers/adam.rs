use std::default;
use std::hash::Hash;

use rand_distr::num_traits::Pow;

use crate::central::{TensorID, get_equation};
use crate::nn::Model;
use crate::optimizers::*;
use std::collections::HashMap;


struct Adam {
    beta_1: f32,
    beta_2: f32,
    t: usize,
    momentum: HashMap<TensorID, Vec<f32>>,
    velocity: HashMap<TensorID, Vec<f32>>,
    params: Vec<TensorID>,
    amsgrad: bool,
    learning_rate: f32,
    weight_decay: f32
}

impl Default for Adam {
    fn default() -> Self {
        Adam {
            beta_1: 0.9,
            beta_2: 0.999,
            t: 1, // We want to start at t = 1, others wise we will have n ^ 0
            momentum: HashMap::new(),
            velocity: HashMap::new(),
            params: vec![],
            amsgrad: false,
            learning_rate: 0.001,
            weight_decay: 0.0
        }
    }
}

impl Adam {
    pub fn new(learning_rate: f32) -> Adam {
        let mut adam_default = Adam::default();
        adam_default.learning_rate = learning_rate;
        return adam_default;
    }
}

impl Optimizer for Adam {

    fn get_parameters(&mut self, model: &mut dyn Model) {
        self.params.extend(model.get_parameters());
        panic!("Need to implement having the velocity and momentum vecs here");
    }

    fn update(&mut self) {
        for param in &self.params {
            let current_grad = get_equation().get_grad_flat_buffer(*param).to_vec();

            let new_momentum : Vec<f32> = current_grad.iter().zip(&self.momentum[param]).map(|(grad, old_momentum)|{
                self.beta_1 * old_momentum + (1.0 - self.beta_1) * grad
            }).map(|m_t|m_t / (1.0 - self.beta_1.powf(self.t as f32))).collect();

            let mut  new_velocity : Vec<f32> = current_grad.iter().map(|x|return x.powf(2.0)).zip(&self.velocity[param]).map(|(grad_sqr, old_velocity)|{
                self.beta_2 * old_velocity + (1.0 - self.beta_2) * grad_sqr
            }).collect();

            if self.amsgrad {
                new_velocity = new_velocity.iter().zip(&self.velocity[param]).map(|(new, old)|{
                    new.max(*old)
                }).collect();
            }

            let velocity_hat : Vec<f32> = new_velocity.iter().map(|v_t|{
                v_t / (1.0 - self.beta_2.powf(self.t as f32))
            }).collect();

            let eps = 1e-08;

            let right_side_of_update : Vec<f32> = new_momentum.iter().zip(&velocity_hat).map(|(momentum, velocity)|{
                -(self.learning_rate * momentum) / (velocity.sqrt() + eps)
            }).collect();

            self.momentum.insert(*param, new_momentum);
            self.velocity.insert(*param, velocity_hat);

            get_equation().update_single_parameter_with_provided_gradient(*param, right_side_of_update);
        }

        self.t += 1;
    }
}

#[cfg(test)]
mod tests {

    #[test]
    pub fn basic_test() {

    }
}