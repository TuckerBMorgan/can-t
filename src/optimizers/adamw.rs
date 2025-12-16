use crate::central::{TensorID, get_equation, zero_all_grads};
use crate::nn::Model;
use crate::optimizers::*;
use std::collections::HashMap;

pub struct AdamW {
    beta_1: f32,
    beta_2: f32,
    t: usize,
    momentum: HashMap<TensorID, Vec<f32>>,
    velocity: HashMap<TensorID, Vec<f32>>,
    params: Vec<TensorID>,
    amsgrad: bool,
    learning_rate: f32,
    weight_decay: f32,
}

impl Default for AdamW {
    fn default() -> Self {
        AdamW {
            beta_1: 0.9,
            beta_2: 0.999,
            t: 1, // We want to start at t = 1, others wise we will have n ^ 0
            momentum: HashMap::new(),
            velocity: HashMap::new(),
            params: vec![],
            amsgrad: true,
            learning_rate: 0.001,
            weight_decay: 0.01,
        }
    }
}

impl AdamW {
    pub fn new(learning_rate: f32) -> AdamW {
        let mut AdamW_default = AdamW::default();
        AdamW_default.learning_rate = learning_rate;
        return AdamW_default;
    }
}

impl Optimizer for AdamW {
    fn get_parameters(&mut self, model: &mut dyn Model) {
        self.params.extend(model.get_parameters());

        for param in &self.params {
            let shape = get_equation().get_tensor_shape(*param);

            let total_elements = shape.total_size();

            let velocity = vec![0.0; total_elements];
            let momentum = vec![0.0; total_elements];
            self.momentum.insert(*param, momentum);
            self.velocity.insert(*param, velocity);
        }
    }

    fn update(&mut self) {
        let eps = 1e-08;

        for param in &self.params {
            let current_weight = get_equation().get_data_flat_buffer(*param).to_vec();
            let current_grad = get_equation().get_grad_flat_buffer(*param).to_vec();

            // 1. Decoupled weight decay
            let decayed_weights: Vec<f32> = current_weight
                .iter()
                .map(|theta| theta - self.learning_rate * self.weight_decay * theta)
                .collect();

            // 2. Update biased first moment estimate
            let new_momentum: Vec<f32> = current_grad
                .iter()
                .zip(&self.momentum[param])
                .map(|(grad, old_m)| self.beta_1 * old_m + (1.0 - self.beta_1) * grad)
                .collect();

            // 3. Update biased second moment estimate
            let mut new_velocity: Vec<f32> = current_grad
                .iter()
                .zip(&self.velocity[param])
                .map(|(grad, old_v)| self.beta_2 * old_v + (1.0 - self.beta_2) * grad.powi(2))
                .collect();

            // 4. AMSGrad: track max of uncorrected velocity
            if self.amsgrad {
                new_velocity = new_velocity
                    .iter()
                    .zip(&self.velocity[param])
                    .map(|(new, old)| new.max(*old))
                    .collect();
            }

            // 5. Bias correction
            let bias_correction_1 = 1.0 - self.beta_1.powi(self.t as i32);
            let bias_correction_2 = 1.0 - self.beta_2.powi(self.t as i32);

            // 6. Compute update
            let updated_weights: Vec<f32> = decayed_weights
                .iter()
                .zip(&new_momentum)
                .zip(&new_velocity)
                .map(|((weight, m), v)| {
                    let m_hat = m / bias_correction_1;
                    let v_hat = v / bias_correction_2;
                    weight - self.learning_rate * m_hat / (v_hat.sqrt() + eps)
                })
                .collect();

            // Store RAW moments (not bias-corrected)
            self.momentum.insert(*param, new_momentum);
            self.velocity.insert(*param, new_velocity);

            get_equation().set_parameter(*param, updated_weights);
        }

        self.t += 1;
    }

    fn zero_grads(&mut self) {
        zero_all_grads();
    }
}

#[cfg(test)]
mod tests {
    use crate::optimizers::AdamW::AdamW;
    use crate::{
        central::{Tensor, get_equation},
        nn::Model,
        optimizers::Optimizer,
    };
    #[test]
    pub fn basic_test() {
        // Optimizers work off of Models so we need to write a little wrapper model
        struct BasicModel {
            weight: Tensor,
        }

        impl BasicModel {
            pub fn new(weight: Tensor) -> BasicModel {
                BasicModel { weight }
            }
        }

        impl Model for BasicModel {
            fn forward(&mut self, input: Tensor) -> Tensor {
                return self.weight + input;
            }

            fn get_parameters(&self) -> Vec<crate::central::TensorID> {
                vec![self.weight.id]
            }
        }

        let mut weight = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![4]);
        weight.set_requires_grad(true);
        weight.set_keep_alive(true);

        let mut basic_model = BasicModel::new(weight);
        let mut sgd = AdamW::new(0.1);

        sgd.get_parameters(&mut basic_model);

        for _ in 0..1 {
            let input = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![4]);
            let output = basic_model.forward(input);
            let expected_output = Tensor::from_vec(vec![3.0, 3.0, 3.0, 3.0], vec![4]);
            let loss = (expected_output - output).pow(2.0).mean(vec![0]);
            loss.backward();
            sgd.update();
            get_equation().compact_tensor_store();
        }

        let updates = basic_model.weight.item().into_raw_vec();

        for element in updates {
            assert!(element - 1.1 < f32::EPSILON, "updates {:?}", element);
        }
    }
}
