use std::default;

use crate::central::{TensorID, get_equation, zero_all_grads};
use crate::nn::Model;
use crate::optimizers::*;

#[derive(Default)]
pub struct SGD {
    parameters: Vec<TensorID>,
    learning_rate: f32,
    maximize: bool,
    weight_decay: f32,
    momentum: f32,
    dampening: f32,
}

impl SGD {
    pub fn new(learning_rate: f32) -> SGD {
        SGD {
            parameters: vec![],
            learning_rate,
            maximize: false,
            weight_decay: 0.0,
            momentum: 0.0,
            dampening: 0.0,
        }
    }
}

impl Optimizer for SGD {
    fn get_parameters(&mut self, model: &mut dyn Model) {
        self.parameters.extend(model.get_parameters());
    }

    fn update(&mut self) {
        if self.dampening.abs() > 0.0 || self.momentum.abs() > 0.0 {
            panic!("Damening and Momentum are not yet implemented for SDG");
        }

        for p in &self.parameters {
            let mut use_learning_rate = self.learning_rate;
            // This will have SDG attempt to maximize the obejctive instate of mizimize it
            // Gradient Ascent vs Gradient Descent
            if self.maximize == false {
                use_learning_rate *= -1.0;
            }
            get_equation().update_single_parameter(*p, use_learning_rate, self.weight_decay);
        }
    }

    fn zero_grads(&mut self) {
        zero_all_grads();
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        central::{Tensor, get_equation},
        nn::Model,
        optimizers::{Optimizer, SGD},
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
        let mut sgd = SGD::new(0.01);

        sgd.get_parameters(&mut basic_model);

        for _ in 0..10 {
            let input = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![4]);
            let output = basic_model.forward(input);
            let expected_output = Tensor::from_vec(vec![3.0, 3.0, 3.0, 3.0], vec![4]);
            let loss = (expected_output - output).pow(2.0).mean(vec![0]);
            loss.backward();
            sgd.update();
            get_equation().compact_tensor_store();
        }
    }
}
