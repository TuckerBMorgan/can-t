use std::default;
use std::hash::Hash;

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
            t: 0,
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
    }

    fn update(&mut self) {
        
    }
}