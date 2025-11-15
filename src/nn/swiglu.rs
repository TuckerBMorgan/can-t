use crate::central::*;
use crate::nn::*;
use std::f32;

pub struct Swiglu {
    alpha: f32,
    limit: f32,
}

impl Swiglu {
    pub fn new(alpha: f32, limit: f32) -> Self {
        Swiglu { alpha, limit }
    }
}

impl Layer for Swiglu {
    fn forward(&mut self, x: Tensor) -> Tensor {
        let final_dimension = x.shape.dimensions().last().unwrap().clone();
        // We need to have an even set of dimensions here
        assert!(final_dimension % 2 == 0);
        let mut evens = vec![];
        let mut odds = vec![];

        for i in 0..final_dimension {
            if i % 2 == 0 {
                evens.push(i as f32);
            } else {
                odds.push(i as f32);
            }
        }
        let evens_len = evens.len();
        let even_indices = Tensor::from_vec(evens, vec![evens_len]);
        let odds_len = odds.len();
        let odds_indices = Tensor::from_vec(odds, vec![odds_len]);
        let x_glu = x.gather(x.shape.dimensions().len() - 1, even_indices);
        let x_linear = x.gather(x.shape.dimensions().len() - 1, odds_indices);
        let x_glu_clamp = x_glu.clamp(f32::MIN, self.limit);
        let x_linear_clamp = x_linear.clamp(-self.limit, self.limit);
        let out_glu = x_glu_clamp * (self.alpha * x_glu_clamp).sigmoid();
        return out_glu + (x_linear_clamp + 1.0);
    }

    fn get_parameters(&self) -> Vec<TensorID> {
        vec![]
    }
}
