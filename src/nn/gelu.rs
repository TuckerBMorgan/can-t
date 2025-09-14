use crate::central::*;
use crate::nn::*;
use std::f32::consts::PI;

pub struct GELU;

impl GELU {
    pub fn new() -> Self { Self }
}

impl Layer for GELU {
    // GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    fn forward(&mut self, inputs: Tensor) -> Tensor {
        let x3 = inputs.pow(3.0f32);
        let alpha: f32 = (2.0f32 / PI).sqrt();           // √(2/π)
        0.5f32 * inputs * (1.0f32 + (alpha * (inputs + 0.044_715f32 * x3)).tanh())
    }

    fn get_parameters(&self) -> Vec<TensorID> { vec![] }
}
