use crate::central::*;
use crate::nn::*;
use std::f32::consts::PI;

pub struct GELU;

impl GELU {
    pub fn new() -> Self {
        Self
    }
}

impl Layer for GELU {
    fn forward(&mut self, x: Tensor) -> Tensor {
        // Constants aligned with PyTorch/Transformers
        // sqrt(2/pi) ~= 0.7978845608028654, but f32 rounded like PyTorch:
        let alpha: f32 = 0.797_884_6;
        let c: f32 = 0.044_715;

        // Compute x + 0.044715*x^3 via multiplies (more stable than powf)
        let x2 = x * x;
        let inner = x * (1.0f32 + c * x2); // x * (1 + 0.044715*x^2)
        let t = (inner * alpha).tanh();
        0.5f32 * x * (1.0f32 + t)
    }

    fn get_parameters(&self) -> Vec<TensorID> {
        vec![]
    }
}
