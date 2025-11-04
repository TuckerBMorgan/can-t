use crate::central::*;
use crate::nn::*;
use std::f32::consts::PI;

/*

def swiglu(x, alpha: float = 1.702, limit: float = 7.0):
    x_glu, x_linear = x[..., ::2], x[..., 1::2]
    # Clamp the input values
    x_glu = x_glu.clamp(min=None, max=limit)
    x_linear = x_linear.clamp(min=-limit, max=limit)
    out_glu = x_glu * torch.sigmoid(alpha * x_glu)
    # Note we add an extra bias of 1 to the linear layer
    return out_glu * (x_linear + 1)
*/

pub struct Swiglu {
    alpha: f32, 
    limit: f32
}

impl Swiglu {
    pub fn new(alpha: f32, limit: f32) -> Self {
        Swiglu { alpha, limit }
    }
}

impl Layer for Swiglu {
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
