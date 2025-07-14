use crate::central::*;
use crate::nn::*;
use std::f32::consts::PI;
pub struct GELU {

}

impl GELU {
    pub fn new() -> GELU {
        GELU {  }
    }
}

impl Layer for GELU {
    //GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))s
    fn forward(&mut self, inputs: Tensor) -> Tensor {
        let x_pow = inputs.pow(3.0);
        let why = 1.0 + (((2.0 / PI).sqrt() * (inputs + 0.044715 * x_pow)).tanh());
        return why;
    }

    fn get_parameters(&self) -> Vec<TensorID> {
        vec![]
    }
}