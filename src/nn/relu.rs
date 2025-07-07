use crate::central::*;
use crate::nn::*;

pub struct ReLU {

}

impl ReLU {
    pub fn new() -> ReLU {
        ReLU {  
            
        }
    }
}

impl Layer for ReLU {
    fn forward(&mut self, mut inputs: Tensor) -> Tensor {
        inputs.relu()
    }

    fn get_parameters(&self) -> Vec<TensorID> {
        vec![]
    }
}