use crate::central::*;

pub trait Layer {
    fn forward(&mut self, inputs: Tensor) -> Tensor;
    fn get_parameters(&self) -> Vec<TensorID>;
}
