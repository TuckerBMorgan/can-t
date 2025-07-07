use crate::central::*;

pub trait Model {
    fn forward(&mut self, input: Tensor) -> Tensor;
    fn get_parameters(&self) -> Vec<TensorID>;
}
