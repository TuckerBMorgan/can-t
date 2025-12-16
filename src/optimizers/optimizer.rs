use crate::nn::Model;

pub trait Optimizer {
    fn get_parameters(&mut self, mode: &mut dyn Model);
    fn update(&mut self);
    fn zero_grads(&mut self);
}
