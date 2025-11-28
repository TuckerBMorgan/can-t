use crate::nn::Model;

trait Optimizer {
    fn get_parameters(&mut self, mode: Box<dyn Model>);
    fn update();
}
