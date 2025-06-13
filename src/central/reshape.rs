use crate::central::*;
use super::get_equation;

impl Tensor {
    /// Reshapes the underlaying tensor into a new shape
    /// it preforms a full copy under the hood
    pub fn reshape(&self, shape: Shape) -> Tensor {
        assert!(self.shape.can_reshape_to(shape));
        // Get the original data source
        // this is in effect a copy for large tensors, so while I am not happy with it, I am ok with for the moment
        let data = self.item().into_raw_vec();        
        let new_tensor = Tensor::create_tensor_data_and_shape_and_operation(shape, data, Operation::Reshape(self.id, shape));
        return new_tensor;
    }
}