
use crate::central::*;
use std::ops::Shl;
use super::get_equation;
use crate::utils::handle_broadcasting;

 
impl Shl for Tensor {
    type Output = Tensor;
    fn shl(self, rhs: Self) -> Self::Output {

        // We want to make sure that the two operands can be matmuled together
        // so we try to broadcast them together if they do not equal each other
        let (working_lfs, working_rhs) = handle_broadcasting(self, rhs);

        // we vend the actual work of doing the matmul to the equation, which can take advantage of platform libs
        let data = get_equation().matmul_tensor(working_lfs.id,working_rhs.id);

        // Shape has helper funciton to figure out the shape for us
        let matmul_shape = working_lfs.shape.matmul_shape(working_rhs.shape);

        let return_tensor = Tensor::create_tensor_data_and_shape_and_operation(matmul_shape, data, Operation::Matmul(working_lfs.id, working_rhs.id ));
        return return_tensor;
    }
}
#[cfg(test)]
mod tests {

    #[test]
    pub fn basic_matmul_test() {
        
    }
}