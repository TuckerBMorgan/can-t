use super::get_equation;
use crate::central::*;

impl Tensor {
    /// Reshapes the underlaying tensor into a new shape
    /// it preforms a full copy under the hood
    pub fn reshape(&self, shape: Shape) -> Tensor {
        assert!(self.shape.can_reshape_to(shape));
        // Get the original data source
        // this is in effect a copy for large tensors, so while I am not happy with it, I am ok with for the moment
        let data = self.item().into_raw_vec();
        let new_tensor = Tensor::create_tensor_data_and_shape_and_operation(
            shape,
            data,
            Operation::Reshape(self.id, shape),
        );
        return new_tensor;
    }
}

/// Handles calculating and passing back the gardient of a reshape operation
pub fn backward_for_reshape(packet: BackproagationPacket) {
    if let Operation::Reshape(from, _shape) = packet.operation {
        // The reshape operation is simple, since at the end of the day the number and order of
        // the elements does not change
        let grad = packet
            .equation
            .get_grad_flat_buffer(packet.incoming_grad)
            .to_owned();

        packet.equation.add_tensor_grad(from, grad.to_vec());
    }
}

#[cfg(test)]
mod tests {
    use crate::central::{Shape, Tensor};

    #[test]
    pub fn basic_reshape_test() {
        let test = Tensor::element(Shape::new(vec![1, 2, 3, 4]), 4.0);
        let reshaped = test.reshape(Shape::new(vec![1, 3, 2, 4]));
        let shape = reshaped.shape;
        assert!(shape.dimensions()[0] == 1);
        assert!(shape.dimensions()[1] == 3);
        assert!(shape.dimensions()[2] == 2);
        assert!(shape.dimensions()[3] == 4);
    }

    #[test]
    pub fn basic_reshape_test_2() {
        let test = Tensor::element(Shape::new(vec![4]), 4.0);
        let reshaped = test.reshape(Shape::new(vec![2, 2]));
        let shape = reshaped.shape;
        assert!(shape.dimensions()[0] == 2);
        assert!(shape.dimensions()[1] == 2);
    }
}
