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
    use crate::utils::GGUFFile;

    fn approx_equal(a: f32, b: f32, epsilon: f32) -> bool {
        (a - b).abs() <= epsilon
    }

    fn compare_tensors(a: Tensor, b: Tensor) {
        let epsilon = 1e-5;
        let a_item = a.item();
        let together = a_item.iter().zip(b.item());
        for (a, b) in together {
            assert!(approx_equal(*a, b, epsilon), "a {} b {}", *a, b);
        }
    }


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

    #[test]
    pub fn basic_reshape_backward_test() {
        let epsilon = 1e-5;
        let mut gguf_file = GGUFFile::new(String::from("./models/tests/reshape/reshape_backward.gguf"));

        let tensor_a = Tensor::from_gguf_file(String::from("reshape_backward_tensor_a"), &mut gguf_file);
        let tensor_a_real_grad = Tensor::from_gguf_file(String::from("reshape_backward_tensor_a_grad"), &mut gguf_file);
        let tensor_b_real = Tensor::from_gguf_file(String::from("reshape_backward_tensor_b"), &mut gguf_file);
        let tensor_b_real_grad = Tensor::from_gguf_file(String::from("reshape_backward_tensor_b_grad"), &mut gguf_file);
        let tensor_c_real =
            Tensor::from_gguf_file(String::from("reshape_backward_tensor_c"), &mut gguf_file);
        let tensor_b = tensor_a.reshape(Shape::new(vec![4]));
        let tensor_c = tensor_b.sum(vec![0], true);        
        tensor_c.backward();
        compare_tensors(tensor_b, tensor_b_real);
        compare_tensors(tensor_c, tensor_c_real);

        let tensor_b_real_grad_item = tensor_b_real_grad.item();
        let together = tensor_b_real_grad_item.iter().zip(tensor_b.grad());
        for (a, b) in together {
            assert!(approx_equal(*a, b, epsilon));
        }

        let tensor_a_real_grad_item = tensor_a_real_grad.item();
        let together = tensor_a_real_grad_item.iter().zip(tensor_a.grad());
        for (a, b) in together {
            assert!(approx_equal(*a, b, epsilon));
        }

    }
}
