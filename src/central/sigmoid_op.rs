use std::f32;

use crate::central::get_equation;

use super::{BackproagationPacket, Operation, Tensor};

impl Tensor {
    /// Applies the sigmoid function element-wise to the tensor
    /// Returns a new tensor with the same shape where each element x is replaced with sigmoid(x)
    pub fn sigmoid(&self) -> Tensor {
        let result: Vec<f32> = get_equation()
            .get_data_flat_buffer(self.id)
            .iter()
            .map(|x| 1.0 / (1.0 + (-x).exp()))
            .collect();
        let tensor = Tensor::create_tensor_data_and_shape_and_operation(
            self.shape,
            result,
            Operation::Sigmoid(self.id),
        );
        return tensor;
    }
}

/// Handles calculating and passing back the gradient of a Sigmoid operation
/// The derivative of sigmoid(x) is
pub fn backward_for_sigmoid(backprop_packet: BackproagationPacket) {
    if let Operation::Sigmoid(source_id) = backprop_packet.operation {
        let in_gradient = backprop_packet
            .equation
            .get_grad_flat_buffer(backprop_packet.incoming_grad);
        let activated = backprop_packet
            .equation
            .get_data_flat_buffer(backprop_packet.incoming_grad);

        // TODO: update this to use the platform specific acceleration, if this is taking to long in the future
        let data: Vec<f32> = activated
            .iter()
            .zip(in_gradient.iter())
            .map(|(y_i, go_i)| go_i * y_i * (1.0 - y_i))
            .collect();

        backprop_packet.equation.add_tensor_grad(source_id, data);
    } else {
        panic!("Wrong operation for backward sigmoid");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::central::{Tensor, zero_all_grads};

    fn approx_equal(a: f32, b: f32, epsilon: f32) -> bool {
        (a - b).abs() <= epsilon
    }

    #[test]
    fn sigmoid_matches_expected_values_for_basic_inputs() {
        let input = Tensor::from_vec(vec![-1.0, 0.0, 1.0], vec![3]);
        let output = input.sigmoid().item();
        let expected = vec![0.26894143, 0.5, 0.7310586];

        for (actual, expected) in output.iter().zip(expected.iter()) {
            assert!(approx_equal(*actual, *expected, 1e-6));
        }
    }

    #[test]
    fn sigmoid_preserves_shape_for_two_dimensional_tensor() {
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, -1.0, -2.0], vec![2, 3]);
        let output = input.sigmoid().item();

        assert_eq!(output.shape(), &[2, 3]);
        assert!(approx_equal(output[[0, 0]], 0.7310586, 1e-6));
        assert!(approx_equal(output[[1, 2]], 0.11920292, 1e-6));
    }

    #[test]
    fn sigmoid_handles_large_magnitude_inputs() {
        let input = Tensor::from_vec(vec![-10.0, -20.0, 10.0, 20.0], vec![4]);
        let output = input.sigmoid().item();

        assert!(approx_equal(output[[0]], 4.539787e-5, 1e-8));
        assert!(approx_equal(output[[1]], 2.0611537e-9, 1e-12));
        assert!(approx_equal(output[[2]], 0.9999546, 1e-6));
        assert!(approx_equal(output[[3]], 0.99999994, 1e-7));
    }

    #[test]
    fn sigmoid_backward_matches_analytical_derivative() {
        zero_all_grads();
        let mut input = Tensor::from_vec(vec![-1.0, 0.0, 1.0, 2.0], vec![4]);
        input.set_requires_grad(true);

        let sigmoid_output = input.sigmoid();
        let sigmoid_values = sigmoid_output.item().to_owned();
        let loss = sigmoid_output.sum(vec![0], true);
        loss.backward();

        let grads = input.grad();
        for idx in 0..4 {
            let sigmoid_val = sigmoid_values[[idx]];
            let expected = sigmoid_val * (1.0 - sigmoid_val);
            assert!(approx_equal(grads[[idx]], expected, 1e-6));
        }
    }

    #[test]
    fn sigmoid_backward_through_nested_sigmoid() {
        zero_all_grads();
        let mut input = Tensor::from_vec(vec![0.5, -1.5, 3.0], vec![3]);
        input.set_requires_grad(true);

        let first = input.sigmoid();
        let first_values = first.item().to_owned();
        let second = first.sigmoid();
        let second_values = second.item().to_owned();
        let loss = second.sum(vec![0], true);
        loss.backward();

        let grads = input.grad();
        for idx in 0..3 {
            let first_val = first_values[[idx]];
            let second_val = second_values[[idx]];
            let expected = first_val * (1.0 - first_val) * second_val * (1.0 - second_val);
            assert!(approx_equal(grads[[idx]], expected, 1e-6));
        }
    }
}
