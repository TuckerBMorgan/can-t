use crate::central::get_equation;

use super::{BackproagationPacket, Operation, Tensor};

impl Tensor {
    /// Applies the sin function element-wise to the tensor
    /// Returns a new tensor with the same shape where each element x is replaced with sin(x)
    pub fn sin(&self) -> Tensor {
        let result: Vec<f32> = get_equation()
            .get_data_flat_buffer(self.id)
            .iter()
            .map(|x| x.sin())
            .collect();
        let tensor = Tensor::create_tensor_data_and_shape_and_operation(
            self.shape,
            result,
            Operation::Sin(self.id),
        );
        return tensor;
    }
}

/// Handles calculating and passing back the gradient of a sin operation
/// The derivative of sin(x) is cos(x)
pub fn backward_for_sin(backprop_packet: BackproagationPacket) {
    if let Operation::Sin(source_id) = backprop_packet.operation {
        let in_gradient = backprop_packet
            .equation
            .get_grad_flat_buffer(backprop_packet.incoming_grad);
        let source_data = backprop_packet.equation.get_data_flat_buffer(source_id);

        // TODO: update this to use the platform specific acceleration, if this is taking to long in the future
        let updated: Vec<f32> = in_gradient
            .iter()
            .zip(source_data.iter())
            .map(|(grad, &x)| grad * x.cos())
            .collect();

        backprop_packet.equation.add_tensor_grad(source_id, updated);
    } else {
        panic!("Wrong operation for backward sin");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::central::zero_all_grads;
    use std::f32::consts::{FRAC_PI_2, PI};

    fn approx_equal(a: f32, b: f32, epsilon: f32) -> bool {
        (a - b).abs() < epsilon
    }

    #[test]
    fn sin_backward_matches_cosine() {
        zero_all_grads();
        let mut input = Tensor::from_vec(vec![0.0, FRAC_PI_2], vec![2]);
        input.set_requires_grad(true);

        let output = input.sin();
        let loss = output.sum(vec![0], true);
        loss.backward();

        let gradients = input.grad();
        assert!(approx_equal(gradients[[0]], 1.0, 1e-6));
        assert!(approx_equal(gradients[[1]], 0.0, 1e-6));
    }

    #[test]
    fn sin_forward_matches_reference_values() {
        zero_all_grads();
        let values = vec![0.0, PI / 6.0, PI / 2.0];
        let expected = values.iter().map(|v| v.sin()).collect::<Vec<f32>>();

        let tensor = Tensor::from_vec(values.clone(), vec![values.len()]);
        let result = tensor.sin().item();

        for (idx, exp) in expected.iter().enumerate() {
            assert!(approx_equal(result[[idx]], *exp, 1e-6));
        }
    }

    #[test]
    fn sin_backward_scales_with_incoming_gradient() {
        zero_all_grads();
        let mut input = Tensor::from_vec(vec![0.25, 1.0], vec![2]);
        input.set_requires_grad(true);

        let weights = Tensor::from_vec(vec![2.0, -3.0], vec![2]);
        let scaled = input.sin() * weights;
        let loss = scaled.sum(vec![0], true);
        loss.backward();

        let gradients = input.grad();
        let expected0 = 0.25f32.cos() * 2.0;
        let expected1 = 1.0f32.cos() * -3.0;

        assert!(approx_equal(gradients[[0]], expected0, 1e-6));
        assert!(approx_equal(gradients[[1]], expected1, 1e-6));
    }
}
