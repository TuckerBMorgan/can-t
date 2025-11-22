use crate::central::*;

impl Tensor {
    /// clamps each element of the tensor between min and max
    /// It does this into a new tensor, not in place
    pub fn clamp(&self, min: f32, max: f32) -> Tensor {
        assert!(min <= max, "Clamp min must be less than or equal to max");
        // Just map each of the values from the original tensor to a new one
        let result: Vec<f32> = get_equation()
            .get_data_flat_buffer(self.id)
            .iter()
            .map(|x| x.clamp(min, max))
            .collect();

        let tensor = Tensor::create_tensor_data_and_shape_and_operation(
            self.shape,
            result,
            Operation::Clamp(self.id, min, max),
        );
        return tensor;
    }
}

pub fn backward_for_clamp(packet: BackproagationPacket) {
    if let Operation::Clamp(from, min, max) = packet.operation {
        let in_gradient = packet.equation.get_grad_flat_buffer(packet.incoming_grad);
        let source_data = packet.equation.get_data_flat_buffer(from);

        // Only those values that are within then clampped range(min, max) keep their grad
        // those values that did get clamped, get no grad
        let updated: Vec<f32> = in_gradient
            .iter()
            .zip(source_data.iter())
            .map(|(grad, &x)| {
                if x < max && x > min {
                    return *grad;
                } else {
                    return 0.0;
                }
            })
            .collect();

        packet.equation.add_tensor_grad(from, updated);
    } else {
        panic!("Wrong operation for backward clamp");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_equal(a: f32, b: f32, epsilon: f32) -> bool {
        (a - b).abs() < epsilon
    }

    #[test]
    fn clamp_forward_handles_fractional_bounds() {
        zero_all_grads();
        let input = Tensor::from_vec(vec![-2.0, -0.25, 0.25, 2.0], vec![4]);
        let output = input.clamp(-0.5, 0.5);

        let result = output.item();
        assert!(approx_equal(result[[0]], -0.5, 1e-6));
        assert!(approx_equal(result[[1]], -0.25, 1e-6));
        assert!(approx_equal(result[[2]], 0.25, 1e-6));
        assert!(approx_equal(result[[3]], 0.5, 1e-6));
    }

    #[test]
    fn clamp_backward_masks_outside_bounds() {
        zero_all_grads();
        let mut input = Tensor::from_vec(vec![-1.0, -0.25, 0.25, 1.5], vec![4]);
        input.set_requires_grad(true);

        let clamped = input.clamp(-0.5, 0.5);
        let loss = clamped.sum(vec![0], true);
        loss.backward();

        let gradients = input.grad();
        assert!(approx_equal(gradients[[0]], 0.0, 1e-6));
        assert!(approx_equal(gradients[[1]], 1.0, 1e-6));
        assert!(approx_equal(gradients[[2]], 1.0, 1e-6));
        assert!(approx_equal(gradients[[3]], 0.0, 1e-6));
    }
}
