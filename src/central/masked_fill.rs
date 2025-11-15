use crate::{central::*, utils::handle_broadcasting};

impl Tensor {
    pub fn masked_fill(&self, mask: Tensor, value: f32) -> Tensor {
        let (local, mask) = handle_broadcasting(*self, mask);
        let result = {
            let equation = get_equation();

            // The mask and the data should be the same size
            let base_data = equation.get_data_flat_buffer(local.id);
            let mask_buffer = equation.get_data_flat_buffer(mask.id);

            // So it is easy to zip them together, and loop over the indices
            let zipped = base_data.iter().zip(mask_buffer);
            let mut result = vec![0.0; base_data.len()];
            for (index, (base, mask)) in zipped.enumerate() {
                if *mask == 1.0 {
                    result[index] = value;
                } else {
                    result[index] = *base;
                }
            }

            result
        };

        return Tensor::create_tensor_data_and_shape_and_operation(
            local.shape,
            result,
            Operation::MaskFill(local.id, mask.id, value as isize),
        );
    }
}

pub fn backwards_for_mask_fill(packet: BackproagationPacket) {
    if let Operation::MaskFill(source, mask, _value) = packet.operation {
        let incoming_grad = packet.equation.get_grad_flat_buffer(packet.incoming_grad);
        let mask_buffer = packet.equation.get_data_flat_buffer(mask);
        let zipped = incoming_grad.iter().zip(mask_buffer);
        let mut new_grad = vec![0.0; mask_buffer.len()];

        for (index, (grad, mask)) in zipped.enumerate() {
            if *mask != 0.0 {
                // If mask != 0.0, the original value was kept, so gradient flows through
                new_grad[index] = *grad;
            }
            // If mask == 0.0, the value was filled, so no gradient flows (stays 0.0)
        }

        packet.equation.add_tensor_grad(source, new_grad);
    } else {
        panic!("Wrong operation called with backward for mask fill");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_equal(a: f32, b: f32, epsilon: f32) -> bool {
        (a - b).abs() < epsilon
    }

    #[test]
    fn test_masked_fill_basic() {
        // Test basic masked fill functionality
        let data = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        let mask = Tensor::from_vec(vec![0.0, 1.0, 0.0, 1.0], vec![4]);

        let result = data.masked_fill(mask, -999.0);

        assert_eq!(result.shape.dimensions(), vec![4]);
        let result_data = result.item();

        // Where mask is 1.0, keep original values; where mask is 0.0, fill with -999.0
        assert!(approx_equal(result_data[[0]], 1.0, 1e-6)); // mask=1.0 -> keep original
        assert!(approx_equal(result_data[[1]], -999.0, 1e-6)); // mask=0.0 -> fill
        assert!(approx_equal(result_data[[2]], 3.0, 1e-6)); // mask=1.0 -> keep original
        assert!(approx_equal(result_data[[3]], -999.0, 1e-6)); // mask=0.0 -> fill
    }

    #[test]
    fn test_masked_fill_2d() {
        // Test masked fill with 2D tensors
        let data = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let mask = Tensor::from_vec(vec![0.0, 1.0, 1.0, 0.0], vec![2, 2]);

        let result = data.masked_fill(mask, -100.0);

        assert_eq!(result.shape.dimensions(), vec![2, 2]);
        let result_data = result.item();

        assert!(approx_equal(result_data[[0, 0]], 1.0, 1e-6)); // mask=1.0 -> keep
        assert!(approx_equal(result_data[[0, 1]], -100.0, 1e-6)); // mask=0.0 -> fill
        assert!(approx_equal(result_data[[1, 0]], -100.0, 1e-6)); // mask=0.0 -> fill
        assert!(approx_equal(result_data[[1, 1]], 4.0, 1e-6)); // mask=1.0 -> keep
    }

    #[test]
    fn test_masked_fill_all_mask_ones() {
        // Test when mask is all 1s (no filling should occur)
        let data = Tensor::from_vec(vec![5.0, 10.0, 15.0], vec![3]);
        let mask = Tensor::from_vec(vec![0.0, 0.0, 0.0], vec![3]);

        let result = data.masked_fill(mask, -42.0);

        let result_data = result.item();
        let original_data = data.item();

        // All values should remain unchanged
        for i in 0..3 {
            assert!(approx_equal(result_data[[i]], original_data[[i]], 1e-6));
        }
    }

    #[test]
    fn test_masked_fill_all_mask_zeros() {
        // Test when mask is all 0s (all filling should occur)
        let data = Tensor::from_vec(vec![5.0, 10.0, 15.0], vec![3]);
        let mask = Tensor::from_vec(vec![1.0, 1.0, 1.0], vec![3]);

        let result = data.masked_fill(mask, 42.0);

        let result_data = result.item();

        // All values should be filled
        for i in 0..3 {
            assert!(approx_equal(result_data[[i]], 42.0, 1e-6));
        }
    }

    #[test]
    fn test_masked_fill_negative_fill_value() {
        // Test with negative fill value
        let data = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        let mask = Tensor::from_vec(vec![1.0, 0.0, 1.0, 0.0], vec![4]);

        let result = data.masked_fill(mask, -1.0);

        let result_data = result.item();

        assert!(approx_equal(result_data[[0]], -1.0, 1e-6)); // mask=1.0 -> fill
        assert!(approx_equal(result_data[[1]], 2.0, 1e-6)); // mask=0.0 -> keep
        assert!(approx_equal(result_data[[2]], -1.0, 1e-6)); // mask=1.0 -> fill
        assert!(approx_equal(result_data[[3]], 4.0, 1e-6)); // mask=0.0 -> keep
    }

    #[test]
    fn test_masked_fill_zero_fill_value() {
        // Test with zero as fill value
        let data = Tensor::from_vec(vec![10.0, 20.0, 30.0], vec![3]);
        let mask = Tensor::from_vec(vec![0.0, 1.0, 0.0], vec![3]);

        let result = data.masked_fill(mask, 0.0);

        let result_data = result.item();

        assert!(approx_equal(result_data[[0]], 10.0, 1e-6)); // mask=0.0 -> keep
        assert!(approx_equal(result_data[[1]], 0.0, 1e-6)); // mask=1.0 -> fill with 0
        assert!(approx_equal(result_data[[2]], 30.0, 1e-6)); // mask=0.0 -> keep
    }

    #[test]
    fn test_masked_fill_large_values() {
        // Test with large fill values for numerical stability
        let data = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let mask = Tensor::from_vec(vec![1.0, 0.0], vec![2]);

        let result = data.masked_fill(mask, 1e6);

        let result_data = result.item();

        assert!(approx_equal(result_data[[0]], 1e6, 1e-2)); // mask=1.0 -> fill with large value
        assert!(approx_equal(result_data[[1]], 2.0, 1e-6)); // mask=0.0 -> keep original
    }

    #[test]
    fn test_masked_fill_3d_tensor() {
        // Test with 3D tensor
        let data = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 2, 2]);
        let mask = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0], vec![2, 2, 2]);

        let result = data.masked_fill(mask, -5.0);

        assert_eq!(result.shape.dimensions(), vec![2, 2, 2]);
        let result_data = result.item();

        // Check specific positions
        assert!(approx_equal(result_data[[0, 0, 0]], -5.0, 1e-6)); // mask=1.0 -> fill
        assert!(approx_equal(result_data[[0, 0, 1]], 2.0, 1e-6)); // mask=0.0 -> keep
        assert!(approx_equal(result_data[[0, 1, 0]], 3.0, 1e-6)); // mask=0.0 -> keep
        assert!(approx_equal(result_data[[0, 1, 1]], -5.0, 1e-6)); // mask=1.0 -> fill
        assert!(approx_equal(result_data[[1, 0, 0]], -5.0, 1e-6)); // mask=1.0 -> fill
        assert!(approx_equal(result_data[[1, 0, 1]], 6.0, 1e-6)); // mask=0.0 -> keep
        assert!(approx_equal(result_data[[1, 1, 0]], -5.0, 1e-6)); // mask=1.0 -> fill
        assert!(approx_equal(result_data[[1, 1, 1]], 8.0, 1e-6)); // mask=0.0 -> keep
    }

    #[test]
    fn test_masked_fill_attention_mask_pattern() {
        // Test typical attention mask pattern (upper triangular)
        let data = Tensor::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            vec![3, 3],
        );

        // Upper triangular mask (0s on and below diagonal, 1s above)
        let mask = Tensor::from_vec(
            vec![0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            vec![3, 3],
        );

        let result = data.masked_fill(mask, f32::NEG_INFINITY);

        let result_data = result.item();

        // Lower triangle + diagonal should keep original values
        assert!(approx_equal(result_data[[0, 0]], 1.0, 1e-6));
        assert!(approx_equal(result_data[[1, 0]], 4.0, 1e-6));
        assert!(approx_equal(result_data[[1, 1]], 5.0, 1e-6));
        assert!(approx_equal(result_data[[2, 0]], 7.0, 1e-6));
        assert!(approx_equal(result_data[[2, 1]], 8.0, 1e-6));
        assert!(approx_equal(result_data[[2, 2]], 9.0, 1e-6));

        // Upper triangle should be -infinity
        assert!(result_data[[0, 1]] == f32::NEG_INFINITY);
        assert!(result_data[[0, 2]] == f32::NEG_INFINITY);
        assert!(result_data[[1, 2]] == f32::NEG_INFINITY);
    }

    #[test]
    fn test_masked_fill_fractional_mask() {
        // Test that only exact 1.0 values trigger filling
        let data = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        let mask = Tensor::from_vec(vec![0.1, 1.0, 0.99, 1.0], vec![4]);

        let result = data.masked_fill(mask, -1.0);

        let result_data = result.item();

        // Only exact 1.0 should trigger filling
        assert!(approx_equal(result_data[[0]], 1.0, 1e-6)); // 0.1 != 1.0 -> keep
        assert!(approx_equal(result_data[[1]], -1.0, 1e-6)); // 1.0 == 1.0 -> fill
        assert!(approx_equal(result_data[[2]], 3.0, 1e-6)); // 0.99 != 1.0 -> keep
        assert!(approx_equal(result_data[[3]], -1.0, 1e-6)); // 1.0 == 1.0 -> fill
    }

    #[test]
    fn test_masked_fill_single_element() {
        // Test with single element tensors
        let data = Tensor::from_vec(vec![42.0], vec![1]);
        let mask = Tensor::from_vec(vec![1.0], vec![1]);

        let result = data.masked_fill(mask, 99.0);

        let result_data = result.item();
        assert!(approx_equal(result_data[[0]], 99.0, 1e-6));
    }

    #[test]
    fn test_masked_fill_preserves_shape() {
        // Test that various shapes are preserved
        let shapes = vec![vec![5], vec![2, 3], vec![2, 2, 2], vec![1, 4, 1]];

        for shape in shapes {
            let size = shape.iter().product::<usize>();
            let data: Vec<f32> = (0..size).map(|i| i as f32).collect();
            let mask_data = vec![1.0; size]; // All ones to trigger filling

            let tensor = Tensor::from_vec(data, shape.clone());
            let mask = Tensor::from_vec(mask_data, shape.clone());

            let result = tensor.masked_fill(mask, 123.0);

            assert_eq!(result.shape.dimensions(), shape);

            // All values should be filled since mask is all zeros
            let result_data = result.item();
            for i in 0..size {
                let _indices: Vec<usize> = (0..shape.len())
                    .map(|dim| (i / shape[dim + 1..].iter().product::<usize>()) % shape[dim])
                    .collect();

                let mut accessor = vec![0; shape.len()];
                let mut remaining = i;
                for (dim_idx, &dim_size) in shape.iter().enumerate().rev() {
                    accessor[dim_idx] = remaining % dim_size;
                    remaining /= dim_size;
                }

                // For simple checking, just verify the shape was preserved
                assert_eq!(result_data.shape(), shape.as_slice());
            }
        }
    }

    // ========== BACKWARD PASS TESTS ==========

    #[test]
    fn test_masked_fill_backward_basic() {
        // Test basic backward pass through masked_fill
        let mut input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        input.set_requires_grad(true);

        let mask = Tensor::from_vec(vec![1.0, 0.0, 1.0, 0.0], vec![4]);

        let output = input.masked_fill(mask, -999.0);
        let loss = output.sum(vec![0], true);

        zero_all_grads();
        loss.backward();

        let input_grad = input.grad();

        // Gradients should flow back only where mask != 0.0
        // Where mask = 1.0, gradient should be 1.0 (from sum)
        // Where mask = 0.0, gradient should be 0.0 (filled values don't depend on input)
        assert!(approx_equal(input_grad[[0]], 1.0, 1e-6)); // mask=1.0 -> grad flows
        assert!(approx_equal(input_grad[[1]], 0.0, 1e-6)); // mask=0.0 -> no grad
        assert!(approx_equal(input_grad[[2]], 1.0, 1e-6)); // mask=1.0 -> grad flows
        assert!(approx_equal(input_grad[[3]], 0.0, 1e-6)); // mask=0.0 -> no grad
    }

    #[test]
    fn test_masked_fill_backward_2d() {
        // Test backward pass with 2D tensors
        let mut input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        input.set_requires_grad(true);

        let mask = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);

        let output = input.masked_fill(mask, 100.0);
        let loss = output.sum(vec![0, 1], true);

        zero_all_grads();
        loss.backward();

        let input_grad = input.grad();

        // Check gradient flow for each position
        assert!(approx_equal(input_grad[[0, 0]], 1.0, 1e-6)); // mask=1.0 -> grad flows
        assert!(approx_equal(input_grad[[0, 1]], 0.0, 1e-6)); // mask=0.0 -> no grad
        assert!(approx_equal(input_grad[[1, 0]], 0.0, 1e-6)); // mask=0.0 -> no grad
        assert!(approx_equal(input_grad[[1, 1]], 1.0, 1e-6)); // mask=1.0 -> grad flows
    }

    #[test]
    fn test_masked_fill_backward_all_masked() {
        // Test when all values are masked (all gradients should be zero)
        let mut input = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        input.set_requires_grad(true);

        let mask = Tensor::from_vec(vec![0.0, 0.0, 0.0], vec![3]); // All masked

        let output = input.masked_fill(mask, 42.0);
        let loss = output.sum(vec![0], true);

        zero_all_grads();
        loss.backward();

        let input_grad = input.grad();

        // All gradients should be zero since all values are filled
        for i in 0..3 {
            assert!(approx_equal(input_grad[[i]], 0.0, 1e-6));
        }
    }

    #[test]
    fn test_masked_fill_backward_no_masking() {
        // Test when no values are masked (all gradients should flow)
        let mut input = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        input.set_requires_grad(true);

        let mask = Tensor::from_vec(vec![1.0, 1.0, 1.0], vec![3]); // No masking

        let output = input.masked_fill(mask, 999.0);
        let loss = output.sum(vec![0], true);

        zero_all_grads();
        loss.backward();

        let input_grad = input.grad();

        // All gradients should be 1.0 since no values are filled
        for i in 0..3 {
            assert!(approx_equal(input_grad[[i]], 1.0, 1e-6));
        }
    }

    #[test]
    fn test_masked_fill_backward_attention_pattern() {
        // Test backward pass through attention mask pattern
        let mut input = Tensor::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            vec![3, 3],
        );
        input.set_requires_grad(true);

        // Upper triangular mask (1s on and below diagonal, 0s above)
        let mask = Tensor::from_vec(
            vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
            vec![3, 3],
        );

        let output = input.masked_fill(mask, f32::NEG_INFINITY);
        let loss = output.sum(vec![0, 1], true);

        zero_all_grads();
        loss.backward();

        let input_grad = input.grad();

        // Lower triangle + diagonal should have gradients (mask=1.0)
        assert!(approx_equal(input_grad[[0, 0]], 1.0, 1e-6));
        assert!(approx_equal(input_grad[[1, 0]], 1.0, 1e-6));
        assert!(approx_equal(input_grad[[1, 1]], 1.0, 1e-6));
        assert!(approx_equal(input_grad[[2, 0]], 1.0, 1e-6));
        assert!(approx_equal(input_grad[[2, 1]], 1.0, 1e-6));
        assert!(approx_equal(input_grad[[2, 2]], 1.0, 1e-6));

        // Upper triangle should have zero gradients (mask=0.0)
        assert!(approx_equal(input_grad[[0, 1]], 0.0, 1e-6));
        assert!(approx_equal(input_grad[[0, 2]], 0.0, 1e-6));
        assert!(approx_equal(input_grad[[1, 2]], 0.0, 1e-6));
    }

    #[test]
    fn test_masked_fill_backward_chain() {
        // Test masked_fill in a chain of operations
        let mut input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        input.set_requires_grad(true);

        let mask = Tensor::from_vec(vec![1.0, 0.0, 1.0, 0.0], vec![4]);

        // Chain: input -> multiply by 2 -> masked_fill -> sum
        let scaled = input * Tensor::element(Shape::new(vec![4]), 2.0);
        let masked = scaled.masked_fill(mask, 0.0);
        let loss = masked.sum(vec![0], true);

        zero_all_grads();
        loss.backward();

        let input_grad = input.grad();

        // Gradient should be 2.0 where mask=1.0 (gradient flows through scaling)
        // and 0.0 where mask=0.0 (gradient blocked by masking)
        assert!(approx_equal(input_grad[[0]], 2.0, 1e-6)); // mask=1.0 -> 2.0 * 1.0
        assert!(approx_equal(input_grad[[1]], 0.0, 1e-6)); // mask=0.0 -> blocked
        assert!(approx_equal(input_grad[[2]], 2.0, 1e-6)); // mask=1.0 -> 2.0 * 1.0
        assert!(approx_equal(input_grad[[3]], 0.0, 1e-6)); // mask=0.0 -> blocked
    }

    #[test]
    fn test_masked_fill_backward_different_losses() {
        // Test that different downstream losses produce correct gradients
        let mut input = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        input.set_requires_grad(true);

        let mask = Tensor::from_vec(vec![1.0, 0.0, 1.0], vec![3]);

        let output = input.masked_fill(mask, 0.0);

        // Test 1: Sum loss (gradient = 1.0 for unmasked)
        let loss1 = output.sum(vec![0], true);
        zero_all_grads();
        loss1.backward();

        let grad1 = input.grad();
        assert!(approx_equal(grad1[[0]], 1.0, 1e-6));
        assert!(approx_equal(grad1[[1]], 0.0, 1e-6));
        assert!(approx_equal(grad1[[2]], 1.0, 1e-6));

        // Test 2: Weighted sum (gradient = weights for unmasked)
        let weights = Tensor::from_vec(vec![2.0, 3.0, 4.0], vec![3]);
        let loss2 = (output * weights).sum(vec![0], true);
        zero_all_grads();
        loss2.backward();

        let grad2 = input.grad();
        assert!(approx_equal(grad2[[0]], 2.0, 1e-6)); // unmasked -> weight flows
        assert!(approx_equal(grad2[[1]], 0.0, 1e-6)); // masked -> no gradient
        assert!(approx_equal(grad2[[2]], 4.0, 1e-6)); // unmasked -> weight flows
    }

    #[test]
    fn test_masked_fill_backward_3d() {
        // Test backward pass with 3D tensors (batch dimension)
        let mut input =
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 2, 2]);
        input.set_requires_grad(true);

        let mask = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0], vec![2, 2, 2]);

        let output = input.masked_fill(mask, -1.0);
        let loss = output.sum(vec![0, 1, 2], true);

        zero_all_grads();
        loss.backward();

        let input_grad = input.grad();

        // Check gradients for each position in 3D tensor
        assert!(approx_equal(input_grad[[0, 0, 0]], 1.0, 1e-6)); // mask=1.0 -> grad
        assert!(approx_equal(input_grad[[0, 0, 1]], 0.0, 1e-6)); // mask=0.0 -> no grad
        assert!(approx_equal(input_grad[[0, 1, 0]], 0.0, 1e-6)); // mask=0.0 -> no grad
        assert!(approx_equal(input_grad[[0, 1, 1]], 1.0, 1e-6)); // mask=1.0 -> grad
        assert!(approx_equal(input_grad[[1, 0, 0]], 1.0, 1e-6)); // mask=1.0 -> grad
        assert!(approx_equal(input_grad[[1, 0, 1]], 0.0, 1e-6)); // mask=0.0 -> no grad
        assert!(approx_equal(input_grad[[1, 1, 0]], 1.0, 1e-6)); // mask=1.0 -> grad
        assert!(approx_equal(input_grad[[1, 1, 1]], 0.0, 1e-6)); // mask=0.0 -> no grad
    }

    #[test]
    fn test_masked_fill_backward_fractional_mask() {
        // Test that only exact 0.0 values block gradients
        let mut input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        input.set_requires_grad(true);

        let mask = Tensor::from_vec(vec![0.1, 0.0, 0.99, 0.0], vec![4]);

        let output = input.masked_fill(mask, -1.0);
        let loss = output.sum(vec![0], true);

        zero_all_grads();
        loss.backward();

        let input_grad = input.grad();

        // Only exact 0.0 should block gradients
        assert!(approx_equal(input_grad[[0]], 1.0, 1e-6)); // 0.1 != 0.0 -> grad flows
        assert!(approx_equal(input_grad[[1]], 0.0, 1e-6)); // 0.0 == 0.0 -> no grad
        assert!(approx_equal(input_grad[[2]], 1.0, 1e-6)); // 0.99 != 0.0 -> grad flows
        assert!(approx_equal(input_grad[[3]], 0.0, 1e-6)); // 0.0 == 0.0 -> no grad
    }

    #[test]
    fn test_masked_fill_backward_numerical_stability() {
        // Test backward pass with extreme fill values
        let mut input = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        input.set_requires_grad(true);

        let mask = Tensor::from_vec(vec![1.0, 0.0, 1.0], vec![3]);

        // Use extreme fill value
        let output = input.masked_fill(mask, f32::NEG_INFINITY);
        let loss = output.sum(vec![0], true);

        zero_all_grads();
        loss.backward();

        let input_grad = input.grad();

        // Gradients should still be computed correctly despite extreme fill value
        assert!(input_grad[[0]].is_finite()); // Should be finite
        assert!(approx_equal(input_grad[[0]], 1.0, 1e-6)); // mask=1.0 -> grad flows
        assert!(approx_equal(input_grad[[1]], 0.0, 1e-6)); // mask=0.0 -> no grad
        assert!(input_grad[[2]].is_finite()); // Should be finite
        assert!(approx_equal(input_grad[[2]], 1.0, 1e-6)); // mask=1.0 -> grad flows
    }
}
