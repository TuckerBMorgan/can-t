use crate::central::{MAX_STACK_AMOUNT, get_equation};

use super::{BackproagationPacket, Operation, Tensor};

impl Tensor {
    pub fn stack(&self, others: Vec<Tensor>, dimension: usize) -> Tensor {
        let first_shape = self.shape;
        for other in &others {
            assert!(
                first_shape == other.shape,
                "All tensors much share the same share for stack to work"
            );
        }

        assert!(
            others.len() > 0,
            "There must be at least one other tensor to stack with"
        );

        // we add a new dimension at the desired location, that new dimension is the total number of dimension we are stacking
        // which is the length of the others vector + 1 to cover the tensor we are inside of so to speak
        let new_shape = self
            .shape
            .add_dimension_at_index(others.len() + 1, dimension);

        let mut output_tensor = self.cat(others[0], dimension);

        for i in 1..others.len() {
            output_tensor = output_tensor.cat(others[i], dimension);
        }

        return output_tensor.reshape(new_shape);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::central::{Shape, zero_all_grads};

    fn approx_equal(a: f32, b: f32, epsilon: f32) -> bool {
        (a - b).abs() < epsilon
    }

    #[test]
    fn stack_two_tensors_dimension_0() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let t2 = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);

        let result = t1.stack(vec![t2], 0);

        // Stacking two [2, 2] tensors along dimension 0 should give [2, 2, 2]
        assert_eq!(result.shape.dimensions(), vec![2, 2, 2]);
        let data = result.item().into_raw_vec();
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn stack_two_tensors_dimension_1() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let t2 = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);

        let result = t1.stack(vec![t2], 1);

        // Stacking two [2, 2] tensors along dimension 1 should give [2, 2, 2]
        assert_eq!(result.shape.dimensions(), vec![2, 2, 2]);
        let data = result.item().into_raw_vec();
        assert_eq!(data, vec![1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0]);
    }

    #[test]
    fn stack_three_tensors_dimension_0() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![1, 2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0], vec![1, 2]);
        let t3 = Tensor::from_vec(vec![5.0, 6.0], vec![1, 2]);

        let result = t1.stack(vec![t2, t3], 0);

        // Stacking three [1, 2] tensors along dimension 0 should give [3, 1, 2]
        assert_eq!(result.shape.dimensions(), vec![3, 1, 2]);
        let data = result.item().into_raw_vec();
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn stack_multiple_1d_tensors() {
        let t1 = Tensor::from_vec(vec![1.0], vec![1]);
        let t2 = Tensor::from_vec(vec![2.0], vec![1]);
        let t3 = Tensor::from_vec(vec![3.0], vec![1]);
        let t4 = Tensor::from_vec(vec![4.0], vec![1]);

        let result = t1.stack(vec![t2, t3, t4], 0);

        // Stacking four [1] tensors along dimension 0 should give [4, 1]
        assert_eq!(result.shape.dimensions(), vec![4, 1]);
        let data = result.item().into_raw_vec();
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn stack_backward_propagates_gradients() {
        zero_all_grads();

        let mut t1 = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        t1.set_requires_grad(true);
        let mut t2 = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        t2.set_requires_grad(true);

        let result = t1.stack(vec![t2], 0);
        // Use mean instead of sum to get a scalar loss
        let loss = result.mean(vec![0, 1, 2]);
        loss.backward();

        // All gradients should be 1/8 since we're taking the mean of 8 elements
        let expected_grad = 1.0 / 8.0;
        for value in t1.grad().iter() {
            assert!(approx_equal(*value, expected_grad, 1e-6));
        }

        for value in t2.grad().iter() {
            assert!(approx_equal(*value, expected_grad, 1e-6));
        }
    }

    #[test]
    fn stack_backward_multiple_tensors() {
        zero_all_grads();

        let mut t1 = Tensor::from_vec(vec![1.0, 2.0], vec![1, 2]);
        t1.set_requires_grad(true);
        let mut t2 = Tensor::from_vec(vec![3.0, 4.0], vec![1, 2]);
        t2.set_requires_grad(true);
        let mut t3 = Tensor::from_vec(vec![5.0, 6.0], vec![1, 2]);
        t3.set_requires_grad(true);

        let result = t1.stack(vec![t2, t3], 0);
        // Use mean instead of sum to get a scalar loss
        let loss = result.mean(vec![0, 1, 2]);
        loss.backward();

        // All gradients should be 1/6 since we're taking the mean of 6 elements
        let expected_grad = 1.0 / 6.0;
        for value in t1.grad().iter() {
            assert!(approx_equal(*value, expected_grad, 1e-6));
        }
        for value in t2.grad().iter() {
            assert!(approx_equal(*value, expected_grad, 1e-6));
        }
        for value in t3.grad().iter() {
            assert!(approx_equal(*value, expected_grad, 1e-6));
        }
    }

    #[test]
    fn stack_backward_with_weighted_sum() {
        zero_all_grads();

        let mut t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        t1.set_requires_grad(true);
        let mut t2 = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        t2.set_requires_grad(true);

        let result = t1.stack(vec![t2], 0);
        // Result has shape [2, 2], so weights must also be [2, 2]
        let weights = Tensor::from_vec(vec![2.0, 3.0, 2.0, 3.0], vec![2, 2]);
        let weighted = result * weights;
        // Use mean to get a scalar loss
        let loss = weighted.mean(vec![0, 1]);
        loss.backward();

        // Gradients should be the weights divided by the number of elements (4)
        let t1_grad = t1.grad();
        assert!(approx_equal(t1_grad[[0]], 2.0 / 4.0, 1e-6));
        assert!(approx_equal(t1_grad[[1]], 3.0 / 4.0, 1e-6));

        let t2_grad = t2.grad();
        assert!(approx_equal(t2_grad[[0]], 2.0 / 4.0, 1e-6));
        assert!(approx_equal(t2_grad[[1]], 3.0 / 4.0, 1e-6));
    }

    #[test]
    #[should_panic(expected = "All tensors much share the same share for stack to work")]
    fn stack_panics_on_shape_mismatch() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::from_vec(vec![3.0, 4.0, 5.0], vec![3]);

        t1.stack(vec![t2], 0);
    }

    #[test]
    #[should_panic(expected = "There must be at least one other tensor to stack with")]
    fn stack_panics_on_empty_vector() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);

        t1.stack(vec![], 0);
    }
}
