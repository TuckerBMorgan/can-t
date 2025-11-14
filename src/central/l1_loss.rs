use crate::central::*;

impl Tensor {
    /// Computes the element-wise L1 loss (mean absolute error) between two tensors.
    /// All dimensions are reduced so the return value is a scalar loss.
    pub fn l1_loss(&self, other: Tensor) -> Tensor {
        let diff = *self - other;

        // |x| = relu(x) + relu(-x), which keeps gradients well-behaved around 0.
        let mut positive = diff;
        let positive = positive.relu();

        let mut negative = -diff;
        let negative = negative.relu();

        let abs_diff = positive + negative;
        let axes: Vec<usize> = (0..abs_diff.shape.number_of_dimension()).collect();

        abs_diff.mean(axes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_equal(a: f32, b: f32, epsilon: f32) -> bool {
        (a - b).abs() <= epsilon
    }

    #[test]
    fn l1_loss_matches_manual_mae_for_vector() {
        let prediction = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        let target = Tensor::from_vec(vec![0.5, 2.5, 2.0, 6.0], vec![4]);

        let loss = prediction.l1_loss(target);
        let result = loss.item();

        let expected = (0.5 + 0.5 + 1.0 + 2.0) / 4.0;
        assert!(approx_equal(result[[0]], expected, 1e-6));
    }

    #[test]
    fn l1_loss_reduces_across_all_dimensions() {
        let prediction = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let target = Tensor::from_vec(vec![0.0, 2.0, 0.0, 4.0, 7.0, 3.0], vec![2, 3]);

        let loss = prediction.l1_loss(target);
        let result = loss.item();

        let differences = vec![1.0, 0.0, 3.0, 0.0, 2.0, 3.0];
        let expected = differences.iter().sum::<f32>() / differences.len() as f32;
        assert!(approx_equal(result[[0]], expected, 1e-6));
    }

    #[test]
    fn l1_loss_is_zero_when_tensors_equal() {
        let values = vec![3.0, -1.0, 4.0, -2.0];
        let prediction = Tensor::from_vec(values.clone(), vec![2, 2]);
        let target = Tensor::from_vec(values, vec![2, 2]);

        let loss = prediction.l1_loss(target);
        let result = loss.item();

        assert!(approx_equal(result[[0]], 0.0, 1e-6));
    }

    #[test]
    fn l1_loss_matches_manual_mae_for_batched_tensor() {
        let prediction = Tensor::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0],
            vec![2, 2, 2],
        );
        let target = Tensor::from_vec(
            vec![0.0, 0.0, 4.0, 2.0, -2.0, -1.0, -4.0, -2.0],
            vec![2, 2, 2],
        );

        let loss = prediction.l1_loss(target);
        let result = loss.item();

        let differences = vec![
            1.0, 2.0, // (0, 0, :)
            1.0, 2.0, // (0, 1, :)
            1.0, 1.0, // (1, 0, :)
            1.0, 2.0, // (1, 1, :)
        ];
        let expected = differences.iter().sum::<f32>() / differences.len() as f32;
        assert!(approx_equal(result[[0]], expected, 1e-6));
    }

    #[test]
    fn l1_loss_backward_produces_sign_gradients() {
        let mut prediction = Tensor::from_vec(vec![1.0, -2.0, 0.5], vec![3]);
        let mut target = Tensor::from_vec(vec![0.0, -1.0, 0.5], vec![3]);

        prediction.set_requires_grad(true);
        target.set_requires_grad(true);

        let loss = prediction.l1_loss(target);
        loss.backward();

        let prediction_grad = prediction.grad();
        assert!(approx_equal(prediction_grad[[0]], 1.0 / 3.0, 1e-6));
        assert!(approx_equal(prediction_grad[[1]], -1.0 / 3.0, 1e-6));
        assert!(approx_equal(prediction_grad[[2]], 0.0, 1e-6));

        let target_grad = target.grad();
        assert!(approx_equal(target_grad[[0]], -1.0 / 3.0, 1e-6));
        assert!(approx_equal(target_grad[[1]], 1.0 / 3.0, 1e-6));
        assert!(approx_equal(target_grad[[2]], 0.0, 1e-6));
    }

    #[test]
    fn l1_loss_backward_zero_difference_leaves_grads_zero() {
        let values = vec![0.5, -1.5, 2.0, -3.0];
        let mut prediction = Tensor::from_vec(values.clone(), vec![2, 2]);
        let mut target = Tensor::from_vec(values, vec![2, 2]);

        prediction.set_requires_grad(true);
        target.set_requires_grad(true);

        let loss = prediction.l1_loss(target);
        loss.backward();

        let prediction_grad = prediction.grad();
        let target_grad = target.grad();
        for idx in 0..4 {
            assert!(approx_equal(prediction_grad[[idx / 2, idx % 2]], 0.0, 1e-6));
            assert!(approx_equal(target_grad[[idx / 2, idx % 2]], 0.0, 1e-6));
        }
    }

    #[test]
    fn l1_loss_backward_respects_mean_reduction_factor() {
        let mut prediction = Tensor::from_vec(vec![2.0, -4.0, 6.0, -8.0], vec![2, 2]);
        let mut target = Tensor::from_vec(vec![1.0, -6.0, 3.0, -2.0], vec![2, 2]);

        prediction.set_requires_grad(true);
        target.set_requires_grad(true);

        let loss = prediction.l1_loss(target);
        loss.backward();

        let prediction_grad = prediction.grad();
        let target_grad = target.grad();
        let total_elems = 4.0;

        assert!(approx_equal(prediction_grad[[0, 0]], 1.0 / total_elems, 1e-6));
        assert!(approx_equal(prediction_grad[[0, 1]], -1.0 / total_elems, 1e-6));
        assert!(approx_equal(prediction_grad[[1, 0]], 1.0 / total_elems, 1e-6));
        assert!(approx_equal(prediction_grad[[1, 1]], -1.0 / total_elems, 1e-6));

        assert!(approx_equal(target_grad[[0, 0]], -1.0 / total_elems, 1e-6));
        assert!(approx_equal(target_grad[[0, 1]], 1.0 / total_elems, 1e-6));
        assert!(approx_equal(target_grad[[1, 0]], -1.0 / total_elems, 1e-6));
        assert!(approx_equal(target_grad[[1, 1]], 1.0 / total_elems, 1e-6));
    }
}
