use crate::central::*;
use crate::nn::*;

const EPSILON: f32 = 1e-5;

pub struct RMSNorm {
    scale: Tensor,
}

impl RMSNorm {
    pub fn new(number_of_features: usize) -> RMSNorm {
        let mut scale = Tensor::ones(Shape::new(vec![number_of_features]));
        scale.set_requires_grad(true);
        scale.set_keep_alive(true);
        RMSNorm { scale }
    }
}

impl Layer for RMSNorm {
    fn forward(&mut self, inputs: Tensor) -> Tensor {
        let dimensions = inputs.shape.dimensions();
        assert!(
            !dimensions.is_empty(),
            "RMSNorm expects inputs with at least one dimension"
        );
        let last_dim = dimensions.len() - 1;

        let squared = inputs.pow(2.0);
        let mean = squared.mean(vec![last_dim]);

        let mut broadcast_shape = dimensions.clone();
        broadcast_shape[last_dim] = 1;
        let mean = mean.reshape(Shape::new(broadcast_shape.clone()));

        let epsilon_tensor = Tensor::element(mean.shape, EPSILON);
        let rms = (mean + epsilon_tensor).pow(0.5);
        let normalized = inputs / rms;

        normalized * self.scale
    }

    fn get_parameters(&self) -> Vec<TensorID> {
        vec![self.scale.id]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_equal(a: f32, b: f32, epsilon: f32) -> bool {
        (a - b).abs() < epsilon
    }

    fn reference_rmsnorm(sample: &[f32], scale: &[f32]) -> Vec<f32> {
        let mean_square: f32 = sample.iter().map(|x| x * x).sum::<f32>() / sample.len() as f32;
        let denom = (mean_square + EPSILON).sqrt();
        sample
            .iter()
            .zip(scale.iter())
            .map(|(value, weight)| value / denom * weight)
            .collect()
    }

    #[test]
    fn test_rmsnorm_parameters_registered() {
        let layer = RMSNorm::new(4);
        let params = layer.get_parameters();

        assert_eq!(params.len(), 1);
        assert_eq!(params[0], layer.scale.id);

        let scale_data = layer.scale.item();
        for i in 0..4 {
            assert!(approx_equal(scale_data[[i]], 1.0, 1e-6));
        }
    }

    #[test]
    fn test_rmsnorm_forward_single_vector() {
        let mut layer = RMSNorm::new(4);
        let input_values = vec![1.0, 2.0, 3.0, 4.0];
        let input = Tensor::from_vec(input_values.clone(), vec![1, 4]);

        let output = layer.forward(input);
        let result = output.item();
        let expected = reference_rmsnorm(&input_values, &[1.0; 4]);

        for i in 0..4 {
            assert!(approx_equal(result[[0, i]], expected[i], 1e-5));
        }
    }

    #[test]
    fn test_rmsnorm_forward_batch() {
        let mut layer = RMSNorm::new(3);
        let input_values = vec![
            1.0, 2.0, 3.0, // sample 0
            4.0, 5.0, 6.0, // sample 1
        ];
        let input = Tensor::from_vec(input_values.clone(), vec![2, 3]);

        let output = layer.forward(input);
        let result = output.item();

        for sample_idx in 0..2 {
            let start = sample_idx * 3;
            let slice = &input_values[start..start + 3];
            let expected = reference_rmsnorm(slice, &[1.0; 3]);
            for feature_idx in 0..3 {
                assert!(approx_equal(
                    result[[sample_idx, feature_idx]],
                    expected[feature_idx],
                    1e-5
                ));
            }
        }
    }

    #[test]
    fn test_rmsnorm_respects_scale_parameter() {
        let mut layer = RMSNorm::new(3);
        let mut custom_scale = Tensor::from_vec(vec![0.5, 1.5, 2.0], vec![3]);
        custom_scale.set_requires_grad(true);
        custom_scale.set_keep_alive(true);
        layer.scale = custom_scale;

        let input_values = vec![2.0, -1.0, 4.0];
        let input = Tensor::from_vec(input_values.clone(), vec![1, 3]);
        let output = layer.forward(input);
        let result = output.item();

        let expected = reference_rmsnorm(&input_values, &[0.5, 1.5, 2.0]);
        for i in 0..3 {
            assert!(approx_equal(result[[0, i]], expected[i], 1e-5));
        }
    }
}
