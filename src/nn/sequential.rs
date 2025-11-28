use crate::central::*;
use crate::nn::*;

pub struct Sequential {
    pub layers: Vec<Box<dyn Layer>>,
}

impl Sequential {
    #[allow(unused)]
    pub fn new(layers: Vec<Box<dyn Layer>>) -> Self {
        Sequential { layers }
    }
}

impl Model for Sequential {
    fn forward(&mut self, input: Tensor) -> Tensor {
        let mut x = input.clone();
        for layer in &mut self.layers {
            x = layer.forward(x);
        }
        return x;
    }

    fn get_parameters(&self) -> Vec<TensorID> {
        let mut parameters = vec![];
        for layer in &self.layers {
            parameters.extend(layer.get_parameters());
        }

        return parameters;
    }
}

impl From<Vec<Box<dyn Layer>>> for Sequential {
    fn from(modules: Vec<Box<dyn Layer>>) -> Sequential {
        Sequential::new(modules)
    }
}

#[cfg(test)]
mod tests {
    use crate::utils::{DataLoader, MnistDataSet};

    use super::*;
    use ndarray::ArrayD;
    fn approx_equal(a: f32, b: f32, epsilon: f32) -> bool {
        (a - b).abs() < epsilon
    }

    #[test]
    fn test_sequential_creation_empty() {
        let model = Sequential::new(vec![]);

        // Empty model should have no parameters
        assert_eq!(model.get_parameters().len(), 0);
        assert_eq!(model.layers.len(), 0);
    }

    #[test]
    fn test_sequential_creation_single_layer() {
        let linear = Linear::new(3, 2, true);
        let mut model = Sequential::new(vec![Box::new(linear)]);

        // Should have 1 layer with 2 parameters (weights + bias)
        assert_eq!(model.layers.len(), 1);
        assert_eq!(model.get_parameters().len(), 2);

        // Test forward pass
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let output = model.forward(input);

        assert_eq!(output.shape.dimensions(), vec![1, 2]);
        let result = output.item();
        assert!(result[[0, 0]].is_finite());
        assert!(result[[0, 1]].is_finite());
    }

    #[test]
    fn test_sequential_linear_tanh_chain() {
        let linear = Linear::new(2, 3, true);
        let tanh = TanhLayer::new();
        let mut model = Sequential::new(vec![Box::new(linear), Box::new(tanh)]);

        // Should have 2 parameters from linear layer (weights + bias)
        assert_eq!(model.get_parameters().len(), 2);

        // Test forward pass
        let input = Tensor::from_vec(vec![1.0, -1.0], vec![1, 2]);
        let output = model.forward(input);

        assert_eq!(output.shape.dimensions(), vec![1, 3]);
        let result = output.item();

        // Output should be bounded by tanh [-1, 1]
        for i in 0..3 {
            assert!(result[[0, i]] >= -1.0);
            assert!(result[[0, i]] <= 1.0);
            assert!(result[[0, i]].is_finite());
        }
    }

    #[test]
    fn test_sequential_mlp_network() {
        // Create a multi-layer perceptron: 4 -> 8 -> 4 -> 1
        let layer1 = Linear::new(4, 8, true);
        let tanh1 = TanhLayer::new();
        let layer2 = Linear::new(8, 4, true);
        let tanh2 = TanhLayer::new();
        let output_layer = Linear::new(4, 1, false);

        let mut model = Sequential::new(vec![
            Box::new(layer1),
            Box::new(tanh1),
            Box::new(layer2),
            Box::new(tanh2),
            Box::new(output_layer),
        ]);

        // Should have 6 parameters: 3 layers with weights + 2 biases (output layer has no bias)
        assert_eq!(model.get_parameters().len(), 5);

        // Test forward pass with batch
        let batch_size = 3;
        let input = Tensor::ones(Shape::new(vec![batch_size, 4]));
        let output = model.forward(input);

        assert_eq!(output.shape.dimensions(), vec![batch_size, 1]);
        let result = output.item();

        // All outputs should be finite
        for i in 0..batch_size {
            assert!(result[[i, 0]].is_finite());
        }
    }

    #[test]
    fn test_sequential_from_vec() {
        let linear = Linear::new(2, 1, true);
        let tanh = TanhLayer::new();

        // Test From trait
        let mut model: Sequential = vec![
            Box::new(linear) as Box<dyn Layer>,
            Box::new(tanh) as Box<dyn Layer>,
        ]
        .into();

        assert_eq!(model.layers.len(), 2);

        let input = Tensor::from_vec(vec![0.5, -0.5], vec![1, 2]);
        let output = model.forward(input);

        assert_eq!(output.shape.dimensions(), vec![1, 1]);
    }

    #[test]
    fn test_sequential_batch_processing() {
        let linear = Linear::new(3, 2, true);
        let tanh = TanhLayer::new();
        let mut model = Sequential::new(vec![Box::new(linear), Box::new(tanh)]);

        // Test with different batch sizes
        let batch_sizes = vec![1, 5, 10];

        for batch_size in batch_sizes {
            let input = Tensor::ones(Shape::new(vec![batch_size, 3]));
            let output = model.forward(input);

            assert_eq!(output.shape.dimensions(), vec![batch_size, 2]);

            // Check all outputs are in tanh range
            let result = output.item();
            for i in 0..batch_size {
                for j in 0..2 {
                    assert!(result[[i, j]] >= -1.0);
                    assert!(result[[i, j]] <= 1.0);
                    assert!(result[[i, j]].is_finite());
                }
            }
        }
    }

    #[test]
    fn test_sequential_xor_training() {
        // Train a simple XOR network using Sequential
        let layer1 = Linear::new(2, 4, true);
        let tanh1 = TanhLayer::new();
        let layer2 = Linear::new(4, 1, true);

        let mut model = Sequential::new(vec![Box::new(layer1), Box::new(tanh1), Box::new(layer2)]);

        // XOR dataset
        let x = Tensor::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], vec![4, 2]);
        let y = Tensor::from_vec(vec![0.0, 1.0, 1.0, 0.0], vec![4, 1]);

        // Set all parameters to require gradients
        let params = model.get_parameters();
        for param_id in params {
            get_equation().set_is_grequires_grad(param_id, true);
        }

        let mut initial_loss = 0.0;
        let mut final_loss = 0.0;

        // Training loop
        for epoch in 0..50 {
            zero_all_grads();
            let output = model.forward(x.clone());
            let loss = (output - y.clone()).pow(2.0).mean(vec![0, 1]);

            if epoch == 0 {
                initial_loss = loss.item()[0];
            }
            if epoch == 49 {
                final_loss = loss.item()[0];
            }

            loss.backward();
            update_parameters(-0.1);
        }

        // Loss should decrease (basic learning check)
        assert!(
            final_loss < initial_loss,
            "Loss should decrease: initial={}, final={}",
            initial_loss,
            final_loss
        );

        // Test final predictions
        let final_output = model.forward(x);
        let result = final_output.item();

        // Results should be closer to target than random
        for i in 0..4 {
            assert!(result[[i, 0]].is_finite());
        }
    }

    #[test]
    fn test_sequential_deterministic_behavior() {
        // Test that same input produces same output (deterministic)
        let linear = Linear::new(2, 3, true);
        let tanh = TanhLayer::new();

        let mut model1 = Sequential::new(vec![Box::new(linear), Box::new(tanh)]);

        let input = Tensor::from_vec(vec![1.0, 2.0], vec![1, 2]);

        let output1 = model1.forward(input.clone());
        let output2 = model1.forward(input);

        let result1 = output1.item();
        let result2 = output2.item();

        // Same input should produce same output
        for i in 0..1 {
            for j in 0..3 {
                assert!(
                    approx_equal(result1[[i, j]], result2[[i, j]], 1e-10),
                    "Sequential should be deterministic"
                );
            }
        }
    }

    fn lerp_array(start: f32, end: f32, steps: usize) -> Vec<f32> {
        if steps == 0 {
            return vec![];
        }

        if steps == 1 {
            return vec![start];
        }

        let mut result = Vec::with_capacity(steps);

        for i in 0..steps {
            let t = i as f32 / (steps - 1) as f32;
            let value = start + t * (end - start);
            result.push(value);
        }

        result
    }

    #[test]
    fn test_sequential_mnist_training() {
        use mnist::{Mnist, MnistBuilder};

        let mnist_data = MnistDataSet::new();
        let mut data_loader = DataLoader::new(Box::new(mnist_data),32 );


        // Create MNIST classifier model: 784 -> 128 -> 64 -> 10
        let mut model = Sequential::new(vec![
            Box::new(Linear::new(784, 128, true)),
            Box::new(ReLU::new()),
            Box::new(Linear::new(128, 64, true)),
            Box::new(ReLU::new()),
            Box::new(Linear::new(64, 10, true)),
        ]);

        // Set all parameters to require gradients
        let params = model.get_parameters();
        for param_id in params {
            get_equation().set_is_grequires_grad(param_id, true);
        }



        let batch_size = 32;
        let epochs = 2;

        let mut initial_loss = 0.0;
        let mut final_loss = 0.0;
        let learning_rates = lerp_array(0.1, 0.01, epochs);


        for epoch in 0..epochs {
            let mut total_loss = 0.0;
            let num_samples = data_loader.number_of_samples();
            let num_batches = (num_samples + batch_size - 1) / batch_size; // Ceiling division

            for batch_idx in 0..num_batches {
                zero_all_grads();

                // Extract batch (handle last batch which might be smaller)
                let start_idx = batch_idx * batch_size;

                let (batch_images, batch_labels) = data_loader.get_random_batch(32);
                // Forward pass
                let outputs = model.forward(batch_images);
                // Compute loss (mean squared error)
                let loss = (outputs - batch_labels).pow(2.0).mean(vec![0, 1]);
                let loss_val = loss.item()[0];
                total_loss += loss_val;
                if epoch == 0 && batch_idx == 0 {
                    initial_loss = loss_val;
                }
                // Backward pass and parameter update
                loss.backward();
                update_parameters(-learning_rates[epoch]);
                clean_up_tensor_store();
            }

            let avg_loss = total_loss / num_batches as f32;
            if epoch == epochs - 1 {
                final_loss = avg_loss;
            }

            println!("Epoch {}: Average Loss = {:.4}", epoch, avg_loss);
        }

        // Verify that loss decreased during training
        assert!(
            final_loss < initial_loss,
            "Loss should decrease: initial={:.4}, final={:.4}",
            initial_loss,
            final_loss
        );
    }
}
