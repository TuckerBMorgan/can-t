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

        // Helper function to load and preprocess MNIST data
        fn load_mnist_data() -> (Tensor, Tensor, Tensor, Tensor) {
            let Mnist {
                trn_img,
                trn_lbl,
                tst_img,
                tst_lbl,
                ..
            } = MnistBuilder::new()
                .label_format_digit()
                //  .training_set_length(1000)  // Small subset for fast testing
                //   .test_set_length(200)
                .finalize();
            println!("Preprocess images, train");
            let train_images = preprocess_images(trn_img, 60000);
            println!("Preprocess labels, train");
            let train_labels = preprocess_labels(trn_lbl);
            println!("Preprocess images, test");
            let test_images = preprocess_images(tst_img, 10000);
            println!("Preprocess images, test");
            let test_labels = preprocess_labels(tst_lbl);

            (train_images, train_labels, test_images, test_labels)
        }

        // Convert u8 pixel values to normalized f32 tensors
        fn preprocess_images(raw_images: Vec<u8>, num_samples: usize) -> Tensor {
            let normalized: Vec<f32> = raw_images
                .iter()
                .map(|&pixel| pixel as f32 / 255.0)
                .collect();

            Tensor::from_vec(normalized, vec![num_samples, 784])
        }

        // Convert digit labels to one-hot encoded vectors
        fn preprocess_labels(raw_labels: Vec<u8>) -> Tensor {
            let num_samples = raw_labels.len();
            let mut one_hot = vec![0.0f32; num_samples * 10];

            for (i, &label) in raw_labels.iter().enumerate() {
                one_hot[i * 10 + label as usize] = 1.0;
            }

            Tensor::from_vec(one_hot, vec![num_samples, 10])
        }

        // Extract batch from tensor by copying rows
        fn extract_batch(
            tensor: &ArrayD<f32>,
            features: usize,
            start_idx: usize,
            end_idx: usize,
        ) -> Tensor {
            let batch_size = end_idx - start_idx;
            let features = features;
            let data = tensor;

            // Pre-allocate the exact size needed
            let mut batch_data = Vec::with_capacity(batch_size * features);

            // Use slice copying for better performance
            for i in start_idx..end_idx {
                let row_start = i * features;
                let row_end = row_start + features;

                // Copy entire row at once (more efficient than element-by-element)
                if let Some(flat_data) = data.as_slice() {
                    batch_data.extend_from_slice(&flat_data[row_start..row_end]);
                } else {
                    // Fallback to element access
                    for j in 0..features {
                        batch_data.push(data[[i, j]]);
                    }
                }
            }

            Tensor::from_vec(batch_data, vec![batch_size, features])
        }

        // Calculate classification accuracy
        fn calculate_accuracy(predictions: &Tensor, targets: &Tensor) -> f32 {
            let pred_data = predictions.item();
            let target_data = targets.item();
            let num_samples = predictions.shape.dimensions()[0];

            let mut correct = 0;
            for i in 0..num_samples {
                // Find predicted class (argmax)
                let mut pred_class = 0;
                let mut max_pred = pred_data[[i, 0]];
                for j in 1..10 {
                    if pred_data[[i, j]] > max_pred {
                        max_pred = pred_data[[i, j]];
                        pred_class = j;
                    }
                }

                // Find true class (argmax owf one-hot)
                let mut true_class = 0;
                for j in 0..10 {
                    if target_data[[i, j]] == 1.0 {
                        true_class = j;
                        break;
                    }
                }

                if pred_class == true_class {
                    correct += 1;
                }
            }

            correct as f32 / num_samples as f32
        }

        // Load MNIST data
        let (train_images, train_labels, test_images, test_labels) = load_mnist_data();
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
        // Training loop
        let train_images_array_d = train_images.item();
        let train_features = train_images.shape.dimensions()[1];

        let test_images_array_d = train_labels.item();
        let test_features = train_labels.shape.dimensions()[1];

        for epoch in 0..epochs {
            let mut total_loss = 0.0;
            let num_samples = train_images.shape.dimensions()[0];
            let num_batches = (num_samples + batch_size - 1) / batch_size; // Ceiling division

            for batch_idx in 0..num_batches {
                zero_all_grads();

                // Extract batch (handle last batch which might be smaller)
                let start_idx = batch_idx * batch_size;
                let end_idx = std::cmp::min(start_idx + batch_size, num_samples);

                let batch_images =
                    extract_batch(&train_images_array_d, train_features, start_idx, end_idx);
                let batch_labels =
                    extract_batch(&test_images_array_d, test_features, start_idx, end_idx);

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

        // Test model accuracy on test set
        let test_outputs = model.forward(test_images.clone());
        let accuracy = calculate_accuracy(&test_outputs, &test_labels);

        println!("Test Accuracy: {:.2}%", accuracy * 100.0);

        // Model should achieve better than random performance (10% for 10 classes)
        assert!(
            accuracy > 0.15,
            "Model should achieve > 15% accuracy, got {:.2}%",
            accuracy * 100.0
        );

        // Basic sanity checks
        assert_eq!(test_outputs.shape.dimensions(), vec![10000, 10]);

        // Verify outputs are finite
        let output_data = test_outputs.item();
        for i in 0..200 {
            for j in 0..10 {
                assert!(
                    output_data[[i, j]].is_finite(),
                    "Output should be finite at [{}, {}]",
                    i,
                    j
                );
            }
        }
    }


    #[test]
    pub fn celeb_a_dcgan_test() {
        let batch_size = 128;
        let image_size = 64;
        let number_of_channels = 3;
        let size_of_latent_z_vector = 100;
        let number_of_feature_maps_generator = 64;
        let number_of_feature_maps_discriminator = 64;
        let number_of_training_epochs = 5;
        let learning_rate = 0.0020;
        let beta_1 = 0.5;


    }
}
