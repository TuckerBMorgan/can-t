use crate::central::*;
use ndarray::prelude::*;

impl Tensor {
    /// Computes cross-entropy loss between logits and targets
    /// # Arguments
    /// * 'targets' - One-hot encoded target tensor with same shape as logits
    pub fn cross_entropy_loss(&self, targets: Tensor) -> Tensor {
        panic!("Implement cross entropy");
    }
}

/// Handles calculating and passing back the gradient of a cross-entropy loss operation
pub fn backward_for_cross_entropy(backprop_packet: BackproagationPacket) {
    if let Operation::CrossEntropy(logits_id, targets_id) = backprop_packet.operation {
        panic!("backward for crossentopy needs backwards");
        
    } else {
        panic!("Wrong operation for backward cross entropy");
    }
}

#[cfg(test)]
mod tests {
    use crate::central::{Shape, Tensor, Operation};

    fn approx_equal(a: f32, b: f32, epsilon: f32) -> bool {
        (a - b).abs() <= epsilon
    }

    #[test]
    fn cross_entropy_simple_test() {
        // Test simple 2-class case
        let logits = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![2, 2]),
            vec![1.0, 0.0, 0.0, 1.0], // Logits for 2 batches, 2 classes
            Operation::Nop,
        );

        let targets = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![2, 2]),
            vec![1.0, 0.0, 0.0, 1.0], // One-hot targets
            Operation::Nop,
        );

        let loss = logits.cross_entropy_loss(targets);
        
        // Should be scalar
        assert_eq!(loss.shape.dimensions(), vec![1]);
        
        let loss_val = loss.item()[[0]];
        // For perfect predictions, loss should be low
        assert!(loss_val < 1.0, "Loss should be small for correct predictions, got {}", loss_val);
    }

    #[test]
    fn cross_entropy_batch_test() {
        // Test with batch size 3, 3 classes
        let logits = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![3, 3]),
            vec![
                2.0, 1.0, 0.1, // Batch 0: class 0 likely
                0.1, 2.0, 1.0, // Batch 1: class 1 likely  
                1.0, 0.1, 2.0, // Batch 2: class 2 likely
            ],
            Operation::Nop,
        );

        let targets = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![3, 3]),
            vec![
                1.0, 0.0, 0.0, // Batch 0: true class 0
                0.0, 1.0, 0.0, // Batch 1: true class 1
                0.0, 0.0, 1.0, // Batch 2: true class 2
            ],
            Operation::Nop,
        );

        let loss = logits.cross_entropy_loss(targets);
        
        assert_eq!(loss.shape.dimensions(), vec![1]);
        let loss_val = loss.item()[[0]];
        
        // Loss should be reasonable for good predictions
        assert!(loss_val > 0.0 && loss_val < 1.0, "Loss should be reasonable, got {}", loss_val);
    }

    #[test]
    fn cross_entropy_wrong_predictions_test() {
        // Test with completely wrong predictions
        let logits = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![2, 2]),
            vec![0.0, 2.0, 2.0, 0.0], // Predicting opposite classes
            Operation::Nop,
        );

        let targets = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![2, 2]),
            vec![1.0, 0.0, 0.0, 1.0], // True classes
            Operation::Nop,
        );

        let loss = logits.cross_entropy_loss(targets);
        
        let loss_val = loss.item()[[0]];
        // Loss should be high for wrong predictions
        assert!(loss_val > 1.0, "Loss should be high for wrong predictions, got {}", loss_val);
    }

    #[test]
    #[should_panic(expected = "Logits and targets must have the same shape")]
    fn cross_entropy_shape_mismatch_test() {
        let logits = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![2, 3]),
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            Operation::Nop,
        );

        let targets = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![2, 2]), // Wrong shape
            vec![1.0, 0.0, 0.0, 1.0],
            Operation::Nop,
        );

        logits.cross_entropy_loss(targets); // Should panic
    }

    #[test]
    #[should_panic(expected = "Cross entropy loss expects 2D tensors")]
    fn cross_entropy_wrong_dimensions_test() {
        let logits = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![4]), // 1D tensor
            vec![1.0, 0.0, 0.0, 0.0],
            Operation::Nop,
        );

        let targets = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![4]), // 1D tensor
            vec![1.0, 0.0, 0.0, 0.0],
            Operation::Nop,
        );

        logits.cross_entropy_loss(targets); // Should panic
    }
}