use super::{BackproagationPacket, Operation};

/// Handles calculating and passing back the gradient of a select operation
/// For select operations, gradients are accumulated back to the source tensor
/// at the positions specified by the indices tensor
pub fn backward_for_select(backprop_packet: BackproagationPacket) {
    if let Operation::Select(source_id, indices_id) = backprop_packet.operation {
        panic!("backwards for select needs work");
    } else {
        panic!("Wrong operation for backward select");
    }
}

#[cfg(test)]
mod tests {
    use crate::central::{Shape, Tensor};
    use crate::central::index::Indexable;
    
    fn approx_equal(a: f32, b: f32, epsilon: f32) -> bool {
        (a - b).abs() <= epsilon
    }
    
    #[test]
    fn basic_select_test() {
        // Create a simple embedding matrix [3, 2]
        let embedding = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![3, 2]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            crate::central::Operation::Nop,
        );
        
        // Create indices tensor [2] with indices [0, 2]
        let indices = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![2]),
            vec![0.0, 2.0],
            crate::central::Operation::Nop,
        );
        
        // Select using view method
        let result = embedding.view(Indexable::FromTensor(indices.id));
        
        // Check result shape should be [2, 2]
        assert_eq!(result.shape.dimensions(), vec![2, 2]);
        
        // Check result values
        let result_data = result.item();
        // Expected: row 0 [1.0, 2.0] and row 2 [5.0, 6.0] from embedding
        
        // Check first selected row (index 0): [1.0, 2.0]
        assert!(approx_equal(result_data[[0, 0]], 1.0, 1e-6));
        assert!(approx_equal(result_data[[0, 1]], 2.0, 1e-6));
        
        // Check second selected row (index 2): [5.0, 6.0]
        assert!(approx_equal(result_data[[1, 0]], 5.0, 1e-6));
        assert!(approx_equal(result_data[[1, 1]], 6.0, 1e-6));
    }
    
    #[test]
    fn select_shape_test() {
        // Test with different shapes
        let embedding = Tensor::zeros(Shape::new(vec![10, 5])); // vocab_size=10, embedding_dim=5
        let indices = Tensor::zeros(Shape::new(vec![3, 4])); // batch_size=3, seq_len=4
        
        let result = embedding.select(indices.id);
        
        // Result should be [3, 4, 5]
        assert_eq!(result.shape.dimensions(), vec![3, 4, 5]);
    }
    
    #[test]
    fn select_direct_test() {
        // Test the select method directly
        let embedding = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![2, 3]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            crate::central::Operation::Nop,
        );
        
        let indices = Tensor::create_tensor_data_and_shape_and_operation(
            Shape::new(vec![1]),
            vec![1.0],
            crate::central::Operation::Nop,
        );
        
        let result = embedding.select(indices.id);
        
        // Should get row 1: [4.0, 5.0, 6.0], result shape should be [1, 3]
        let result_data = result.item();
        assert_eq!(result.shape.dimensions(), vec![1, 3]);
        assert!(approx_equal(result_data[[0, 0]], 4.0, 1e-6));
        assert!(approx_equal(result_data[[0, 1]], 5.0, 1e-6));
        assert!(approx_equal(result_data[[0, 2]], 6.0, 1e-6));
    }
}