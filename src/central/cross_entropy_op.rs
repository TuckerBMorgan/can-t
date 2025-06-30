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
