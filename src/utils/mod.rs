mod gguf;
mod timing;
pub use gguf::*;
pub use timing::*;

use crate::central::Tensor;

/// Handles broadcasting two operands to the right size, was doing this in a few places, put it into a util function
/// Arugments
/// *lhs* - the left hand opearand
/// *rhs* - the right hand operand
pub fn handle_broadcasting(lhs: Tensor, rhs: Tensor) -> (Tensor, Tensor) {
        // We want to make sure that the two operands can be muled on a elementwise way
        // so we try to broadcast them together if they do not equal each other
        let mut working_rhs = rhs;
        if working_rhs.shape != lhs.shape {
            if working_rhs.shape.should_broadcast(lhs.shape) && lhs.shape.can_broadcast(working_rhs.shape) {
                working_rhs = rhs.broadcast(lhs.shape);
            }
        }

        let mut working_lfs = lhs;
        if working_lfs.shape != working_rhs.shape {
            if working_lfs.shape.should_broadcast(working_rhs.shape) && working_lfs.shape.can_broadcast(working_rhs.shape) {
                working_lfs = working_lfs.broadcast(working_rhs.shape);
            }
        }
        assert!(working_lfs.shape == working_rhs.shape);

        return (working_lfs, working_rhs);
}