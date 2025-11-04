mod gguf;
mod timing;
pub use gguf::*;
pub use timing::*;

use crate::central::{MAX_DIMS, Tensor};

/// Handles broadcasting two operands to the right size, was doing this in a few places, put it into a util function
/// Arugments
/// *lhs* - the left hand opearand
/// *rhs* - the right hand operand
pub fn handle_broadcasting(lhs: Tensor, rhs: Tensor) -> (Tensor, Tensor) {
    // We want to make sure that the two operands can be muled on a elementwise way
    // so we try to broadcast them together if they do not equal each other
    let mut working_rhs = rhs;
    if working_rhs.shape != lhs.shape {
        if working_rhs.shape.should_broadcast(lhs.shape)
            && lhs.shape.can_broadcast(working_rhs.shape)
        {
            working_rhs = rhs.broadcast(lhs.shape);
        }
    }

    let mut working_lfs = lhs;
    if working_lfs.shape != working_rhs.shape {
        if working_lfs.shape.should_broadcast(working_rhs.shape)
            && working_lfs.shape.can_broadcast(working_rhs.shape)
        {
            working_lfs = working_lfs.broadcast(working_rhs.shape);
        }
    }
    assert!(working_lfs.shape == working_rhs.shape);

    return (working_lfs, working_rhs);
}

pub fn padding_dimenions_to_max(in_dimension: Vec<usize>) -> [usize; MAX_DIMS] {
    assert!(in_dimension.len() <= MAX_DIMS);
    assert!(in_dimension.len() != 0);

    let mut return_dimension = [1; MAX_DIMS];

    for (index, dinension) in in_dimension.iter().rev().enumerate() {
        return_dimension[MAX_DIMS - 1 - index] = *dinension;
    }

    return return_dimension;
}

#[cfg(test)]
mod tests {
    use super::padding_dimenions_to_max;

    #[test]
    pub fn padding_test() {
        let initial_array = vec![1];
        let after_change = padding_dimenions_to_max(initial_array);
        assert!(after_change[0] == 1);
        assert!(after_change[1] == 1);
        assert!(after_change[2] == 1);
        assert!(after_change[3] == 1);
    }

    #[test]
    pub fn padding_test_2() {
        let initial_array = vec![4, 1];
        let after_change = padding_dimenions_to_max(initial_array);
        assert!(after_change[0] == 1);
        assert!(after_change[1] == 1);
        assert!(after_change[2] == 4);
        assert!(after_change[3] == 1);
    }

    #[test]
    pub fn padding_test_3() {
        let initial_array = vec![3, 4, 1];
        let after_change = padding_dimenions_to_max(initial_array);
        assert!(after_change[0] == 1);
        assert!(after_change[1] == 3);
        assert!(after_change[2] == 4);
        assert!(after_change[3] == 1);
    }

    #[test]
    pub fn padding_test_4() {
        let initial_array = vec![3, 3, 4, 1];
        let after_change = padding_dimenions_to_max(initial_array);
        assert!(after_change[0] == 3);
        assert!(after_change[1] == 3);
        assert!(after_change[2] == 4);
        assert!(after_change[3] == 1);
    }
}
