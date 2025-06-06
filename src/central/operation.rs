use super::{Shape, tensor::TensorID};
use std::fmt;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Operation {
    /// No operation, this will not pass any gradient
    Nop,
    Add(TensorID, TensorID), // Left side operand tensor, right side operand tensor
    Mul(TensorID, TensorID), // Left side operand tensor, right side operand tensor
    BroadCast(TensorID, Shape), // Tensor we are broadcasting, the shape we are broadcasting from?
    Sum(TensorID, [usize; 4], usize, bool), // The Tensor that was summed, the dimensions that where summed, the number of summed demensions, keep_dimensions
                                      // Rust does not let you copy/clone vecs, and we will not be supporting matrices greater then 4, so we can use [usize;4] as a stand in for
                                      // the vec
    Pow(TensorID, TensorID)
}
