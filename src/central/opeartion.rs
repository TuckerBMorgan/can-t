use super::{tensor::TensorID, Shape};
use std::fmt;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Operation {
    /// No operation, this will not pass any gradient
    Nop,
    Add(TensorID, TensorID),
    BroadCast(TensorID, Shape)
}
