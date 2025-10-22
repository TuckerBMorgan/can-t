mod add_op;
mod cat_op;
mod chunk_op;
mod clamp_op;
mod cos_op;
mod cross_entropy_op;
mod diagonal_op;
mod einsum;
mod equation;
mod index;
mod log;
mod masked_fill;
mod matmul_op;
mod mean_op;
mod movedim_op;
mod mul_op;
mod operation;
mod permute;
mod pow_op;
mod relu_op;
mod reshape;
mod select_op;
mod shape;
mod sin_op;
mod softmax_op;
mod std_op;
mod sum_op;
mod tanh_op;
mod tensor;
mod transpose_op;
mod unsqueeze_op;

pub use add_op::*;
pub use cat_op::*;
pub use chunk_op::*;
pub use clamp_op::*;
pub use cos_op::*;
pub use cross_entropy_op::*;
pub use diagonal_op::*;
pub use einsum::*;
pub use equation::*;
pub use index::*;
pub use lazy_static::*;
pub use log::*;
pub use masked_fill::*;
pub use matmul_op::*;
pub use mean_op::*;
pub use movedim_op::*;
pub use mul_op::*;
pub use operation::*;
pub use permute::*;
pub use pow_op::*;
pub use relu_op::*;
pub use reshape::*;
pub use select_op::*;
pub use shape::*;
pub use sin_op::*;
pub use softmax_op::*;
use std::sync::{Mutex, MutexGuard};
pub use std_op::*;
pub use sum_op::*;
pub use tanh_op::*;
pub use tensor::*;
pub use transpose_op::*;
pub use unsqueeze_op::*;

lazy_static! {
    static ref SINGLETON_INSTANCE: Mutex<Equation> = Mutex::new(Equation::new());
}

pub fn get_equation() -> MutexGuard<'static, Equation> {
    loop {
        let lock = SINGLETON_INSTANCE.lock();

        match lock {
            Ok(equation) => {
                return equation;
            }
            Err(_) => {
                continue;
            }
        }
    }
}

pub fn zero_all_grads() {
    loop {
        let lock = SINGLETON_INSTANCE.lock();

        match lock {
            Ok(mut equation) => {
                equation.zero_grad();
                return;
            }
            Err(_) => {
                continue;
            }
        }
    }
}

pub fn update_parameters(learning_rate: f32) {
    get_equation().update_parameters(learning_rate);
}
