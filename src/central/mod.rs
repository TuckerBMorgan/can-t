mod add_op;
mod cat_op;
mod chunk_op;
mod clamp_op;
mod cos_op;
mod cross_entropy_op;
mod debugging;
mod diagonal_op;
mod einsum;
mod equation;
mod gather_op;
mod index;
mod l1_loss;
mod log;
mod masked_fill;
mod matmul_op;
mod max_op;
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
mod sigmoid_op;
mod sin_op;
mod softmax_op;
mod std_op;
mod sum_op;
mod tanh_op;
mod tensor;
mod topk_op;
mod transpose_op;
mod unsqueeze_op;
mod stack_op;

pub use add_op::*;
pub use cat_op::*;
pub use chunk_op::*;
pub use clamp_op::*;
pub use cos_op::*;
#[allow(unused_imports)]
pub use cross_entropy_op::*;
pub use debugging::*;
pub use diagonal_op::*;
#[allow(unused_imports)]
pub use einsum::*;
pub use equation::*;
pub use gather_op::*;
pub use index::*;
#[allow(unused_imports)]
pub use l1_loss::*;
pub use log::*;
pub use masked_fill::*;
pub use matmul_op::*;
pub use max_op::*;
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
pub use sigmoid_op::*;
pub use sin_op::*;
pub use softmax_op::*;
use std::sync::{LazyLock, Mutex, MutexGuard};
pub use std_op::*;
pub use sum_op::*;
pub use tanh_op::*;
pub use tensor::*;
pub use topk_op::*;
pub use transpose_op::*;
pub use unsqueeze_op::*;
pub use stack_op::*;

static SINGLETON_INSTANCE: LazyLock<Mutex<Equation>> =
    LazyLock::new(|| Mutex::new(Equation::new()));

pub fn get_equation() -> MutexGuard<'static, Equation> {
    // Handle poisoned mutex without a busy loop
    SINGLETON_INSTANCE.lock().unwrap_or_else(|e| e.into_inner())
}

pub fn zero_all_grads() {
    let mut eq = SINGLETON_INSTANCE.lock().unwrap_or_else(|e| e.into_inner());
    eq.zero_grad();
}

pub fn update_parameters(learning_rate: f32) {
    get_equation().update_parameters(learning_rate);
}

pub fn clean_up_tensor_store() {
    get_equation().garbage_collect();
}

pub fn clip_gradients(max_norm: f32) {
    get_equation().clip_grad_norm(max_norm);
}

pub fn validate_tensor_store() {
    get_equation().validate_tensor_store();
}
