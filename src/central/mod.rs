mod add_op;
mod cross_entropy_op;
mod equation;
mod matmul_op;
mod mean_op;
mod mul_op;
mod operation;
mod pow_op;
mod reshape;
mod select_op;
mod shape;
mod std_op;
mod tanh_op;
mod sum_op;
mod tensor;
mod index;
mod view;

pub use add_op::*;
pub use cross_entropy_op::*;
pub use equation::*;
pub use lazy_static::*;
pub use matmul_op::*;
pub use mean_op::*;
pub use mul_op::*;
pub use operation::*;
pub use pow_op::*;
pub use reshape::*;
pub use select_op::*;
pub use shape::*;
pub use std_op::*;
pub use tanh_op::*;
use std::sync::{Mutex, MutexGuard};
pub use sum_op::*;
pub use tensor::*;
pub use index::*;
pub use view::*;

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
