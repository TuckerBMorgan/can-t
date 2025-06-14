mod add_op;
mod equation;
mod matmul_op;
mod mul_op;
mod operation;
mod pow_op;
mod reshape;
mod shape;
mod sum_op;
mod tensor;

pub use add_op::*;
pub use equation::*;
pub use lazy_static::*;
pub use matmul_op::*;
pub use mul_op::*;
pub use operation::*;
pub use pow_op::*;
pub use reshape::*;
pub use shape::*;
use std::sync::{Mutex, MutexGuard};
pub use sum_op::*;
pub use tensor::*;

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
