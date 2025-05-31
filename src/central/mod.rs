mod shape;
mod tensor;
mod equation;
mod operation;
mod add_op;
mod mul_op;
mod sum;

use std::sync::{Mutex, MutexGuard};
pub use tensor::*;
pub use shape::*;
pub use equation::*;
pub use lazy_static::*;
pub use operation::*;
pub use add_op::*;
pub use mul_op::*;
pub use sum::*;

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
