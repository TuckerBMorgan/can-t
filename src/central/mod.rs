mod shape;
mod tensor;
mod equation;
mod opeartion;

use std::sync::{Mutex, MutexGuard};
pub use tensor::*;
pub use shape::*;
pub use equation::*;
pub use lazy_static::*;
pub use opeartion::*;

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
