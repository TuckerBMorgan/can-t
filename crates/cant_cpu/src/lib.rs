mod tensor_add;
mod tensor_mul;

pub mod prelude {
    pub use crate::tensor_add::*;
    pub use crate::tensor_mul::*;
}
