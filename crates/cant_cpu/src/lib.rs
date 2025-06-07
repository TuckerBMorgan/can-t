mod tensor_add;
mod tensor_mul;
mod tensor_matmul;

pub mod prelude {
    pub use crate::tensor_add::*;
    pub use crate::tensor_mul::*;
    pub use crate::tensor_matmul::*;
}
