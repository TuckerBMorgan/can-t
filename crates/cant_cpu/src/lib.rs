mod tensor_add;
mod tensor_matmul;
mod tensor_mul;
mod tensor_sub;

// This is also defined in the main src crate, maybe think about having some other crate
pub(crate) const MAX_DIMS: usize = 10;

pub mod prelude {
    pub use crate::tensor_add::*;
    pub use crate::tensor_matmul::*;
    pub use crate::tensor_mul::*;
    pub use crate::tensor_sub::*;
}
