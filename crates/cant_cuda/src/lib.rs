mod tensor_add;
mod tensor_matmul;
mod tensor_mul;
mod tensor_sub;

pub use lazy_static::*;

use cudarc::driver::{LaunchAsync, LaunchConfig};
use cudarc::nvrtc::{Ptx, compile_ptx};

lazy_static! {
    static ref DEV : std::sync::Arc<cudarc::driver::CudaDevice> = cudarc::driver::CudaDevice::new(0).unwrap();
}

pub mod prelude {
    pub use crate::tensor_add::*;
    pub use crate::tensor_matmul::*;
    pub use crate::tensor_mul::*;
    pub use crate::tensor_sub::*;
}
