mod tensor_add;
mod tensor_matmul;
mod tensor_mul;
mod tensor_sub;
pub use lazy_static::*;

use metal::*;
use std::mem;

// This is also defined in the main src crate, maybe think about having some other crate
pub(crate) const MAX_DIMS: usize = 10;

lazy_static! {
    static ref METAL_DEVICE: Device = Device::system_default().expect("No Metal device found");
    static ref METAL_QUEUE: CommandQueue = METAL_DEVICE.new_command_queue();
    static ref METAL_LIBRARY: Library = METAL_DEVICE
        .new_library_with_source(
            include_str!("../shaders/all_shaders.metal"),
            &CompileOptions::new()
        )
        .expect("Failed to compile Metal shaders");
}

pub mod prelude {
    pub use crate::tensor_add::*;
    pub use crate::tensor_matmul::*;
    pub use crate::tensor_mul::*;
    pub use crate::tensor_sub::*;
}
