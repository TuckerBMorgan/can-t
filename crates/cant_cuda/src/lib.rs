pub use lazy_static::*;

use cudarc::driver::{LaunchAsync, LaunchConfig};
use cudarc::nvrtc::{compile_ptx, Ptx};
lazy_static! {
    static ref CONTEXT: CudaContext   = cudarc::driver::CudaContext::new(0)?;
    static ref stream = ctx.default_stream();
}
