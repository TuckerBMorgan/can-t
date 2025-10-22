use crate::*;
use cudarc::driver::{LaunchAsync, LaunchConfig};
use cudarc::nvrtc::{Ptx, compile_ptx};
pub fn tensor_mul(a: &[f32], b: &[f32]) -> Vec<f32> {
    let PTX: Ptx = compile_ptx(include_str!("../shaders/all_kernels.cu")).unwrap();
    DEV.load_ptx(PTX, "mul_arrays", &["mul_arrays"]).unwrap();
    let func = DEV.get_func("mul_arrays", "mul_arrays").unwrap();
    let tile_size = 16;

    let m = a.len();
    let n = a.len();
    let k = a.len();

    let grid_rows = (m + tile_size - 1) / tile_size;
    let grid_cols = (k + tile_size - 1) / tile_size;

    let grid_dims = (grid_cols as u32, grid_rows as u32, 1);
    let cfg = LaunchConfig {
        block_dim: (tile_size as u32, tile_size as u32, 1),
        grid_dim: grid_dims,
        shared_mem_bytes: 0,
    };
    let a_on_device = DEV.htod_sync_copy(&a).unwrap();
    let b_on_device = DEV.htod_sync_copy(&b).unwrap();

    let c_host = vec![0.0f32; a.len()];
    let mut out_on_device = DEV.htod_sync_copy(&c_host).unwrap();

    unsafe {
        let result = func.launch(
            cfg,
            (&a_on_device, &b_on_device, &mut out_on_device, m, n, k),
        );
        match result {
            Ok(_) => {}
            Err(e) => {
                println!("Error: {:?}", e);
            }
        }
    }
    let c_host = DEV.dtoh_sync_copy(&out_on_device).unwrap();
    return c_host;
}
