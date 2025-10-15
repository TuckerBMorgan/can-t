use crate::*;
use cudarc::{
    driver::{LaunchConfig},
    nvrtc::compile_ptx,
};


// Checks if the tensor shapes are compatible for batched matrix multiplication
fn valid_shape(a: [usize; 4], b: [usize; 4]) {
    // Ensure outer batch dimensions match
    assert!(a[0] == b[0], "{:?} {:?}", a, b);
    // Ensure inner batch dimensions match
    assert!(a[1] == b[1], "{:?} {:?}", a, b);
    // Ensure the inner dimensions of A and B are compatible for matrix multiplication
    assert!(a[3] == b[2], "{:?} {:?}", a, b);
}

pub fn tensor_matmul(a: &[f32], a_shape: [usize; 4], b: &[f32], b_shape: [usize; 4]) -> Vec<f32> {
    // Shapes: A = [B1, B2, M, K], B = [B1, B2, K, N]
    let (b1_a, b2_a, m, k) = (a_shape[0], a_shape[1], a_shape[2], a_shape[3]);
    let (b1_b, b2_b, k_b, n) = (b_shape[0], b_shape[1], b_shape[2], b_shape[3]);

    assert_eq!(b1_a, b1_b, "batch dim 0 mismatch");
    assert_eq!(b2_a, b2_b, "batch dim 1 mismatch");
    assert_eq!(k, k_b, "inner dim mismatch: K");
    assert_eq!(a.len(), b1_a * b2_a * m * k, "A buffer size != product of A shape");
    assert_eq!(b.len(), b1_b * b2_b * k * n, "B buffer size != product of B shape");

    let total_batches = b1_a * b2_a;
    let out_len = total_batches * m * n;

    // --- CUDA kernel (same indexing as your Metal shader) ---
    // C layout: [batch, M, N], A: [batch, M, K], B: [batch, K, N]
    const KERNEL_SRC: &str = r#"
        extern "C" __global__ void batchedMatMul(const float* __restrict__ A,
                        const float* __restrict__ B,
                        float* __restrict__ C,
                        unsigned M, unsigned N, unsigned K, unsigned totalBatches)
        {
            unsigned j = blockIdx.x * blockDim.x + threadIdx.x; // column (N)
            unsigned i = blockIdx.y * blockDim.y + threadIdx.y; // row (M)
            unsigned batchIndex = blockIdx.z;                   // batch (B1*B2)

            if (i >= M || j >= N || batchIndex >= totalBatches) return;

            unsigned aRowOffset = ((batchIndex * M) + i) * K;  // A[b,i,*]
            float sum = 0.0f;
            for (unsigned kk = 0; kk < K; ++kk) {
                float a_val = A[aRowOffset + kk];
                float b_val = B[(((batchIndex * K) + kk) * N) + j];
                sum += a_val * b_val;
            }
            C[(((batchIndex * M) + i) * N) + j] = sum;
        }
    "#;

    // 1) Create a CUDA context & stream

    // 2) Compile CUDA -> PTX at runtime, load the module, get the function
    let ptx = compile_ptx(KERNEL_SRC).expect("nvrtc compile failed");      // nvrtc compile  :contentReference[oaicite:2]{index=2}
    DEV.load_ptx(ptx, "batchedMatMul", &["batchedMatMul"]).unwrap();
    let func = DEV.get_func("batchedMatMul", "batchedMatMul").unwrap();
    // 3) Copy inputs to device / allocate output
    let d_a = DEV.htod_sync_copy(a).expect("copy A to device");            // slice->device  :contentReference[oaicite:5]{index=5}
    let d_b = DEV.htod_sync_copy(b).expect("copy B to device");            // slice->device  :contentReference[oaicite:6]{index=6}

    let c_host = vec![0.0f32; out_len];
    let mut out_on_device = DEV.htod_sync_copy(&c_host).unwrap();

    // 4) Launch configuration: 16x16 threads per block, 3D grid over (N, M, batches)
    let (tx, ty) = (16u32, 16u32);
    let grid_x = ((n as u32) + tx - 1) / tx;
    let grid_y = ((m as u32) + ty - 1) / ty;
    let grid_z = total_batches as u32;

    let cfg = LaunchConfig {
        grid_dim: (grid_x, grid_y, grid_z),
        block_dim: (tx, ty, 1),
        shared_mem_bytes: 0,
    }; // explicit 3D launch config pattern used in real codebases  :contentReference[oaicite:8]{index=8}

    // 5) Build args and launch
    unsafe {
        let result = func.launch(
            cfg,
            (&d_a, &d_b, &mut out_on_device, m, n, k, total_batches),
        );
        match result {
            Ok(_) => {},
            Err(e) => {
                println!("Error {:?}", e);
            }
        }
    }

    // 6) Copy back to host
    let out: Vec<f32> = DEV.dtoh_sync_copy(&out_on_device).expect("copy C to host"); // device->vec    :contentReference[oaicite:10]{index=10}
    out
}