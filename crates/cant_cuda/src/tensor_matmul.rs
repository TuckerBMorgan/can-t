use crate::*;
use cudarc::{driver::LaunchConfig, nvrtc::compile_ptx};

// Checks if the tensor shapes are compatible for batched matrix multiplication
fn valid_shape(a: [usize; 4], b: [usize; 4]) {
    // Ensure outer batch dimensions match
    assert!(a[0] == b[0], "{:?} {:?}", a, b);
    // Ensure inner batch dimensions match
    assert!(a[1] == b[1], "{:?} {:?}", a, b);
    // Ensure the inner dimensions of A and B are compatible for matrix multiplication
    assert!(a[3] == b[2], "{:?} {:?}", a, b);
}

use crate::MAX_DIMS;

fn valid_shape(a: [usize; MAX_DIMS], b: [usize; MAX_DIMS]) {
    // batch dims must match exactly (no broadcasting here)
    for i in 0..(MAX_DIMS - 2) {
        assert!(a[i] == b[i], "Batch dims mismatch for matmul: {:?} vs {:?}", a, b);
    }
    // inner matmul dims must align: (.., M, K) x (.., K, N)
    assert!(
        a[MAX_DIMS - 1] == b[MAX_DIMS - 2],
        "Inner dims mismatch for matmul: {:?} vs {:?}", a, b
    );
}

fn product(slice: &[usize]) -> usize {
    slice.iter().copied().fold(1usize, |acc, x| acc.saturating_mul(x))
}

pub fn loop_count(shape: [usize; MAX_DIMS]) -> usize {
    // product of batch dims only
    product(&shape[..(MAX_DIMS - 2)])
}

pub fn tensor_matmul(a: &[f32], a_shape: [usize; MAX_DIMS], b: &[f32], b_shape: [usize; MAX_DIMS]) -> Vec<f32> {
    // Shapes: A = [B1, B2, M, K], B = [B1, B2, K, N]


    let m = a_shape[MAX_DIMS - 2];
    let k = a_shape[MAX_DIMS - 1];
    let k_b = b_shape[MAX_DIMS - 2];
    let n = b_shape[MAX_DIMS - 1];


    let total_batches = loop_count(a_shape);
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
    let ptx = compile_ptx(KERNEL_SRC).expect("nvrtc compile failed"); // nvrtc compile  :contentReference[oaicite:2]{index=2}
    DEV.load_ptx(ptx, "batchedMatMul", &["batchedMatMul"])
        .unwrap();
    let func = DEV.get_func("batchedMatMul", "batchedMatMul").unwrap();
    // 3) Copy inputs to device / allocate output
    let d_a = DEV.htod_sync_copy(a).expect("copy A to device"); // slice->device  :contentReference[oaicite:5]{index=5}
    let d_b = DEV.htod_sync_copy(b).expect("copy B to device"); // slice->device  :contentReference[oaicite:6]{index=6}

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
            Ok(_) => {}
            Err(e) => {
                println!("Error {:?}", e);
            }
        }
    }

    // 6) Copy back to host
    let out: Vec<f32> = DEV.dtoh_sync_copy(&out_on_device).expect("copy C to host"); // device->vec    :contentReference[oaicite:10]{index=10}
    out
}
