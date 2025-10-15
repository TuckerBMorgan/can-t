extern "C" __global__
void add_arrays(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] + b[i];
    }
}

void mul_arrays(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] * b[i];
    }
}

void sub_arrays(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] - b[i];
    }
}

// Mirror your Metal dims
struct MatMulDims {
    unsigned int M, N, K, B2;
};

extern "C" __global__
void batchedMatMul(const float* __restrict__ A,
                   const float* __restrict__ B,
                   float* __restrict__ C,
                   MatMulDims dims)
{
    const unsigned int j = blockIdx.x * blockDim.x + threadIdx.x; // N-dim (cols)
    const unsigned int i = blockIdx.y * blockDim.y + threadIdx.y; // M-dim (rows)
    const unsigned int batchIndex = blockIdx.z * blockDim.z + threadIdx.z;

    const unsigned int M = dims.M, N = dims.N, K = dims.K;

    if (i >= M || j >= N) return;

    // Compute one element C[batchIndex, i, j] with row-major packing per your Metal code
    float sum = 0.0f;

    // A offset: [batchIndex, i, *]
    const unsigned int aRowOffset = (batchIndex * M + i) * K;

    // Accumulate dot product over K
    for (unsigned int k = 0; k < K; ++k) {
        const float a_val = A[aRowOffset + k];                         // A[b,i,k]
        const float b_val = B[((batchIndex * K + k) * N) + j];         // B[b,k,j]
        sum += a_val * b_val;
    }

    // Write C[batchIndex, i, j]
    C[((batchIndex * M + i) * N) + j] = sum;
}