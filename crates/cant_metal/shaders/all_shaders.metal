struct MatMulDims {
    uint M, N, K, B2;
};

kernel void batchedMatMul(const device float *A [[buffer(0)]],
                          const device float *B [[buffer(1)]],
                          device float *C [[buffer(2)]],
                          constant MatMulDims& dims [[buffer(3)]],
                          uint3 gid [[thread_position_in_grid]])
{
    uint M = dims.M, N = dims.N, K = dims.K;
    uint batchIndex = gid.z;
    uint i = gid.y;
    uint j = gid.x;
    if (i >= M || j >= N) return;  // guard against any extra threads
    uint b1 = batchIndex / dims.B2;
    uint b2 = batchIndex % dims.B2;
    
    // Compute one element C[b1,b2,i,j]
    float sum = 0.0;
    // Linear offsets for the start of A[b1,b2,i,*] and B[b1,b2,*,j]
    uint aRowOffset = (((batchIndex * M) + i) * dims.K);
    uint bColOffset = (((batchIndex * dims.K) /*+ k*/) * N + j);
    // Loop over k dimension to accumulate dot product
    for (uint k = 0; k < K; ++k) {
        float a_val = A[aRowOffset + k];
        float b_val = B[(((batchIndex * dims.K) + k) * N + j)];
        sum += a_val * b_val;
        // (We could also increment bColOffset by N each loop and use B[bColOffset],
        //  to avoid recomputing indices inside the loop.)
    }
    // Write result
    C[(((batchIndex * M) + i) * N + j)] = sum;
}

kernel void mul_arrays(device const float* inA,
                       device const float* inB,
                       device float* result,
                       uint index [[thread_position_in_grid]])
{
    // the for-loop is replaced with a collection of threads, each of which
    // calls this function.
    result[index] = inA[index] * inB[index];
}

kernel void add_arrays(device const float* inA,
                       device const float* inB,
                       device float* result,
                       uint index [[thread_position_in_grid]])
{
    // the for-loop is replaced with a collection of threads, each of which
    // calls this function.
    result[index] = inA[index] + inB[index];
}

kernel void sub_arrays(device const float* inA,
                       device const float* inB,
                       device float* result,
                        constant uint& array_length [[buffer(3)]],
                       uint index [[thread_position_in_grid]])
{
    // the for-loop is replaced with a collection of threads, each of which
    // calls this function.
    if (index >= array_length) return;  // Bounds check
    result[index] = inA[index] - inB[index];
}