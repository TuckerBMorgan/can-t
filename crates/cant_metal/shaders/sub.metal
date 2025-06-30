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