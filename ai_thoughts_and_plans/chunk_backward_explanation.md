# Chunk Backward Pass

## Recap of the Forward Step
`Tensor::chunk(dim, chunks)` partitions the input tensor along axis `dim`. We choose slice sizes so they cover the entire axis (larger slices get distributed to the earliest chunks when the division is uneven). Each returned tensor owns:
- A copy of the slice captured from the forward pass.
- An `Operation::Chunk(source_id, start_index, axis)` tag recording: which tensor produced it (`source_id`), where the slice begins (`start_index`), and the axis used for slicing (`axis`).

## Goal of Backpropagation
During backprop, each chunk receives its gradient tensor (same shape as that chunk). The task is to accumulate these slices back into the gradient buffer of the original tensor such that:
- Gradients appear only in the slice that produced the chunk.
- Gradients from multiple chunks sum where slices would overlap (for chunking they do not overlap, but accumulation semantics remain consistent). The untouched regions must stay zero.

## Step-by-Step Logic
1. **Look up metadata:** The backward handler matches the enum variant to recover the source tensor ID, `start_index`, and `axis` for this chunk.
2. **Fetch chunk gradient:** `Equation::get_grad_flat_buffer(packet.incoming_grad)` returns the raw gradient slice for the chunk. We convert this into an `ArrayD` so it can be reshaped and sliced using `ndarray` helpers.
3. **Build destination buffer:** Create an all-zero `ArrayD` with the original tensor’s shape. This temporary array will receive the chunk gradient in the exact window that matches the forward slice.
4. **Insert slice:** Using `slice_axis_mut(Axis(axis), start..end)` we copy the chunk gradient into the corresponding range (`start = start_index`, `end = start_index + chunk_size`). This mirrors how the forward pass extracted data.
5. **Return to equation:** Once the slice is in place, convert the array back to `Vec<f32>` and call `Equation::add_tensor_grad(source_id, vec)` to accumulate it into the central gradient store. Because `add_tensor_grad` sums elementwise, this integrates seamlessly if multiple operations feed the same tensor.

## Why It Works
- Every chunk corresponds to a disjoint view; copying each gradient back into its original window reconstructs the full derivative with respect to the parent tensor.
- The start index is all the context needed; the chunk size is implicit in the gradient shape.
- Using `add_tensor_grad` preserves the graph’s additive semantics, matching how PyTorch dispatches `torch.chunk` backwards.

## Edge Considerations
- Uneven splits: the chunk size can vary; we compute `end = start + grad_shape[axis]`, so uneven partitions still map correctly.
- Autograd graph: because we store a separate chunk tensor (with its own ID and shape), the backward step can be invoked independently for each chunk as the graph unwinds.

This approach ensures the backward pass is deterministic, easy to reason about, and mirrors the forward slicing exactly.
