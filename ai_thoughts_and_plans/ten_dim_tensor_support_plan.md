# Ten-Dimensional Tensor Support Plan

## Current 4D Constraints (Inventory)
- `Shape` stores dimensions in a fixed `[usize; 4]`, asserts `len <= 4`, and helpers such as `add_dimension_at_index`, `matmul_broadcast`, and `matmul_shape` are written assuming a hard 4D ceiling.
- Metadata carried in `Operation::{Sum, Mean, Std, Permute}` and `Equation::swap_axes` rely on `[usize; 4]` or `[isize; 4]`, keeping the enum `Copy` but preventing >4 axes from being recorded.
- Utility routines like `utils::padding_dimenions_to_four` and backprop paths that call it (e.g., broadcast and matmul gradients) hard-code four slots.
- `Indexable` stops at `Quadruable`, and `Equation::set_single_value` reshapes tensors to a 4D view before indexing.
- Matmul forward/backward flows (`matmul_op.rs`, `Equation::matmul_tensor`, backends in `crates/cant_{cpu,metal,cuda}`) reshape everything to 4D batches before dispatching to platform code.
- Ancillary logic (permute/movedim tests, `other.rs` einsum helper, parts of nn modules) iterate with `for i in 0..4`, implicitly assuming four leading dims.
- Unit tests cover up to rank-4 tensors and include guard tests that expect a panic when more dims are provided.

## Upgrade Plan
1. **Introduce a shared dimensional ceiling**
   - Define `const MAX_DIMS: usize = 10` in a central module (likely `shape.rs` or a new `central/constants.rs`) and replace magic `4` literals with `MAX_DIMS` (or derived values) across the codebase.
   - Update `Shape` internals to store `[usize; MAX_DIMS]`, adjust constructors, and audit every method for loops/ranges that assume four elements. Add validation to reject dims > `MAX_DIMS`.

2. **Generalize helpers and metadata storage**
   - Replace `padding_dimenions_to_four` with a generic `pad_dimensions` helper returning `[usize; MAX_DIMS]` (or switch to `Vec<usize>` where dynamic length is cleaner).
   - Update `Operation` variants (`Sum`, `Mean`, `Std`, `Permute`) to carry `[usize; MAX_DIMS]` / `[isize; MAX_DIMS]` and revise creators/backprop consumers to iterate over `num_axes` instead of four fixed slots.
   - Ensure `Operation` can remain `Copy`; if not practical, accept making it `Clone` + `PartialEq` and update the ecosystem accordingly.

3. **Rework indexing utilities**
   - Extend `Indexable` to support up to 10 indices (e.g., add `Quintuple`+ enums or migrate to a small vector inside the variant).
   - Refactor `Equation::set_single_value` to compute strides for arbitrary rank rather than padding to four dims.
   - Audit any other APIs that expose index literals (tests, convenience methods) and expand them to cover higher rank.

4. **Broadcasting, reshape, and shape algebra**
   - Review `Shape::{add_dimension_at_index, remove_index, permute, matmul_broadcast, matmul_shape, broadcast_shape}` to ensure logic generalizes to rank-10 tensors (e.g., iterate over `rank` instead of assuming four slots, rebuild matmul broadcast rules over an arbitrary batch prefix).
   - Add unit tests that exercise broadcasting and matmul shape inference with 5–10D inputs.

5. **Matmul data flow overhaul**
   - Update `matmul_op.rs` forward/backward paths to operate on arbitrary ranks: flatten batch dims into a single leading dimension (or compute strides dynamically) before calling backend kernels.
   - Redesign `Equation::matmul_tensor` / `matmul_vector` to accept `&[usize]` and perform the same flattening; if backends still require 4D, add a conversion layer that coalesces extra batch dims while preserving semantics.
   - Modify backend implementations in `crates/cant_cpu`, `cant_metal`, and `cant_cuda` to accept either dynamic shapes or `[usize; MAX_DIMS]`, updating validation logic and loop nests accordingly.
   - Ensure backward swaps (`Equation::swap_axes`) and gradient matmuls mirror the new representations; rewrite the helper to operate on arbitrary ranks using computed strides instead of four nested loops.

6. **Reduction, permutation, and other ops**
   - Update `sum`, `mean`, `std`, `permute`, `movedim`, `diagonal`, `chunk`, and any other ops that stash axis metadata to use the new constant, iterating over `num_axes` rather than hardcoded `0..4`.
   - Revisit broadcast backprop (`Operation::BroadCast` case in `Equation::backward_for_value`) to use the generalized padding helper.
   - Review nn layers (layer norm, rmsnorm, embedding) and utilities for explicit `0..4` loops; convert to rank-driven iteration or helper functions.
   - Expand the `other.rs` einsum helper to accept `[usize; MAX_DIMS]` or a dynamic slice.

7. **Tests and fixtures**
   - Update existing tests that expect a panic at rank>4 and add new positive coverage for rank 5–10 across broadcasting, matmul, reductions, permutations, and NN layers.
   - Add regression tests ensuring backend outputs match CPU reference for higher-rank batched matmul.
   - Consider adding GGUF fixtures with tensors exceeding four dims if integration tests rely on them.

8. **Validation & cleanup**
   - Run `cargo fmt`, `cargo clippy --all-targets --all-features`, and `cargo test` across the workspace.
   - For backend changes, execute platform-specific checks (e.g., Metal-enabled tests if available).
   - Document any limitations (e.g., still capped at `MAX_DIMS = 10`) in `README` or developer docs.

## Open Questions / Risks
- If `Operation` loses `Copy`, assess performance impact on autograd bookkeeping and consider using `SmallVec<[usize; MAX_DIMS]>` to keep stack allocation.
- Backend kernels may need non-trivial refactors; if rewriting to fully dynamic dims is costly, evaluate flattening multiple batch axes into a single dimension prior to dispatch.
- Confirm GGUF loader and any serialization code does not embed 4D expectations.
- Ensure memory growth from larger arrays is acceptable; consider lazy allocation or per-rank storage if not.

