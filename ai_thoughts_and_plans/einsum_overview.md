# Einsum Overview

## What Is Einsum?
“Einsum” is short for **Einstein summation notation**. It is a compact way to describe tensor operations by specifying how axes (a.k.a. dimensions) align, broadcast, and contract. The notation originated in physics to remove explicit summation symbols: whenever an index appears twice, it implies summing over that index. Modern tensor libraries adopt it because a single string like `"ij,jk->ik"` can encode loops, broadcasts, and reductions that would otherwise require verbose code.

In PyTorch, `torch.einsum(spec, *operands)` parses a specification string (`spec`) and a list of tensors (`operands`), then produces a tensor whose axes are defined by the indices after the optional `->` arrow. Without an arrow, PyTorch keeps all indices that appear only once and sums over the rest.

## Core Rules
1. **Indices (labels)** are letters or ellipses in the specification.
   - Lowercase letters (`a`–`z`) or uppercase (`A`–`Z`) denote axes.
   - `...` represents a flexible group of axes that must be identical across operands where it appears.
2. **Each operand** (tensor) is described by a comma-separated substring. The length of that substring must match the tensor’s rank, except when using `...`.
3. **Alignment:** Axes with the same label must have equal sizes (after broadcasting rules are applied for ellipses or singleton dimensions).
4. **Reduction:** If a label appears in multiple operands but not in the output side, einsum sums over that axis.
5. **Output:**
   - If a `->` clause exists, output axes follow the order in that clause.
   - Without `->`, axes that appear exactly once (across all operands) survive in sorted order; repeated labels are reduced.

## “Mental Interpreter” Example
Consider matrix multiplication using `torch.einsum("ij,jk->ik", A, B)`:

1. **Operands:**
   - `A` labelled `i j`, shape `(M, K)`
   - `B` labelled `j k`, shape `(K, N)`
2. **Shared index `j`:** present in both operands, absent from output → sum over `j`.
3. **Remaining indices `i` and `k`:** appear in the output order `i k`, so the result shape is `(M, N)`.
4. **Elementwise view:**
   ```text
   output[i, k] = sum_{j=0}^{K-1} A[i, j] * B[j, k]
   ```

The same rules handle more complex combinations:
- **Inner product:** `"i,i->"` emits a scalar sum.
- **Trace:** `"ii->"` on a square matrix keeps diagonal terms and sums them.
- **Outer product:** `"i,j->ij"` produces a rank-2 tensor with no reduction because indices are unique.
- **Batch matmul with broadcast:** `"bij,bjk->bik"` consumes a batch axis `b` that remains untouched while `j` is reduced.

## Relation to Familiar Ops
Einsum acts as a superset of many primitives:
- Dot products, outer products, matrix multiplication, batched matmul.
- Tensor contractions used in attention blocks (`"bhd,bld->bhl"`).
- Permutations and traces (by reordering or dropping indices).

Because it is fully declarative, einsum can target optimized kernels (BLAS, cuBLAS, Metal Performance Shaders, etc.) depending on the pattern.

## Algorithm Sketch for Implementation
1. **Parse spec:** split on `->` and `,`, validate labels, track ellipses.
2. **Normalize shapes:** expand ellipses to concrete axis lists per operand; check compatibility (size matches unless one side is 1 for broadcasting).
3. **Determine contractions:** any label appearing in multiple operands but not in the output indicates reduction axes.
4. **Plan execution:**
   - Simple cases (vector dot, matmul, batched matmul) can dispatch directly to existing ops.
   - General case may require reshaping operands, applying broadcasts, performing elementwise products, then reducing using `sum` over contracted axes.
5. **Order axes:** permute the surviving axes into the requested output layout.
6. **Gradient story:** autodiff must mirror the forward plan; each contraction corresponds to summing gradients back along operand axes while respecting broadcasts.

## Practical Notes for Our Codebase
- **Type system:** We can represent labels via small enums or string IDs; mapping to axis indices should happen early to avoid string work in kernels.
- **Equation graph:** Einsum can be lowered into primitive operations (`reshape`, `broadcast`, `mul`, `sum`, `transpose`, `matmul`). Capturing this lowering step keeps the computational graph compatible with existing backward passes.
- **Performance:** Pay attention to operand ordering—swapping large tensors early can increase temporary sizes. PyTorch performs heuristics; we can start naive and add optimization passes later.
- **Edge cases:** handle repeated labels within a single operand (diagonal extraction), scalar outputs, zero-dimensional tensors, and ellipsis-only specs (e.g., `"...->..."`).

Building einsum support unlocks a concise API surface while leveraging our existing ops. Once the parser and lowering pipeline are in place, many higher-level features (attention, tensor factorizations) become easier to express and optimize.
