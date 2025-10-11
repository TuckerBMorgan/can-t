# Replacing `einsum("i,j->ij")` Without Einsum

The expression `torch.einsum("i,j->ij", t, inv_freq)` produces the outer product of two 1-D tensors:
- `t` labelled with axis `i` (length `A`),
- `inv_freq` labelled with axis `j` (length `B`),
- output indices `i j` → result shape `(A, B)` where each entry is `t[i] * inv_freq[j]`.

Because the pattern has no reductions or broadcasts beyond standard outer product semantics, we can decompose it into primitive steps we already support:

1. **Reshape + Broadcast View:**
   - View `t` as shape `(A, 1)` and `inv_freq` as shape `(1, B)`.
   - Broadcasting an elementwise multiply of these views yields an `(A, B)` matrix where each row reuses `t[i]` and each column reuses `inv_freq[j]`.

2. **Outer Product Primitive:**
   - Many libraries expose an explicit `outer` helper that internally performs the same reshape/broadcast idea. Implementing a small `Tensor::outer(&self, other)` would provide the needed functionality without general einsum parsing.

3. **Matmul Proxy:**
   - Interpret `t` as a column vector `(A, 1)` and `inv_freq` as a row vector `(1, B)`, then use matrix multiplication: `(A, 1) @ (1, B) = (A, B)`.
   - This relies on having a lightweight reshape or view operation and the existing `matmul` kernel.

4. **Repeat/Sum Strategy:**
   - Repeat `t` across columns (`repeat` or `expand`) and `inv_freq` across rows, then elementwise multiply. Less efficient, but conceptually straightforward if we lack flexible broadcasting.

5. **Autograd Considerations:**
   - All alternatives boil down to elementwise multiplication after reshaping, so automatic differentiation mirrors the einsum result. We must ensure reshape/view ops do not clone data unnecessarily to keep gradients flowing through the original tensors.

6. **API Design:**
   - Providing a higher-level helper such as `Tensor::outer(&self, &other)` keeps the call site expressive (`freqs = t.outer(inv_freq)`) while avoiding full einsum support. This helper can internally choose the most efficient primitive path (broadcasted multiply vs. matmul) depending on backend capabilities.

So yes, `torch.einsum("i,j->ij")` can be replicated entirely with existing tensor operations. The key is identifying it as an outer product and using reshape + broadcast or matmul to realize it.
