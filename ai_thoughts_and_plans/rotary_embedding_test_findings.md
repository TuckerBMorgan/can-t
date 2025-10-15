# Rotary Embedding Test Notes

## Current Status
- Added regression tests covering:
  - `compute_cos_sin_returns_outer_product_shape`
  - `apply_rotary_embedding_preserves_input_shape`
  - `apply_rotary_embedding_identity_cos_sin_no_change`
  - `apply_rotary_embedding_quadrature_rotation`
- `compute_cos_sin` now explicitly forms the outer product via `t.unsqueeze(-1) << inv_freq.unsqueeze(0)`.
- Updated `Shape::unsqueeze` to accept PyTorch-style indices and expanded its test coverage.

## Test Results
- All new rotary embedding tests pass after the operand reshaping and `unsqueeze` fixes.

No outstanding issues detected with the rotary embedding helper following these adjustments.
