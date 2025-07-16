# LayerNorm Tests to Convert to GGUF Format

This document lists all the tests in `src/nn/layer_norm.rs` that do NOT currently have corresponding GGUF files for PyTorch comparison.

## Tests Already With GGUF Files ✅
- `test_layer_norm_forward_single_sample_pytorch_comparison` - Has `layer_norm_single_sample.gguf`
- `test_layer_norm_forward_zero_variance_pytorch_comparison` - Has `layer_norm_zero_variance.gguf` ✅ **COMPLETED**
- `test_layer_norm_backward_zero_variance_pytorch_comparison` - Has `layer_norm_backward_zero_variance.gguf` ✅ **COMPLETED**
- `test_layer_norm_forward_batch_pytorch_comparison` - Has `layer_norm_batch.gguf` ✅ **COMPLETED**
- `test_layer_norm_backward_simple_pytorch_comparison` - Has `layer_norm_backward_simple.gguf` ✅ **COMPLETED** (Clean implementation)
- `test_layer_norm_forward_known_values_pytorch_comparison` - Has `layer_norm_forward_known_values.gguf` ✅ **COMPLETED**
- `test_layer_norm_backward_batch_pytorch_comparison` - Has `layer_norm_backward_batch.gguf` ✅ **COMPLETED**
- `test_layer_norm_forward_large_values_pytorch_comparison` - Has `layer_norm_forward_large_values.gguf` ✅ **COMPLETED**
- `test_layer_norm_backward_numerical_stability_pytorch_comparison` - Has `layer_norm_backward_numerical_stability.gguf` ✅ **COMPLETED**
- `test_layer_norm_forward_preserves_batch_independence_pytorch_comparison` - Has `layer_norm_forward_preserves_batch_independence.gguf` ✅ **COMPLETED**
- `test_layer_norm_forward_different_feature_sizes_pytorch_comparison` - Has `layer_norm_forward_different_feature_sizes.gguf` ✅ **COMPLETED**
- `test_layer_norm_forward_multiple_calls_pytorch_comparison` - Has `layer_norm_forward_multiple_calls.gguf` ✅ **COMPLETED**
- `test_layer_norm_backward_gradient_flow_pytorch_comparison` - Has `layer_norm_backward_gradient_flow.gguf` ✅ **COMPLETED**

## Tests That Need GGUF Conversion ❌

### Forward Pass Tests
1. **`test_layer_norm_creation`** (lines 69-86)
   - Tests LayerNorm initialization with weight=1.0, bias=0.0
   - Could create GGUF for parameter verification

2. **`test_layer_norm_forward_single_sample`** (lines 89-107) 
   - Original test that was used to create the GGUF comparison
   - Could be deprecated in favor of GGUF version

3. ~~**`test_layer_norm_forward_batch`** (lines 156-182)~~ ✅ **COMPLETED**
   - ~~Tests batch processing with 2 samples of 3 features each~~
   - ~~Input: [[1,2,3], [4,5,6]] shape [2,3]~~

4. ~~**`test_layer_norm_forward_zero_variance`** (lines 185-201)~~ ✅ **COMPLETED**
   - ~~**IMPORTANT**: Tests uniform input [2,2,2,2] - zero variance case~~
   - ~~This addresses the GPT2Block gradient issue we identified~~
   - ~~Input: [2,2,2,2] shape [1,4]~~

5. **`test_layer_norm_forward_known_values`** (lines 204-220)
   - Tests simple case [0,1] with manual calculation verification
   - Input: [0,1] shape [1,2]

6. **`test_layer_norm_forward_negative_values`** (lines 223-241)
   - Tests negative inputs [-2,-1,0]
   - Input: [-2,-1,0] shape [1,3]

7. **`test_layer_norm_forward_large_values`** (lines 244-263)
   - Tests numerical stability with large values [100,200,300]
   - Input: [100,200,300] shape [1,3]

8. **`test_layer_norm_forward_different_feature_sizes`** (lines 266-294)
   - Tests multiple feature sizes: 1, 2, 5, 10
   - Multiple inputs based on feature size

9. **`test_layer_norm_forward_multiple_calls`** (lines 297-315)
   - Tests deterministic behavior with repeated calls
   - Input: [1,2,3] shape [1,3]

10. **`test_layer_norm_forward_preserves_batch_independence`** (lines 318-341)
    - Tests batch independence with different scales
    - Input: [[1,2], [10,20]] shape [2,2]

### Backward Pass Tests
11. **`test_layer_norm_backward_simple`** (lines 346-379)
    - Basic gradient test
    - Input: [1,2] shape [1,2] with requires_grad=true

12. **`test_layer_norm_backward_batch`** (lines 382-422)
    - Batch gradient test
    - Input: [[1,2,3], [4,5,6]] shape [2,3] with requires_grad=true

13. **`test_layer_norm_backward_gradient_flow`** (lines 425-477)
    - Tests gradient differences with different inputs
    - Inputs: [1,2] and [1.1,2.1] shape [1,2]

14. **`test_layer_norm_backward_parameter_gradients`** (lines 480-511)
    - Tests parameter gradient computation
    - Input: [1,4,7] shape [1,3] with requires_grad=true

15. **`test_layer_norm_backward_numerical_stability`** (lines 514-544)
    - Tests gradient stability with large values
    - Input: [1000,2000,3000,4000] shape [1,4] with requires_grad=true

16. ~~**`test_layer_norm_backward_zero_variance`** (lines 547-582)~~ ✅ **COMPLETED**
    - ~~**IMPORTANT**: Tests gradient behavior with uniform input~~
    - ~~This is crucial for understanding the GPT2Block gradient issue~~
    - ~~Input: [5,5,5] shape [1,3] with requires_grad=true~~

17. **`test_layer_norm_backward_in_network`** (lines 585-625)
    - Tests LayerNorm in a Linear->LayerNorm network
    - Input: [1,2] shape [1,2] -> Linear(2,3) -> LayerNorm(3)

18. **`test_layer_norm_backward_different_batch_sizes`** (lines 628-664)
    - Tests different batch sizes: 1, 2, 4
    - Multiple inputs based on batch size

## Priority Tests for GGUF Conversion

### High Priority (directly related to GPT2Block issues):
1. ~~**`test_layer_norm_forward_zero_variance`**~~ ✅ **COMPLETED** - Tests the uniform input problem
2. ~~**`test_layer_norm_backward_zero_variance`**~~ ✅ **COMPLETED** - Tests gradient flow with uniform input
3. ~~**`test_layer_norm_forward_batch`**~~ ✅ **COMPLETED** - Common use case in transformers

### Medium Priority (good coverage tests):
4. ~~**`test_layer_norm_backward_simple`**~~ ✅ **COMPLETED** - Basic backward functionality (Clean implementation)
5. ~~**`test_layer_norm_forward_known_values`**~~ ✅ **COMPLETED** - Manual verification case
6. ~~**`test_layer_norm_backward_batch`**~~ ✅ **COMPLETED** - Batch backward functionality

### Lower Priority (edge cases and stability):
7. ~~**`test_layer_norm_forward_large_values`**~~ ✅ **COMPLETED** - Numerical stability
8. ~~**`test_layer_norm_backward_numerical_stability`**~~ ✅ **COMPLETED** - Gradient stability
9. ~~**`test_layer_norm_forward_preserves_batch_independence`**~~ ✅ **COMPLETED** - Batch independence

## Notes
- Tests 4 and 16 (zero variance cases) are most important for fixing the GPT2Block gradient issue
- Several backward tests have commented-out assertions that might need fixing
- The `test_layer_norm_backward_parameter_gradients` has an early return statement that skips assertions