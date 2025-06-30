# 📊 Standard Deviation Operations in Tensor Libraries: A Complete Guide

*Understanding the mathematical foundations, implementation approaches, and computational challenges*

---

## 📚 Table of Contents

1. [Mathematical Foundation](#mathematical-foundation)
2. [Tensor Context and Axes](#tensor-context-and-axes)  
3. [Forward Pass Approaches](#forward-pass-approaches)
4. [Backward Pass (Gradient Computation)](#backward-pass-gradient-computation)
5. [Implementation Challenges](#implementation-challenges)
6. [Practical Examples](#practical-examples)
7. [Performance Considerations](#performance-considerations)

---

## 🧮 Mathematical Foundation

### What is Standard Deviation?

Standard deviation (σ) measures how spread out data points are from their mean (μ). It's the square root of variance.

**Mathematical Definition:**

```
For a dataset X = {x₁, x₂, ..., xₙ}:

1. Mean: μ = (1/N) * Σᵢ xᵢ
2. Variance: σ² = (1/N) * Σᵢ (xᵢ - μ)²  [Population]
              or (1/(N-1)) * Σᵢ (xᵢ - μ)²  [Sample]
3. Standard Deviation: σ = √(σ²)
```

### Population vs Sample Standard Deviation

- **Population (ddof=0):** Divide by N (PyTorch default, our implementation)
- **Sample (ddof=1):** Divide by N-1 (NumPy default, statistical analysis)

**Example:**
```
Data: [1, 2, 3, 4, 5]
Mean: μ = 3.0

Population variance: σ² = [(1-3)² + (2-3)² + (3-3)² + (4-3)² + (5-3)²] / 5
                         = [4 + 1 + 0 + 1 + 4] / 5 = 2.0
Population std: σ = √2.0 ≈ 1.414

Sample variance: σ² = 10 / 4 = 2.5
Sample std: σ = √2.5 ≈ 1.581
```

---

## 🎯 Tensor Context and Axes

### What Does "Std Along Axes" Mean?

In tensor operations, we compute standard deviation along specified dimensions (axes), collapsing those dimensions while preserving others.

**Visual Example - 2D Tensor:**

```
Input tensor [2, 3]:
[[1, 2, 3],
 [4, 5, 6]]

std(axis=0) - Along rows (compute std of each column):
Result [3]: [std([1,4]), std([2,5]), std([3,6])]
         = [1.5, 1.5, 1.5]

std(axis=1) - Along columns (compute std of each row):  
Result [2]: [std([1,2,3]), std([4,5,6])]
         = [0.816, 0.816]

std(axis=[0,1]) - Along all axes (global std):
Result [1]: [std([1,2,3,4,5,6])]
         = [1.708]
```

### Multi-Dimensional Complexity (UPDATED)

**3D Example [2, 2, 3] - Detailed Breakdown:**

```
Input tensor shape [2, 2, 3]:
[[[1, 2, 3],     ← layer 0, row 0
  [4, 5, 6]],    ← layer 0, row 1
 [[7, 8, 9],     ← layer 1, row 0  
  [10,11,12]]]   ← layer 1, row 1
```


**Single Axis Operations:**

```
std(axis=0) → [2, 3]: std across layers (first dimension)
Result[i,j] = std of values at position [*, i, j] across all layers

Position [0,0]: std([1, 7]) = 3.0
Position [0,1]: std([2, 8]) = 3.0  
Position [0,2]: std([3, 9]) = 3.0
Position [1,0]: std([4, 10]) = 3.0
Position [1,1]: std([5, 11]) = 3.0
Position [1,2]: std([6, 12]) = 3.0

Final result: [[3, 3, 3],
               [3, 3, 3]]
```

```
std(axis=1) → [2, 3]: std across rows (second dimension)  
Result[i,j] = std of values at position [i, *, j] across all rows

Layer 0:
Position [0,0]: std([1, 4]) = 1.5
Position [0,1]: std([2, 5]) = 1.5
Position [0,2]: std([3, 6]) = 1.5

Layer 1:  
Position [1,0]: std([7, 10]) = 1.5
Position [1,1]: std([8, 11]) = 1.5
Position [1,2]: std([9, 12]) = 1.5

Final result: [[1.5, 1.5, 1.5],
               [1.5, 1.5, 1.5]]
```

```
std(axis=2) → [2, 2]: std across columns (third dimension)
Result[i,j] = std of values at position [i, j, *] across all columns

Position [0,0]: std([1, 2, 3]) = 0.816
Position [0,1]: std([4, 5, 6]) = 0.816  
Position [1,0]: std([7, 8, 9]) = 0.816
Position [1,1]: std([10, 11, 12]) = 0.816

Final result: [[0.816, 0.816],
               [0.816, 0.816]]
```

**Multi-Axis Operations (The Complex Cases):**

```
std(axis=[0,2]) → [2]: std across first and third dimensions
This collapses dimensions 0 and 2, keeping only dimension 1 (rows)

For each row position j, we collect ALL values where the row index is j,
regardless of layer or column:

Row 0 (j=0): Collect all [layer, 0, column] positions
  [0,0,0]=1, [0,0,1]=2, [0,0,2]=3, [1,0,0]=7, [1,0,1]=8, [1,0,2]=9
  Values: [1, 2, 3, 7, 8, 9]
  std([1, 2, 3, 7, 8, 9]) = 3.162

Row 1 (j=1): Collect all [layer, 1, column] positions  
  [0,1,0]=4, [0,1,1]=5, [0,1,2]=6, [1,1,0]=10, [1,1,1]=11, [1,1,2]=12
  Values: [4, 5, 6, 10, 11, 12]
  std([4, 5, 6, 10, 11, 12]) = 3.162

Final result: [3.162, 3.162]
```

```
std(axis=[1,2]) → [2]: std across second and third dimensions  
This collapses dimensions 1 and 2, keeping only dimension 0 (layers)

For each layer i, we collect ALL values in that layer:

Layer 0 (i=0): All values in first layer
  Values: [1, 2, 3, 4, 5, 6]  
  std([1, 2, 3, 4, 5, 6]) = 1.708

Layer 1 (i=1): All values in second layer
  Values: [7, 8, 9, 10, 11, 12]
  std([7, 8, 9, 10, 11, 12]) = 1.708  

Final result: [1.708, 1.708]
```

```
std(axis=[0,1]) → [3]: std across first and second dimensions
This collapses dimensions 0 and 1, keeping only dimension 2 (columns)

For each column position k, we collect ALL values in that column
across all layers and rows:

Column 0 (k=0): All values where column index is 0
  [0,0,0]=1, [0,1,0]=4, [1,0,0]=7, [1,1,0]=10
  Values: [1, 4, 7, 10]
  std([1, 4, 7, 10]) = 3.708

Column 1 (k=1): All values where column index is 1  
  [0,0,1]=2, [0,1,1]=5, [1,0,1]=8, [1,1,1]=11
  Values: [2, 5, 8, 11]
  std([2, 5, 8, 11]) = 3.708

Column 2 (k=2): All values where column index is 2
  [0,0,2]=3, [0,1,2]=6, [1,0,2]=9, [1,1,2]=12  
  Values: [3, 6, 9, 12]
  std([3, 6, 9, 12]) = 3.708

Final result: [3.708, 3.708, 3.708]
```

```
std(axis=[0,1,2]) → [1]: global std (scalar)
All values: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
std = 3.452

Final result: [3.452]
```

**Pattern Recognition:**
- **Single axis:** Simple element-wise std along one dimension
- **Multi-axis:** Collect values by "grouping" along remaining dimensions
- **Output shape:** Remove the specified axes from input shape
- **Computation:** For each output position, gather all input values that map to it

---

## 🚀 Forward Pass Approaches

### Approach 1: Sequential Axis Processing

**Concept:** Apply single-axis std operations sequentially for multi-axis cases.

```rust
fn std_sequential(tensor: &Array, axes: &[usize]) -> Array {
    let mut result = tensor.clone();
    
    // Sort axes in descending order to avoid index shifting
    let mut sorted_axes = axes.to_vec();
    sorted_axes.sort_by(|a, b| b.cmp(a));
    
    for &axis in &sorted_axes {
        result = result.std_axis(Axis(axis), 0.0);
    }
    
    result
}
```

**Pros:**
- ✅ Simple to implement
- ✅ Leverages existing single-axis operations
- ✅ Clear logic flow

**Cons:**
- ❌ **Mathematically incorrect for multi-axis cases!**
- ❌ Sequential reduction changes the meaning
- ❌ Can produce wrong results (like 0.0 when all intermediate values are equal)

**Why It Fails:**
```
Input: [[1,2,3], [4,5,6]]  (shape [2,3])

Sequential std([0,1]):
1. std(axis=0) → [1.5, 1.5, 1.5]  (shape [3])
2. std(axis=0) → 0.0  (shape [1]) ← WRONG! All values equal

Correct std([0,1]): 
Global std([1,2,3,4,5,6]) = 1.708 ← RIGHT!
```

### Approach 2: Reshape and Flatten Method

**Concept:** Reshape tensor to group the target axes, then compute std on the flattened group.

```rust
fn std_reshape_flatten(tensor: &Array, axes: &[usize]) -> Array {
    let original_shape = tensor.shape();
    
    if axes.len() == 1 {
        // Single axis - use ndarray directly
        return tensor.std_axis(Axis(axes[0]), 0.0);
    }
    
    // Multi-axis: determine which axes to keep vs flatten
    let keep_axes: Vec<usize> = (0..original_shape.len())
        .filter(|i| !axes.contains(i))
        .collect();
    
    if keep_axes.is_empty() {
        // All axes being std'd - global std
        let flattened = tensor.view().into_shape((tensor.len(),))?;
        let std_val = flattened.std(0.0);
        return Array::from_elem((), std_val).into_dyn();
    } else {
        // Some axes kept - reshape and compute std along flattened dimension
        let (keep_shape, flatten_size) = compute_reshape_dimensions(
            original_shape, &keep_axes, axes
        );
        
        let reshaped = tensor.view().into_shape((keep_shape, flatten_size))?;
        return reshaped.std_axis(Axis(1), 0.0);
    }
}
```

**Pros:**
- ✅ Mathematically correct
- ✅ Handles all axis combinations properly
- ✅ Efficient memory usage
- ✅ Leverages optimized ndarray operations

**Cons:**
- ❌ More complex implementation
- ❌ Requires careful shape manipulation
- ❌ May require memory copying for non-contiguous views

### Approach 3: Direct Mathematical Implementation

**Concept:** Implement the mathematical formula directly with explicit indexing.

```rust
fn std_direct_math(tensor: &Array, axes: &[usize]) -> Array {
    let original_shape = tensor.shape();
    let output_shape = compute_output_shape(original_shape, axes);
    let mut result = Array::zeros(output_shape);
    
    // For each output position, compute std of corresponding input elements
    for output_idx in result.indexed_iter_mut() {
        let input_indices = map_output_to_input_indices(output_idx.0, axes, original_shape);
        
        // Collect values for this std computation
        let values: Vec<f32> = input_indices.iter()
            .map(|idx| tensor[idx])
            .collect();
        
        // Compute std directly
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let variance = values.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>() / values.len() as f32;
        let std_val = variance.sqrt();
        
        *output_idx.1 = std_val;
    }
    
    result
}
```

**Pros:**
- ✅ Maximum control over computation
- ✅ Easy to understand and debug
- ✅ Handles edge cases explicitly
- ✅ No reshaping complexity

**Cons:**
- ❌ Much slower than vectorized operations
- ❌ More complex indexing logic
- ❌ Harder to optimize

---

## ⬅️ Backward Pass (Gradient Computation)

### Mathematical Derivation

The gradient of standard deviation with respect to input elements:

```
Given: σ = √(Σᵢ(xᵢ - μ)² / N)

Step 1: Chain rule
∂L/∂xⱼ = (∂L/∂σ) * (∂σ/∂xⱼ)

Step 2: Derivative of std
∂σ/∂xⱼ = (1/σ) * (1/N) * (xⱼ - μ)

Step 3: Final gradient
∂L/∂xⱼ = (∂L/∂σ) * (1/σ) * (1/N) * (xⱼ - μ)
```

**Physical Interpretation:**
- If `xⱼ > μ` (above mean): positive contribution to gradient
- If `xⱼ < μ` (below mean): negative contribution to gradient  
- If `xⱼ = μ` (at mean): zero contribution
- Larger deviations have proportionally larger gradients

### Implementation Challenges

#### Challenge 1: Zero Standard Deviation

**Problem:** When σ = 0 (all values equal), gradient formula has division by zero.

**Solutions:**
```rust
// Option 1: Clamp to small epsilon
let std_clamped = std_val.max(1e-8);
let grad = incoming_grad * (1.0 / std_clamped) * (1.0 / n) * (x - mean);

// Option 2: Special case handling
let grad = if std_val < 1e-8 { 
    0.0  // No gradient when no variation
} else {
    incoming_grad * (1.0 / std_val) * (1.0 / n) * (x - mean)
};

// Option 3: Use stable computation
let inv_std = if std_val > 1e-8 { 1.0 / std_val } else { 0.0 };
let grad = incoming_grad * inv_std * (1.0 / n) * (x - mean);
```

#### Challenge 2: Multi-Axis Gradient Broadcasting

**Problem:** Gradients must be broadcast back to original tensor shape.

```rust
fn backward_for_std_multi_axis(
    incoming_grad: &Array,
    source_data: &Array, 
    axes: &[usize]
) -> Array {
    let source_shape = source_data.shape();
    let mut output_grad = Array::zeros(source_shape);
    
    // For each element in source tensor
    for (source_idx, &source_val) in source_data.indexed_iter() {
        // Find which std computation this element belongs to
        let std_group_idx = map_source_to_std_group(source_idx, axes, source_shape);
        
        // Get the corresponding incoming gradient
        let incoming = incoming_grad[std_group_idx];
        
        // Compute mean and std for this group
        let (group_mean, group_std) = compute_group_stats(source_data, source_idx, axes);
        
        // Apply gradient formula
        let grad = if group_std > 1e-8 {
            incoming * (1.0 / group_std) * (1.0 / group_size) * (source_val - group_mean)
        } else {
            0.0
        };
        
        output_grad[source_idx] = grad;
    }
    
    output_grad
}
```

#### Challenge 3: Numerical Stability

**Problem:** Standard deviation computation can be numerically unstable.

**Stable Computation Methods:**

```rust
// Method 1: Two-pass algorithm (more stable)
fn stable_std_two_pass(values: &[f32]) -> (f32, f32) {
    let n = values.len() as f32;
    
    // First pass: compute mean
    let mean = values.iter().sum::<f32>() / n;
    
    // Second pass: compute variance
    let variance = values.iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f32>() / n;
    
    (mean, variance.sqrt())
}

// Method 2: Welford's online algorithm (most stable)
fn welford_std(values: &[f32]) -> (f32, f32) {
    let mut mean = 0.0;
    let mut m2 = 0.0;
    
    for (i, &value) in values.iter().enumerate() {
        let n = (i + 1) as f32;
        let delta = value - mean;
        mean += delta / n;
        let delta2 = value - mean;
        m2 += delta * delta2;
    }
    
    let variance = m2 / values.len() as f32;
    (mean, variance.sqrt())
}
```

---

## 🚧 Implementation Challenges

### 1. Memory Layout and Performance

**Challenge:** Tensor operations must respect memory layout for performance.

```rust
// Bad: Non-contiguous access pattern
for axis in axes {
    result = result.std_axis(Axis(axis), 0.0);  // May require copying
}

// Good: Reshape to create contiguous layout
let reshaped = tensor.view().into_shape(optimized_shape)?;
let result = reshaped.std_axis(Axis(target_axis), 0.0);
```

### 2. Shape Broadcasting Complexity

**Example Problem:**
```
Input shape: [3, 4, 5, 6]
std(axes=[1, 3]) should produce shape: [3, 5]

Challenge: Map each output element to its input group efficiently
```

**Solution Approach:**
```rust
fn compute_std_groups(input_shape: &[usize], axes: &[usize]) -> StdGroupMap {
    // Precompute which input indices contribute to each output index
    let output_shape = remove_axes_from_shape(input_shape, axes);
    let mut groups = HashMap::new();
    
    for input_idx in input_shape.indices() {
        let output_idx = project_to_output_space(input_idx, axes);
        groups.entry(output_idx).or_insert_with(Vec::new).push(input_idx);
    }
    
    StdGroupMap { groups, output_shape }
}
```

### 3. Edge Cases

**Empty Axes:**
```rust
if axes.is_empty() {
    // Return tensor filled with zeros (as per current implementation)
    return Array::zeros(tensor.shape());
}
```

**Single Element Groups:**
```rust
if group_size == 1 {
    // std of single element is always 0
    return 0.0;
}
```

**Out of Bounds Axes:**
```rust
for &axis in axes {
    if axis >= tensor.ndim() {
        panic!("axis {} is out of bounds for tensor with {} dimensions", axis, tensor.ndim());
    }
}
```

**Duplicate Axes:**
```rust
let unique_axes: HashSet<_> = axes.iter().collect();
if unique_axes.len() != axes.len() {
    panic!("duplicate axes not allowed");
}
```

---

## 🧪 Practical Examples

### Example 1: Simple 1D Case

```rust
// Input: [1.0, 2.0, 3.0, 4.0, 5.0]
// std(axis=0)

// Forward:
let mean = 3.0;
let variance = [(1-3)² + (2-3)² + (3-3)² + (4-3)² + (5-3)²] / 5 = 2.0;
let std = √2.0 = 1.414;

// Backward (incoming_grad = 1.0):
let n = 5.0;
let grads = [
    1.0 * (1/1.414) * (1/5) * (1-3) = -0.283,
    1.0 * (1/1.414) * (1/5) * (2-3) = -0.141, 
    1.0 * (1/1.414) * (1/5) * (3-3) =  0.000,
    1.0 * (1/1.414) * (1/5) * (4-3) =  0.141,
    1.0 * (1/1.414) * (1/5) * (5-3) =  0.283
];
// Note: gradients sum to 0 (as expected for mean-centered operation)
```

### Example 2: Multi-Axis 2D Case

```rust
// Input: [[1, 2], [3, 4]]  (shape [2, 2])
// std(axes=[0, 1]) - global std

// Forward:
let all_values = [1, 2, 3, 4];
let mean = 2.5;
let variance = [(1-2.5)² + (2-2.5)² + (3-2.5)² + (4-2.5)²] / 4 = 1.25;
let std = √1.25 = 1.118;

// Backward (incoming_grad = 1.0):
let grads = [
    [1.0 * (1/1.118) * (1/4) * (1-2.5), 1.0 * (1/1.118) * (1/4) * (2-2.5)],
    [1.0 * (1/1.118) * (1/4) * (3-2.5), 1.0 * (1/1.118) * (1/4) * (4-2.5)]
] = [
    [-0.336, -0.112],
    [ 0.112,  0.336]
];
```

### Example 3: Axis-Specific 2D Case  

```rust
// Input: [[1, 2, 3], [4, 5, 6]]  (shape [2, 3])
// std(axis=0) - std of each column

// Forward:
// Column 0: std([1, 4]) = 1.5
// Column 1: std([2, 5]) = 1.5  
// Column 2: std([3, 6]) = 1.5
// Result: [1.5, 1.5, 1.5]

// Backward (incoming_grad = [1.0, 1.0, 1.0]):
// For each column independently:
let grads = [
    // Row 0: each element gets gradient from its column's std
    [1.0 * (1/1.5) * (1/2) * (1-2.5), 1.0 * (1/1.5) * (1/2) * (2-3.5), 1.0 * (1/1.5) * (1/2) * (3-4.5)],
    // Row 1: 
    [1.0 * (1/1.5) * (1/2) * (4-2.5), 1.0 * (1/1.5) * (1/2) * (5-3.5), 1.0 * (1/1.5) * (1/2) * (6-4.5)]
] = [
    [-0.500, -0.500, -0.500],
    [ 0.500,  0.500,  0.500]
];
```

---

## ⚡ Performance Considerations

### 1. Memory Access Patterns

**Optimal:** Operations that follow memory layout
```rust
// Good: std along last axis (contiguous in memory)
tensor.std_axis(Axis(tensor.ndim() - 1), 0.0)

// Potentially slower: std along first axis (strided access)
tensor.std_axis(Axis(0), 0.0)
```

### 2. Vectorization Opportunities

**SIMD-Friendly Operations:**
- Mean computation: Pure vector sum and divide
- Variance computation: Vector subtract, square, and sum
- Square root: Can be vectorized with appropriate CPU instructions

### 3. Memory Usage

**Trade-offs:**
```rust
// Memory-efficient but slower
fn std_in_place_iterations() {
    // Compute mean and std with multiple passes over data
    // Lower memory usage, higher computational cost
}

// Memory-intensive but faster  
fn std_with_intermediate_storage() {
    // Store intermediate results (means, squared differences)
    // Higher memory usage, lower computational cost
}
```

### 4. Parallelization Opportunities

**Independent Computations:**
```rust
// Each output element can be computed independently
parallel_for(output_indices, |output_idx| {
    let input_group = map_to_input_group(output_idx);
    let std_val = compute_std_for_group(input_group);
    output[output_idx] = std_val;
});
```

---

## 🎯 Recommended Implementation Strategy

### Phase 1: Core Functionality
1. **Single-axis std:** Leverage ndarray's optimized `std_axis`
2. **Multi-axis std:** Use reshape-and-flatten approach
3. **Basic backward pass:** Implement gradient formula with stability checks

### Phase 2: Robustness  
1. **Edge case handling:** Zero std, empty axes, out-of-bounds
2. **Numerical stability:** Implement Welford's algorithm for large datasets
3. **Comprehensive testing:** Unit tests covering all axis combinations

### Phase 3: Performance
1. **Memory optimization:** Minimize copying and reshaping
2. **Vectorization:** Use SIMD operations where possible  
3. **Parallelization:** Parallelize independent computations

### Phase 4: Advanced Features
1. **Different ddof values:** Support both population and sample std
2. **Mixed precision:** Support f16/f32/f64 operations
3. **GPU acceleration:** Extend to CUDA/Metal implementations

---

## 📝 Summary

Standard deviation operations in tensor libraries are deceptively complex:

**Key Insights:**
- ✅ **Mathematical correctness is paramount** - sequential axis processing fails
- ✅ **Reshape-and-flatten is the most reliable approach** for multi-axis cases  
- ✅ **Backward pass requires careful gradient broadcasting** and stability handling
- ✅ **Edge cases (zero std, single elements) need explicit handling**
- ✅ **Performance requires attention to memory layout** and vectorization

**Common Pitfalls:**
- ❌ Using sequential single-axis operations for multi-axis std
- ❌ Ignoring numerical stability (division by zero, floating-point precision)
- ❌ Incorrect gradient broadcasting in backward pass
- ❌ Not handling edge cases consistently

The current Cant implementation has the right structure but needs:
1. **Fixed multi-axis computation** (replace sequential with reshape approach)
2. **Complete backward pass implementation** with stability checks
3. **Comprehensive edge case handling**

This foundation will make std operations robust, mathematically correct, and ready for production use in machine learning applications.

---

## 📋 Appendix: Helper Function Implementations

This section provides complete implementations of the helper functions referenced throughout the guide.

### A.1 Shape and Dimension Utilities

```rust
/// Computes the reshape dimensions for multi-axis std operations
/// Returns (keep_dimensions_shape, flatten_dimension_size)
fn compute_reshape_dimensions(
    original_shape: &[usize], 
    keep_axes: &[usize], 
    std_axes: &[usize]
) -> (Vec<usize>, usize) {
    let mut keep_shape = Vec::new();
    let mut flatten_size = 1;
    
    // Calculate shape for dimensions we're keeping
    for &axis in keep_axes {
        keep_shape.push(original_shape[axis]);
    }
    
    // Calculate total size of dimensions we're flattening
    for &axis in std_axes {
        flatten_size *= original_shape[axis];
    }
    
    (keep_shape, flatten_size)
}

/// Removes specified axes from a shape to compute output shape
fn remove_axes_from_shape(input_shape: &[usize], axes: &[usize]) -> Vec<usize> {
    input_shape.iter()
        .enumerate()
        .filter(|(i, _)| !axes.contains(i))
        .map(|(_, &dim)| dim)
        .collect()
}

/// Computes output shape for std operation
fn compute_output_shape(input_shape: &[usize], axes: &[usize]) -> Vec<usize> {
    let output_shape = remove_axes_from_shape(input_shape, axes);
    
    // If all axes are removed, return shape [1] for scalar result
    if output_shape.is_empty() {
        vec![1]
    } else {
        output_shape
    }
}
```

### A.2 Index Mapping Functions

```rust
/// Maps output indices back to corresponding input indices for std computation
/// Returns all input indices that contribute to the given output position
fn map_output_to_input_indices(
    output_idx: &[usize],
    std_axes: &[usize], 
    original_shape: &[usize]
) -> Vec<Vec<usize>> {
    let mut input_indices = Vec::new();
    
    // Generate all combinations of indices for the std axes
    let std_axis_ranges: Vec<Vec<usize>> = std_axes.iter()
        .map(|&axis| (0..original_shape[axis]).collect())
        .collect();
    
    // Generate cartesian product of all std axis values
    let std_combinations = cartesian_product(&std_axis_ranges);
    
    for std_idx_combination in std_combinations {
        let mut full_input_idx = vec![0; original_shape.len()];
        
        // Fill in the kept dimensions from output index
        let mut output_pos = 0;
        for (dim, &size) in original_shape.iter().enumerate() {
            if !std_axes.contains(&dim) {
                full_input_idx[dim] = output_idx[output_pos];
                output_pos += 1;
            }
        }
        
        // Fill in the std dimensions from current combination
        for (i, &axis) in std_axes.iter().enumerate() {
            full_input_idx[axis] = std_idx_combination[i];
        }
        
        input_indices.push(full_input_idx);
    }
    
    input_indices
}

/// Helper function to compute cartesian product of index ranges
fn cartesian_product(lists: &[Vec<usize>]) -> Vec<Vec<usize>> {
    if lists.is_empty() {
        return vec![vec![]];
    }
    
    let mut result = vec![vec![]];
    
    for list in lists {
        let mut new_result = Vec::new();
        for existing in &result {
            for &item in list {
                let mut new_combo = existing.clone();
                new_combo.push(item);
                new_result.push(new_combo);
            }
        }
        result = new_result;
    }
    
    result
}

/// Maps source tensor index to which std group it belongs to
/// Returns the output index in the result tensor
fn map_source_to_std_group(
    source_idx: &[usize], 
    std_axes: &[usize], 
    _source_shape: &[usize]
) -> Vec<usize> {
    // Project source index to output space by removing std axes
    source_idx.iter()
        .enumerate()
        .filter(|(i, _)| !std_axes.contains(i))
        .map(|(_, &idx)| idx)
        .collect()
}
```

### A.3 Statistical Computation Functions

```rust
/// Computes mean and standard deviation for a group of values
/// Returns (mean, std_deviation)
fn compute_group_stats(values: &[f32]) -> (f32, f32) {
    if values.is_empty() {
        return (0.0, 0.0);
    }
    
    if values.len() == 1 {
        return (values[0], 0.0);
    }
    
    // Two-pass algorithm for numerical stability
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    
    let variance = values.iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f32>() / values.len() as f32;
    
    (mean, variance.sqrt())
}

/// Welford's algorithm for numerically stable online mean and variance
struct WelfordAccumulator {
    count: usize,
    mean: f32,
    m2: f32,
}

impl WelfordAccumulator {
    fn new() -> Self {
        Self { count: 0, mean: 0.0, m2: 0.0 }
    }
    
    fn update(&mut self, value: f32) {
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f32;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;
    }
    
    fn finalize(&self) -> (f32, f32) {
        if self.count < 2 {
            return (self.mean, 0.0);
        }
        
        let variance = self.m2 / self.count as f32;
        (self.mean, variance.sqrt())
    }
}

/// Compute stats for a group using Welford's algorithm
fn compute_group_stats_welford(values: &[f32]) -> (f32, f32) {
    let mut acc = WelfordAccumulator::new();
    for &value in values {
        acc.update(value);
    }
    acc.finalize()
}
```

### A.4 Complete Std Group Mapping System

```rust
use std::collections::HashMap;

/// Precomputed mapping from output indices to input index groups
struct StdGroupMap {
    groups: HashMap<Vec<usize>, Vec<Vec<usize>>>,
    output_shape: Vec<usize>,
}

impl StdGroupMap {
    /// Creates a complete mapping for std computation
    fn new(input_shape: &[usize], std_axes: &[usize]) -> Self {
        let output_shape = compute_output_shape(input_shape, std_axes);
        let mut groups = HashMap::new();
        
        // For each possible output index
        for output_idx in index_iterator(&output_shape) {
            let input_indices = map_output_to_input_indices(
                &output_idx, std_axes, input_shape
            );
            groups.insert(output_idx, input_indices);
        }
        
        Self { groups, output_shape }
    }
    
    /// Gets all input indices that contribute to a given output position
    fn get_input_indices(&self, output_idx: &[usize]) -> Option<&Vec<Vec<usize>>> {
        self.groups.get(output_idx)
    }
    
    /// Iterator over all output positions
    fn output_indices(&self) -> impl Iterator<Item = Vec<usize>> + '_ {
        index_iterator(&self.output_shape)
    }
}

/// Iterator over all possible indices for a given shape
fn index_iterator(shape: &[usize]) -> impl Iterator<Item = Vec<usize>> + '_ {
    IndexIterator::new(shape)
}

struct IndexIterator {
    shape: Vec<usize>,
    current: Vec<usize>,
    done: bool,
}

impl IndexIterator {
    fn new(shape: &[usize]) -> Self {
        let current = vec![0; shape.len()];
        let done = shape.iter().any(|&dim| dim == 0);
        
        Self {
            shape: shape.to_vec(),
            current,
            done,
        }
    }
}

impl Iterator for IndexIterator {
    type Item = Vec<usize>;
    
    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        
        let result = self.current.clone();
        
        // Increment indices (like odometer)
        let mut carry = 1;
        for i in (0..self.current.len()).rev() {
            self.current[i] += carry;
            if self.current[i] >= self.shape[i] {
                self.current[i] = 0;
                carry = 1;
            } else {
                carry = 0;
                break;
            }
        }
        
        if carry == 1 {
            self.done = true;
        }
        
        Some(result)
    }
}
```

### A.5 Backward Pass Helper Functions

```rust
/// Computes gradients for std operation using precomputed group mapping
fn compute_std_gradients(
    incoming_grad: &Array,
    source_data: &Array,
    group_map: &StdGroupMap,
) -> Array {
    let mut output_grad = Array::zeros(source_data.shape());
    
    // For each output position
    for output_idx in group_map.output_indices() {
        let input_indices = group_map.get_input_indices(&output_idx).unwrap();
        
        // Collect values for this std group
        let values: Vec<f32> = input_indices.iter()
            .map(|idx| source_data[&idx[..]])
            .collect();
        
        // Compute group statistics
        let (group_mean, group_std) = compute_group_stats_welford(&values);
        
        // Get incoming gradient for this output position
        let incoming = if output_idx.is_empty() { 
            incoming_grad[[0]] // Scalar case
        } else {
            incoming_grad[&output_idx[..]]
        };
        
        // Compute gradient for each input in this group
        let n = values.len() as f32;
        let inv_std = if group_std > 1e-8 { 1.0 / group_std } else { 0.0 };
        
        for (value, input_idx) in values.iter().zip(input_indices.iter()) {
            let grad = incoming * inv_std * (1.0 / n) * (value - group_mean);
            output_grad[&input_idx[..]] = grad;
        }
    }
    
    output_grad
}

/// Safer gradient computation with numerical stability checks
fn compute_stable_std_gradient(
    incoming_grad: f32,
    value: f32,
    group_mean: f32,
    group_std: f32,
    group_size: usize,
) -> f32 {
    if group_size <= 1 {
        return 0.0; // No gradient for single-element groups
    }
    
    // Use epsilon clamping for numerical stability
    let epsilon = 1e-8;
    let stable_std = group_std.max(epsilon);
    
    let n = group_size as f32;
    incoming_grad * (1.0 / stable_std) * (1.0 / n) * (value - group_mean)
}
```

### A.6 Memory-Efficient Implementation

```rust
/// Memory-efficient std computation that avoids creating large intermediate arrays
fn compute_std_memory_efficient(
    tensor: &Array,
    axes: &[usize],
) -> Result<Array, Box<dyn std::error::Error>> {
    let output_shape = compute_output_shape(tensor.shape(), axes);
    let mut result = Array::zeros(&output_shape[..]);
    
    // Process each output element independently
    for output_idx in index_iterator(&output_shape) {
        let input_indices = map_output_to_input_indices(
            &output_idx, axes, tensor.shape()
        );
        
        // Stream values without storing them all in memory
        let mut acc = WelfordAccumulator::new();
        for input_idx in &input_indices {
            let value = tensor[&input_idx[..]];
            acc.update(value);
        }
        
        let (_, std_val) = acc.finalize();
        
        if output_idx.is_empty() {
            result[[0]] = std_val; // Scalar case
        } else {
            result[&output_idx[..]] = std_val;
        }
    }
    
    Ok(result)
}
```

### A.7 Testing Utilities

```rust
#[cfg(test)]
mod test_helpers {
    use super::*;
    
    /// Creates test tensor with known pattern for validation
    fn create_test_tensor_3d() -> Array3<f32> {
        Array3::from_shape_fn((2, 2, 3), |(i, j, k)| {
            (i * 6 + j * 3 + k + 1) as f32
        })
    }
    
    /// Validates that gradients sum to zero (as expected for mean-centered operations)
    fn validate_gradient_sum(gradients: &Array) -> bool {
        let sum: f32 = gradients.iter().sum();
        (sum.abs() < 1e-6) // Allow small floating-point errors
    }
    
    /// Computes numerical gradient for testing analytical implementation
    fn numerical_gradient_std(
        tensor_data: &[f32],
        shape: &[usize],
        axes: &[usize],
        element_idx: usize,
        h: f32,
    ) -> f32 {
        let mut data_plus = tensor_data.to_vec();
        let mut data_minus = tensor_data.to_vec();
        
        data_plus[element_idx] += h;
        data_minus[element_idx] -= h;
        
        let tensor_plus = Array::from_shape_vec(shape, data_plus).unwrap();
        let tensor_minus = Array::from_shape_vec(shape, data_minus).unwrap();
        
        let std_plus = compute_std_memory_efficient(&tensor_plus, axes).unwrap();
        let std_minus = compute_std_memory_efficient(&tensor_minus, axes).unwrap();
        
        let sum_plus: f32 = std_plus.iter().sum();
        let sum_minus: f32 = std_minus.iter().sum();
        
        (sum_plus - sum_minus) / (2.0 * h)
    }
    
    #[test]
    fn test_gradient_correctness() {
        let tensor = create_test_tensor_3d();
        let axes = vec![0, 2];
        
        // Compute analytical gradients
        let group_map = StdGroupMap::new(tensor.shape(), &axes);
        let gradients = compute_std_gradients(&Array::ones((2,)), &tensor.into_dyn(), &group_map);
        
        // Verify with numerical gradients
        let tensor_flat: Vec<f32> = tensor.iter().cloned().collect();
        for (i, &analytical) in gradients.iter().enumerate() {
            let numerical = numerical_gradient_std(
                &tensor_flat, tensor.shape(), &axes, i, 1e-4
            );
            assert!((analytical - numerical).abs() < 1e-3, 
                "Gradient mismatch at index {}: analytical={}, numerical={}", 
                i, analytical, numerical);
        }
    }
}
```

---

*"The devil is in the details, but the angels are in the implementation."* - Software Engineering Proverb 👨‍💻