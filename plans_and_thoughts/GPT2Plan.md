# GPT-2 Implementation Plan for Cant ML Library

## Executive Summary

This document outlines a comprehensive plan to extend the Cant ML library from its current state (basic tensor operations with autodiff) to a full-featured neural network library capable of loading and running GPT-2 models. The plan is structured in 4 phases over approximately 8-10 weeks.

## Current State Assessment

### ✅ **Strengths (Already Implemented)**
- Robust automatic differentiation system with global computation graph
- Core tensor operations: `add`, `mul`, `matmul`, `sum`, `mean`, `tanh`, `softmax`
- GPU acceleration via Metal (macOS) with CPU fallback
- Shape broadcasting and tensor manipulation
- GGUF file format support for model loading
- Cross-entropy loss and basic optimization
- Comprehensive test suite with gradient verification

### ❌ **Critical Gaps**
- Missing transformer-specific operations (GELU, LayerNorm, Transpose)
- No high-level layer abstractions (Linear, Embedding, Attention)
- No model management framework
- Missing optimizers beyond basic SGD
- No proper neural network architecture patterns

---

## Phase 1: Core Operations & Infrastructure (Weeks 1-2)

### Milestone 1.1: Essential Mathematical Operations

**Priority: CRITICAL** | **Timeline: Week 1**

#### New Operations to Implement

1. **GELU Activation Function**
   ```rust
   // Add to Operation enum
   Operation::Gelu(TensorID)
   
   // Implementation approaches:
   // Option A: Exact GELU = 0.5 * x * (1 + erf(x / sqrt(2)))
   // Option B: Approximation = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
   ```

2. **Layer Normalization**
   ```rust
   Operation::LayerNorm(source: TensorID, weight: TensorID, bias: TensorID, axis: isize, eps: f32)
   
   // LayerNorm(x) = (x - mean) / sqrt(var + eps) * weight + bias
   // Critical for transformer stability
   ```

3. **Tensor Manipulation Operations**
   ```rust
   Operation::Transpose(source: TensorID, axes: Vec<isize>)
   Operation::Permute(source: TensorID, dims: Vec<usize>)
   Operation::Concat(sources: Vec<TensorID>, axis: isize)
   Operation::Split(source: TensorID, sizes: Vec<usize>, axis: isize)
   Operation::Sqrt(source: TensorID)
   Operation::Rsqrt(source: TensorID)  // More efficient 1/sqrt(x)
   ```

#### Implementation Details

**File Structure:**
```
src/central/
├── gelu_op.rs          # GELU activation implementation
├── layer_norm_op.rs    # Layer normalization
├── transpose_op.rs     # Tensor transpose/permute
├── concat_op.rs        # Concatenation operations
├── sqrt_op.rs          # Square root operations
└── mod.rs              # Update exports
```

**Testing Requirements:**
- Gradient verification for all new operations
- Numerical stability tests
- Performance benchmarks vs PyTorch
- Edge case handling (NaN, infinity, zero gradients)

### Milestone 1.2: Advanced Tensor Operations

**Priority: HIGH** | **Timeline: Week 2**

#### Additional Operations

1. **Advanced Activations**
   ```rust
   Operation::Relu(TensorID)
   Operation::Sigmoid(TensorID)
   Operation::Silu(TensorID)  // Swish activation
   ```

2. **Utility Operations**
   ```rust
   Operation::Clamp(source: TensorID, min: f32, max: f32)
   Operation::Dropout(source: TensorID, p: f32, training: bool)
   Operation::Masked(source: TensorID, mask: TensorID, value: f32)
   ```

3. **Efficient Batched Operations**
   ```rust
   Operation::BatchMatmul(sources: Vec<TensorID>)  // Batched matrix multiplication
   Operation::Einsum(sources: Vec<TensorID>, equation: String)  // Einstein summation
   ```

---

## Phase 2: Neural Network Framework (Weeks 3-4)

### Milestone 2.1: Layer Abstractions

**Priority: CRITICAL** | **Timeline: Week 3**

#### Core Layer Framework

Create a new module structure for high-level neural network components:

```
src/nn/
├── mod.rs              # Neural network module exports
├── linear.rs           # Linear/Dense layer
├── embedding.rs        # Embedding layer
├── layer_norm.rs       # LayerNorm layer wrapper
├── activation.rs       # Activation function layers
├── attention.rs        # Attention mechanisms
└── transformer.rs      # Transformer building blocks
```

#### Layer Interface Design

```rust
// Base trait for all layers
pub trait Layer {
    fn forward(&self, input: Tensor) -> Tensor;
    fn parameters(&self) -> Vec<&Tensor>;
    fn parameters_mut(&mut self) -> Vec<&mut Tensor>;
    fn train(&mut self);
    fn eval(&mut self);
}

// Linear layer implementation
pub struct Linear {
    weight: Tensor,
    bias: Option<Tensor>,
    in_features: usize,
    out_features: usize,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let weight = Tensor::xavier_normal(Shape::new(vec![in_features, out_features]));
        let bias = Some(Tensor::zeros(Shape::new(vec![out_features])));
        // Set requires_grad for both
        Self { weight, bias, in_features, out_features }
    }
}

impl Layer for Linear {
    fn forward(&self, input: Tensor) -> Tensor {
        let output = input << self.weight;
        match &self.bias {
            Some(bias) => output + bias,
            None => output,
        }
    }
}
```

#### Essential Layers

1. **Linear Layer** - Fully connected layer with weight initialization
2. **Embedding Layer** - Token/position embedding (wraps existing `select`)
3. **LayerNorm Layer** - Wrapper around LayerNorm operation
4. **Activation Layers** - GELU, ReLU, etc. as layers

### Milestone 2.2: Model Architecture Framework

**Priority: HIGH** | **Timeline: Week 4**

#### Model Management System

```rust
// Base model trait
pub trait Model {
    fn forward(&self, input: Tensor) -> Tensor;
    fn parameters(&self) -> Vec<&Tensor>;
    fn parameters_mut(&mut self) -> Vec<&mut Tensor>;
    fn train(&mut self);
    fn eval(&mut self);
    fn save(&self, path: &str) -> Result<(), Box<dyn std::error::Error>>;
    fn load(path: &str) -> Result<Self, Box<dyn std::error::Error>> where Self: Sized;
}

// Sequential model container
pub struct Sequential {
    layers: Vec<Box<dyn Layer>>,
    training: bool,
}

impl Sequential {
    pub fn new() -> Self { Self { layers: vec![], training: true } }
    pub fn add_layer(mut self, layer: Box<dyn Layer>) -> Self {
        self.layers.push(layer);
        self
    }
}
```

#### Parameter Management

```rust
// Parameter optimizer trait
pub trait Optimizer {
    fn step(&mut self, parameters: &mut [&mut Tensor]);
    fn zero_grad(&mut self, parameters: &mut [&mut Tensor]);
}

// Adam optimizer implementation
pub struct Adam {
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    step_count: usize,
    // Store momentum and velocity for each parameter
    momentum: HashMap<TensorID, Tensor>,
    velocity: HashMap<TensorID, Tensor>,
}
```

---

## Phase 3: Attention Mechanisms (Weeks 5-6)

### Milestone 3.1: Attention Operations

**Priority: CRITICAL** | **Timeline: Week 5**

#### Scaled Dot-Product Attention

```rust
pub struct ScaledDotProductAttention {
    scale: f32,
    dropout: f32,
}

impl ScaledDotProductAttention {
    pub fn forward(&self, q: Tensor, k: Tensor, v: Tensor, mask: Option<Tensor>) -> Tensor {
        // scores = Q @ K^T / sqrt(d_k)
        let scores = (q << k.transpose(-2, -1)) / self.scale;
        
        // Apply causal mask if provided
        let scores = match mask {
            Some(mask) => scores.masked_fill(mask, f32::NEG_INFINITY),
            None => scores,
        };
        
        // Apply softmax and dropout
        let weights = scores.softmax(-1);
        let weights = if self.dropout > 0.0 {
            weights.dropout(self.dropout, self.training)
        } else {
            weights
        };
        
        // Apply attention weights to values
        weights << v
    }
}
```

#### Multi-Head Attention

```rust
pub struct MultiHeadAttention {
    num_heads: usize,
    head_dim: usize,
    scale: f32,
    
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    
    attention: ScaledDotProductAttention,
}

impl MultiHeadAttention {
    pub fn new(embed_dim: usize, num_heads: usize) -> Self {
        assert_eq!(embed_dim % num_heads, 0);
        let head_dim = embed_dim / num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();
        
        Self {
            num_heads,
            head_dim,
            scale,
            q_proj: Linear::new(embed_dim, embed_dim),
            k_proj: Linear::new(embed_dim, embed_dim),
            v_proj: Linear::new(embed_dim, embed_dim),
            out_proj: Linear::new(embed_dim, embed_dim),
            attention: ScaledDotProductAttention::new(scale),
        }
    }
    
    pub fn forward(&self, x: Tensor, mask: Option<Tensor>) -> Tensor {
        let (batch_size, seq_len, embed_dim) = x.shape().dims3();
        
        // Project to Q, K, V
        let q = self.q_proj.forward(x);
        let k = self.k_proj.forward(x);
        let v = self.v_proj.forward(x);
        
        // Reshape for multi-head: [batch, seq, heads, head_dim]
        let q = q.reshape(Shape::new(vec![batch_size, seq_len, self.num_heads, self.head_dim]));
        let k = k.reshape(Shape::new(vec![batch_size, seq_len, self.num_heads, self.head_dim]));
        let v = v.reshape(Shape::new(vec![batch_size, seq_len, self.num_heads, self.head_dim]));
        
        // Transpose to [batch, heads, seq, head_dim]
        let q = q.transpose(1, 2);
        let k = k.transpose(1, 2);
        let v = v.transpose(1, 2);
        
        // Apply attention
        let attended = self.attention.forward(q, k, v, mask);
        
        // Transpose back and reshape
        let attended = attended.transpose(1, 2);
        let attended = attended.reshape(Shape::new(vec![batch_size, seq_len, embed_dim]));
        
        // Final projection
        self.out_proj.forward(attended)
    }
}
```

### Milestone 3.2: Causal Masking & Position Encoding

**Priority: HIGH** | **Timeline: Week 6**

#### Causal Attention Mask

```rust
pub fn causal_mask(seq_len: usize) -> Tensor {
    let mut mask_data = vec![0.0; seq_len * seq_len];
    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            mask_data[i * seq_len + j] = 1.0; // Mask future positions
        }
    }
    Tensor::from_vec(mask_data, vec![seq_len, seq_len])
}
```

#### Positional Encoding

```rust
pub fn sinusoidal_position_encoding(seq_len: usize, d_model: usize) -> Tensor {
    let mut pos_encoding = vec![0.0; seq_len * d_model];
    
    for pos in 0..seq_len {
        for i in (0..d_model).step_by(2) {
            let angle = pos as f32 / 10000.0_f32.powf(i as f32 / d_model as f32);
            pos_encoding[pos * d_model + i] = angle.sin();
            if i + 1 < d_model {
                pos_encoding[pos * d_model + i + 1] = angle.cos();
            }
        }
    }
    
    Tensor::from_vec(pos_encoding, vec![seq_len, d_model])
}
```

---

## Phase 4: GPT-2 Implementation (Weeks 7-8)

### Milestone 4.1: Transformer Block

**Priority: CRITICAL** | **Timeline: Week 7**

#### GPT-2 Transformer Block

```rust
pub struct GPT2Block {
    ln_1: LayerNorm,
    attn: MultiHeadAttention,
    ln_2: LayerNorm,
    mlp: Sequential,
}

impl GPT2Block {
    pub fn new(config: &GPT2Config) -> Self {
        let mlp = Sequential::new()
            .add_layer(Box::new(Linear::new(config.n_embd, 4 * config.n_embd)))
            .add_layer(Box::new(GELU::new()))
            .add_layer(Box::new(Linear::new(4 * config.n_embd, config.n_embd)));
        
        Self {
            ln_1: LayerNorm::new(config.n_embd),
            attn: MultiHeadAttention::new(config.n_embd, config.n_head),
            ln_2: LayerNorm::new(config.n_embd),
            mlp,
        }
    }
    
    pub fn forward(&self, x: Tensor, mask: Option<Tensor>) -> Tensor {
        // Pre-LayerNorm architecture
        let attended = self.attn.forward(self.ln_1.forward(x), mask);
        let x = x + attended; // Residual connection
        
        let mlp_out = self.mlp.forward(self.ln_2.forward(x));
        x + mlp_out // Residual connection
    }
}
```

#### GPT-2 Configuration

```rust
#[derive(Debug, Clone)]
pub struct GPT2Config {
    pub vocab_size: usize,
    pub n_embd: usize,      // Embedding dimension (768 for GPT-2 small)
    pub n_layer: usize,     // Number of transformer blocks (12 for GPT-2 small)
    pub n_head: usize,      // Number of attention heads (12 for GPT-2 small)
    pub n_positions: usize, // Maximum sequence length (1024 for GPT-2)
    pub dropout: f32,
    pub layer_norm_epsilon: f32,
}

impl GPT2Config {
    pub fn gpt2_small() -> Self {
        Self {
            vocab_size: 50257,
            n_embd: 768,
            n_layer: 12,
            n_head: 12,
            n_positions: 1024,
            dropout: 0.1,
            layer_norm_epsilon: 1e-5,
        }
    }
    
    pub fn gpt2_medium() -> Self { /* ... */ }
    pub fn gpt2_large() -> Self { /* ... */ }
}
```

### Milestone 4.2: Complete GPT-2 Model

**Priority: CRITICAL** | **Timeline: Week 8**

#### Full GPT-2 Model Implementation

```rust
pub struct GPT2Model {
    config: GPT2Config,
    wte: Embedding,           // Token embeddings
    wpe: Embedding,           // Positional embeddings
    blocks: Vec<GPT2Block>,   // Transformer blocks
    ln_f: LayerNorm,          // Final layer norm
    lm_head: Linear,          // Language modeling head
}

impl GPT2Model {
    pub fn new(config: GPT2Config) -> Self {
        let blocks = (0..config.n_layer)
            .map(|_| GPT2Block::new(&config))
            .collect();
        
        Self {
            wte: Embedding::new(config.vocab_size, config.n_embd),
            wpe: Embedding::new(config.n_positions, config.n_embd),
            blocks,
            ln_f: LayerNorm::new(config.n_embd),
            lm_head: Linear::new(config.n_embd, config.vocab_size),
            config,
        }
    }
    
    pub fn forward(&self, input_ids: Tensor, position_ids: Option<Tensor>) -> Tensor {
        let (batch_size, seq_len) = input_ids.shape().dims2();
        
        // Token embeddings
        let token_embeds = self.wte.forward(input_ids);
        
        // Position embeddings
        let pos_ids = position_ids.unwrap_or_else(|| {
            Tensor::arange(0, seq_len as f32, 1.0).reshape(Shape::new(vec![1, seq_len]))
        });
        let pos_embeds = self.wpe.forward(pos_ids);
        
        // Combined embeddings
        let mut hidden_states = token_embeds + pos_embeds;
        
        // Causal mask
        let mask = causal_mask(seq_len);
        
        // Pass through transformer blocks
        for block in &self.blocks {
            hidden_states = block.forward(hidden_states, Some(mask));
        }
        
        // Final layer norm
        hidden_states = self.ln_f.forward(hidden_states);
        
        // Language modeling head
        self.lm_head.forward(hidden_states)
    }
    
    pub fn generate(&self, input_ids: Tensor, max_length: usize) -> Tensor {
        let mut current_ids = input_ids;
        
        for _ in 0..max_length {
            let logits = self.forward(current_ids, None);
            let next_token_logits = logits.select_last_token();
            let next_token = next_token_logits.argmax(-1);
            current_ids = current_ids.cat(next_token, -1);
        }
        
        current_ids
    }
}
```

---

## Phase 5: Model Loading & Optimization (Weeks 9-10)

### Milestone 5.1: Weight Loading from HuggingFace

**Priority: HIGH** | **Timeline: Week 9**

#### HuggingFace Model Loading

```rust
// Support for loading PyTorch state_dict files
pub struct ModelLoader {
    pub fn load_gpt2_weights(model: &mut GPT2Model, path: &str) -> Result<(), Box<dyn Error>> {
        // Parse PyTorch .bin or .safetensors file
        // Map weight names to model parameters
        // Handle shape differences and layer naming conventions
    }
    
    pub fn load_from_hub(model_name: &str, cache_dir: Option<&str>) -> Result<GPT2Model, Box<dyn Error>> {
        // Download model files from HuggingFace Hub
        // Load config.json and pytorch_model.bin
        // Return configured and loaded model
    }
}
```

### Milestone 5.2: Optimizers & Training Support

**Priority: MEDIUM** | **Timeline: Week 10**

#### Advanced Optimizers

```rust
// Adam optimizer with proper momentum tracking
pub struct Adam { /* ... */ }

// AdamW optimizer with weight decay
pub struct AdamW { /* ... */ }

// Learning rate schedulers
pub trait LRScheduler {
    fn step(&mut self) -> f32;
}

pub struct CosineAnnealingLR { /* ... */ }
pub struct LinearWarmupLR { /* ... */ }
```

#### Training Loop Framework

```rust
pub struct Trainer {
    model: Box<dyn Model>,
    optimizer: Box<dyn Optimizer>,
    scheduler: Option<Box<dyn LRScheduler>>,
    
    pub fn train_step(&mut self, batch: &TrainingBatch) -> f32 {
        // Forward pass
        // Loss computation
        // Backward pass
        // Optimizer step
        // Return loss
    }
    
    pub fn eval_step(&mut self, batch: &TrainingBatch) -> f32 {
        // Evaluation without gradient computation
    }
}
```

---

## Implementation Guidelines

### Code Quality Standards

1. **Documentation**: Every public function must have comprehensive rustdoc
2. **Testing**: All operations require gradient verification tests
3. **Performance**: Benchmark critical paths against PyTorch
4. **Memory Safety**: No unsafe code without thorough justification
5. **Error Handling**: Proper Result types for fallible operations

### Testing Strategy

```rust
// Comprehensive test suite for each component
#[cfg(test)]
mod tests {
    // Numerical gradient verification
    fn test_operation_gradients() { /* ... */ }
    
    // Performance benchmarks
    fn bench_against_pytorch() { /* ... */ }
    
    // Integration tests
    fn test_gpt2_forward_pass() { /* ... */ }
    
    // Model loading tests
    fn test_load_pretrained_weights() { /* ... */ }
}
```

### Performance Optimization

1. **Memory Management**: Efficient tensor allocation and reuse
2. **GPU Utilization**: Maximize Metal acceleration usage
3. **Batching**: Efficient batch processing for training
4. **Caching**: Cache commonly used computations (attention masks, position encodings)

### Error Handling Strategy

```rust
// Comprehensive error types
#[derive(Debug, thiserror::Error)]
pub enum CantError {
    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch { expected: Vec<usize>, actual: Vec<usize> },
    
    #[error("Model loading failed: {reason}")]
    ModelLoadError { reason: String },
    
    #[error("CUDA/Metal operation failed: {details}")]
    BackendError { details: String },
}
```

---

## Success Metrics

### Phase Completion Criteria

1. **Phase 1**: All core operations pass gradient verification tests
2. **Phase 2**: Can build and train simple neural networks (MLP, CNN)
3. **Phase 3**: Multi-head attention works correctly on toy problems
4. **Phase 4**: GPT-2 forward pass produces correct outputs
5. **Phase 5**: Can load and run pretrained GPT-2 models

### Performance Benchmarks

- **Memory Usage**: Within 2x of PyTorch for equivalent operations
- **Speed**: Within 3x of PyTorch for forward pass
- **Accuracy**: Numerical outputs match PyTorch within 1e-5 tolerance

### Final Deliverables

1. **Complete GPT-2 Implementation**: Load and run all GPT-2 variants (small, medium, large)
2. **Text Generation**: Coherent text generation from prompts
3. **Fine-tuning Support**: Ability to fine-tune models on custom datasets
4. **Documentation**: Comprehensive API documentation and examples
5. **Benchmarks**: Performance comparison suite vs PyTorch

---

## Risk Mitigation

### Technical Risks

1. **Memory Limitations**: Large models may exceed available memory
   - *Mitigation*: Implement gradient checkpointing and model parallelism
   
2. **Performance Bottlenecks**: Rust autodiff overhead
   - *Mitigation*: Profile and optimize hot paths, consider JIT compilation
   
3. **Numerical Stability**: Attention operations can be unstable
   - *Mitigation*: Implement proper numerical safeguards and testing

### Timeline Risks

1. **Scope Creep**: Feature requests beyond GPT-2
   - *Mitigation*: Strict scope adherence, defer non-essential features
   
2. **Integration Complexity**: Unforeseen interaction issues
   - *Mitigation*: Incremental development with continuous testing

---

## Conclusion

This plan provides a structured path from the current Cant library state to a full GPT-2-capable neural network framework. The 10-week timeline is aggressive but achievable with focused development effort. The modular approach allows for iterative progress and early validation of components.

The resulting library will be a significant achievement, providing:
- Modern transformer architecture support
- High-performance GPU acceleration
- Clean, safe Rust implementation
- Production-ready neural network framework

Success in this endeavor would position Cant as a viable alternative to PyTorch for certain use cases, particularly where Rust's safety and performance characteristics are valued.