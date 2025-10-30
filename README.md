```
      ██████╗ █████╗ ███╗   ██╗████████╗
     ██╔════╝██╔══██╗████╗  ██║╚══██╔══╝
     ██║     ███████║██╔██╗ ██║   ██║
     ██║     ██╔══██║██║╚██╗██║   ██║
     ╚██████╗██║  ██║██║ ╚████║   ██║
      ╚═════╝╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝
```

# Cant

A machine learning library written in Rust that provides PyTorch-like functionality with automatic differentiation and tensor operations. Designed as a learning project to understand ML concepts from the ground up.

## Features

- **Automatic Differentiation**: Complete backward pass implementation with gradient computation
- **Tensor Operations**: Multi-dimensional array operations with broadcasting support  
- **CPU Acceleration**: High-performance CPU operations using ndarray
- **Metal GPU Acceleration**: macOS GPU acceleration using Metal shaders
- **PyTorch-like API**: Familiar interface for PyTorch users

## Quick Start

### Build and Test

```bash
# Run tests
cargo test

# Build the project
cargo build

# Format code (required before commits)
rustfmt src/**/*.rs crates/**/*.rs
```

### Code Coverage

```bash
# Generate coverage report
cargo llvm-cov --html
open target/llvm-cov/html/index.html
```

## Architecture

### Core Components (`src/central/`)

- **`tensor.rs`** - Main Tensor struct and operations
- **`equation.rs`** - Global computation graph manager
- **`operation.rs`** - Enum defining all supported operations
- **`shape.rs`** - Shape handling and broadcasting logic
- Individual operation files (`add_op.rs`, `matmul_op.rs`, etc.)

### Acceleration Backends

- **`cant_cpu`** - CPU-based tensor operations using ndarray
- **`cant_metal`** - Metal shader-based GPU acceleration for macOS
- **'cant_cuda'** - Cuda based Backend(not has often tested)

### Utilities (`src/utils/`)

- GGUF file support for loading models
- Timing utilities for performance measurement

## Key Design Patterns

1. **Global Equation System**: Uses a singleton to manage the computation graph and tensor storage
2. **Tensor ID System**: Each tensor has a unique ID for efficient lookups
3. **Operation Tracking**: All operations store their inputs for backward pass computation
4. **Tri Backend Support**: Operations can execute on CPU, Metal, or CUDA

## Testing

- **Unit Tests**: Comprehensive micrograd compatibility tests in `src/lib.rs`
- **Model Tests**: GGUF file-based tests in `models/tests/` directory
- **Operation Tests**: Verification of add, matmul, pow, reshape, sum operations

## Platform Support

- **macOS**: Full CPU and Metal GPU acceleration support
- **Other platforms**: CPU-only support via ndarray backend, and CUDA


## What is the point?

If I where to say what this project had other then a place for me to learn is that this is a rather hackable library.
Pytorch, TensorFlow, JAX all have be production code bases and as such can take a lot of effort to make the smallest change

Can-t on the other hand is rather simple. We might not have all of the bells and whistle you find in them, but I feel comfortable
saying that you can understand its inner workings faster then the other two. 

I thought back to the small hackable learning libaries I used when i was at college, where the idea was less about being the best
and instead about being tools for learning. 

## License

MIT License

## Author

Tucker Morgan