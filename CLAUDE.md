# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Cant is a machine learning library written in Rust that aims to provide PyTorch-like functionality. It's designed as a learning project to understand ML concepts without heavy AI assistance. The project implements automatic differentiation, tensor operations, and supports both CPU and Metal (macOS GPU) acceleration.

## Build and Test Commands

```bash
# Run tests
cargo test

# Build the project
cargo build

# Generate code coverage
cargo llvm-cov --html
open target/llvm-cov/html/index.html

# Format code (required before commits)
rustfmt src/**/*.rs crates/**/*.rs

# Generate documentation
cargo doc --no-deps --document-private-items --open
```

## Architecture

### Core Components

- **Central Module** (`src/central/`): Core tensor operations and automatic differentiation
  - `tensor.rs`: Main Tensor struct and operations
  - `equation.rs`: Global computation graph manager
  - `operation.rs`: Enum defining all supported operations
  - `shape.rs`: Shape handling and broadcasting logic
  - Individual operation files (`add_op.rs`, `matmul_op.rs`, etc.)

- **Acceleration Crates**:
  - `cant_cpu`: CPU-based tensor operations using ndarray
  - `cant_metal`: Metal shader-based GPU acceleration for macOS

- **Utilities** (`src/utils/`): GGUF file support and timing utilities

### Key Design Patterns

1. **Global Equation System**: Uses a lazy_static Mutex-protected singleton (`Equation`) to manage the computation graph and tensor data storage
2. **Tensor ID System**: Each tensor has a unique TensorID for lookups in the global equation
3. **Operation Tracking**: All operations store their inputs via the Operation enum for backward pass computation
4. **Dual Backend Support**: Operations can be executed on CPU (ndarray) or Metal GPU

### Testing Strategy

- Unit tests in `src/lib.rs` with comprehensive micrograd compatibility tests
- Model tests using GGUF files in `models/tests/` directory
- Test data includes various operation types (add, matmul, pow, reshape, sum)

## Code Style Requirements

1. Function-level comments are required for all functions
2. Run `rustfmt` before committing
3. Add tests for any new features
4. Follow PyTorch-like API design where possible
5. Favor readability over performance optimizations
6. Explicit parameter naming (avoid default arguments pattern from Python)

## Development Notes

- Uses Rust 2024 edition
- No AI coding tools were used in development (intentional learning constraint)
- Project focuses on educational value over production readiness
- Metal acceleration only available on macOS