# Custom Operation Integration Strategies

Cant needs a way for downstream crates to contribute bespoke operations that the core runtime can recognize at compile time. Below are three distinct approaches we can pursue, each compatible with the current workspace layout and coding guidelines.

## 1. Generic Operation Set Trait
- Introduce a new `OperationSet` trait in `src/central` that exposes methods for registering operations and resolving kernel implementations.
- Parameterize top-level execution entry points (e.g., `GraphExecutor`, `CompiledGraph`) over an `OpSet: OperationSet`, defaulting to Cant’s built-ins via a `DefaultOpSet` type alias in `src/lib.rs`.
- Downstream users provide their own `struct MyOpSet;` that implements `OperationSet`, with associated constants (signatures, tensor arity, gradient hooks) defined at compile time.
- Provide helper macros in `src/utils/macros.rs` to declare operations succinctly and ensure doc comment coverage and `cargo fmt` compliance.
- Pros: zero runtime overhead; integrates cleanly with Rust’s type system; enforces compile-time verification of required hooks. Cons: requires touching many generic boundaries; could make type signatures verbose and raise trait object coherence concerns.

## 2. Static Registry Powered by `inventory`
- Leverage the `inventory` crate (or `linkme` as a `no_std` alternative) to build a static registry of operations during compilation.
- Define a `#[derive(Operation)]` proc-macro in `src/nn/derive_operation.rs` (new module) that expands to an `inventory::submit!` call and generates glue traits for forward/backward passes.
- Core execution paths consult the registry to resolve op metadata, defaulting to built-ins if no override is submitted.
- Downstream crates enable a Cargo feature (e.g., `custom_ops`) and annotate their structs with `#[derive(Operation)]` to auto-register at compile time.
- Pros: minimal changes to existing call sites; flexible opt-in per crate; supports modular add-ons distributed as separate crates. Cons: relies on linker section behavior, which can be brittle on some targets; proc-macro and inventory usage add compile-time complexity.

## 3. Build-Script Code Generation Pipeline
- Add an optional build step (`build.rs`) that reads declarative operation specs (TOML/JSON or Rust stubs) from `models/tests` or a user-specified path.
- The build script emits Rust modules under `src/generated/operations.rs`, defining strongly typed structs implementing an `Operation` trait and wiring them into the dispatcher.
- Users integrate by providing spec files or calling helper functions inside their own `build.rs` to append new operations, ensuring they compile into the binary.
- Provide templates in `examples/` demonstrating how to feed custom layer definitions and regenerate the code via `cargo build`.
- Pros: keeps the runtime API simple; allows richer metadata (e.g., kernel parameters) without hand-writing Rust; can share artifacts with other tooling. Cons: introduces build-time dependency on file formats; less ergonomic for purely Rust consumers; risk of stale generated code if rebuild triggers are missed.

Each path can be iterated on independently; we can prototype all three and pick the approach that best balances ergonomics, compile-time guarantees, and maintenance overhead.
