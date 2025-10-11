# Repository Guidelines

## Project Structure & Module Organization
Core library lives in `src/`, with central ops in `src/central`, neural network helpers in `src/nn`, utilities in `src/utils`, and entry points aggregated by `src/lib.rs`. Backend crates (`crates/cant_cpu`, `crates/cant_metal`) implement execution backends. High-level model artifacts and GGUF fixtures sit under `models/tests`, while runnable demos live in `examples/`. Keep large assets in `data/` and document experiments in `plans_and_thoughts/`.

## Build, Test, and Development Commands
- `cargo build` – compile the full workspace, including backend crates.
- `cargo test` – run unit and integration tests; use `cargo test -- --ignored` for heavier suites.
- `cargo fmt --all` – apply the required formatting pass before committing.
- `cargo clippy --all-targets --all-features` – lint for common mistakes.
- `cargo llvm-cov --html` then `open target/llvm-cov/html/index.html` to review coverage.
- `cargo run --example select_demo` to validate example wiring.

## Coding Style & Naming Conventions
Prefer idiomatic Rust: four-space indentation, `snake_case` for files/modules, `CamelCase` for types, and `SCREAMING_SNAKE_CASE` for constants. Follow `code_style_guide.md` to mirror PyTorch’s API feel; replicate argument names even when Python uses defaults. Contribution guidelines require function-level doc comments summarizing behavior. Format every change with `cargo fmt` and address clippy feedback before review.

## Testing Guidelines
Every new operation needs unit coverage in `src/lib.rs` or an adjacent module test. Use GGUF fixtures in `models/tests` to prove model-level behavior and add regression inputs near similar examples. Name test functions with the pattern `<module>_<behavior>`. Run `cargo test` locally and attach coverage deltas when behavior changes; prefer keeping coverage steady via `cargo llvm-cov`.

## Commit & Pull Request Guidelines
Follow the existing history: short, descriptive commit subjects in the imperative/Title style (e.g., “Add masked fill gradients”). Squash noisy work-in-progress commits before opening a PR. Each PR must include: a summary of intent, links to related issues or experiments, confirmation that `cargo test` and `cargo fmt` passed, and screenshots or logs when UI/CLI output changes. Flag any required model artifacts and document how to regenerate them.
