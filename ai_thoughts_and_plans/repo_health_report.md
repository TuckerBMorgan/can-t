# Repository Health Report

_Date: 2025-02-14_

## Scope
High-level review of the repo outside of `examples/`, focusing on documentation, process, tooling hygiene, and other non-feature health signals.

## Strengths
- Clear top-level README plus `code_style_guide.md` and `contribution_guide.md` convey core intent: PyTorch-like ergonomics, emphasis on doc comments, and formatting expectations.
- Project structure mirrors the architecture described in `AGENTS.md`: operations live in `src/central`, backend crates stay under `crates/`, and exploratory ideas are captured in `ai_thoughts_and_plans/` & `plans_and_thoughts/`.
- CI exists (`.github/workflows/`) and at least exercises one meaningful MNIST training path, ensuring heavy model artifacts can still run on PRs.

## Key Findings

### Build & Test Hygiene
- Running targeted tests such as `cargo test gather_op` or `cargo test sigmoid_op` triggers ~95 warnings across many modules (e.g., `src/central/cross_entropy_op.rs:2`, `src/central/index.rs:2`, `src/central/topk_op.rs:3`). These are mostly `unused import` or `dead_code` warnings, which drown out actionable failures.
- The Cargo manifest uses an edition of `2024`. Stable Rust will not support the 2024 edition until the official release; this forces contributors onto nightly, conflicting with the “hackable learning library” positioning.
- `[target.'cfg(target_os = "windows")'.workspace.members]` causes the warning `unused manifest key` because Cargo does not support `workspace` inside target-specific tables. This happens on every build/test, adding more noise.
- GPU support is always initialised on macOS through `crates/cant_metal/src/lib.rs`. On machines without a Metal-capable GPU (or in CI), simple unit tests panic with `No Metal device found`, poisoning the `lazy_static` singletons and crashing subsequent tests. There is no documented CPU-only switch besides avoiding macOS entirely.
- CI only runs `cargo test test_sequential_mnist_training -- --nocapture`. There is no formatting or lint gate; clippy, fmt, doc tests, and the rest of the unit suite are not exercised. There are also two copies of the same workflow file (`.github/workflow/mnist_test.yaml` and `.github/workflows/rust.yml`).
- Because tensors share a global `Equation` singleton (`src/central/mod.rs:74`), tests must call `zero_all_grads()` in a strict order to avoid state bleed. When a test panics, the shared state is reused silently (thanks to `lock().unwrap_or_else(|e| e.into_inner())`), masking the original problem and risking flaky cross-test interactions.

### Documentation & Comment Quality
- The README explains architecture and workflow but still instructs contributors to run `rustfmt src/**/*.rs` instead of the workspace-safe `cargo fmt --all`. Coverage, linting, and backend setup instructions are otherwise sparse.
- Both `code_style_guide.md` and `contribution_guide.md` are short checklists (two and five bullet points respectively). They do not address how to add operations, when to touch backend crates, or how to keep docs/tests synchronized.
- Contribution guidelines require “function-level doc comments,” yet the majority of operations (e.g., `src/central/gather_op.rs`, `src/central/sigmoid_op.rs`, `src/central/topk_op.rs`) either lack doc comments entirely or contain single-sentence placeholders. Consistency is low.
- Many TODOs exist without owners or tracking issues (`rg -n TODO` surfaces at least eight in `src/central/` alone). Examples include `TODO: update this comment` in `matmul_op.rs:71` and “TODO: Add dynamic index assert” in `shape.rs:166`.

### Dependency & Configuration Hygiene
- Several dependencies are left as floating `*` (e.g., `tokenizers = "*"`, `regex = "*"`). This makes builds non-reproducible and risky for supply-chain stability.
- GPU backend libraries (`cant_metal`, prospective `cant_cuda`) are always enabled based on `target_os` rather than Cargo features. There is no easy documented way to force CPU-only mode or opt out when the hardware/software stack is absent.
- `DebuggingOptions.NaNCheck` (src/central/debugging.rs) deliberately violates Rust naming conventions, producing a warning on each build. Either rename the field or suppress the lint intentionally.

### Commenting, Formatting, and General Organization
- Formatting generally follows rustfmt, but there are inconsistent doc comments vs. inline comments. Some modules include large blocks of narrative comments (e.g., `gather_op.rs`), while others expose complex logic with no explanation (`equation.rs`, `operation.rs`).
- `Operation::SmoothL1Loss` in `src/central/equation.rs` still calls `panic!()` during backprop, meaning invoking this op leaks runtime panics rather than graceful errors.
- The repo includes both `.DS_Store` and `.venv` directories at the root, signaling that some local artifacts may leak into commits if not carefully ignored.

## Recommendations
1. **Tame the warning flood**: run `cargo fix --all-targets --workspace` (or targeted modules) and re-enable warnings-as-errors in CI once the backlog is cleared. This makes true regressions more visible.
2. **Stabilise the toolchain**: revert the edition to `2021` (for now) and remove the unsupported `[target.'cfg(...)'.workspace]` stanza. Replace it with a documented `cargo` alias or instructions for Windows-specific crates.
3. **Make GPU backends opt-in**: guard `cant_metal` initialisation behind a feature flag or runtime probe so CPU-only contributors/CI do not panic. Document how to choose a backend.
4. **Expand CI coverage**: add `cargo fmt -- --check`, `cargo clippy --workspace --all-targets`, and the general `cargo test` matrix. Deduplicate the two workflow files.
5. **Enforce the doc-comment rule**: add a lint (e.g., `#![deny(missing_docs)]` for public modules) or at least enforce it in reviews. Provide examples in `code_style_guide.md` to set expectations.
6. **Track TODOs**: convert lingering TODOs into GitHub issues or a `plans_and_thoughts` entry so they do not disappear. Reference issue IDs directly in the code comments.
7. **Document the singleton contract**: describe the lifecycle of `Equation` in README or a developer doc, including when to call `zero_all_grads` and how to avoid state leaks. Consider a test helper that resets the singleton between cases.
8. **Pin dependencies**: replace `*` versions with explicit semver ranges and audit for outdated crates.
9. **Update onboarding docs**: extend `contribution_guide.md` with the real workflow (fmt, clippy, tests, coverage). Link to `AGENTS.md` for architecture, and mention any large model resource requirements.

Addressing the above will reduce contributor friction, make CI signals more trustworthy, and keep this “hackable” learning project from accumulating avoidable entropy.
