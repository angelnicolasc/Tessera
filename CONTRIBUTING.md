# Contributing to Tessera

Thanks for considering a contribution. Tessera is a foundations-stage project; the bar for
adding code is high (the README's Sprint 0 status section lists what's deferred — most
deferred items are tracked as labelled issues).

## Quick start

```bash
# Toolchain
rustup show                              # uses rust-toolchain.toml (1.82.0)
pip install -U maturin pre-commit ruff pyright

# Hooks
pre-commit install

# Build + test
just build && just test
```

## Conventions

* **Conventional commits** (`feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `chore:`,
  `bench:`, `perf:`, `build:`, `ci:`). `release-please` consumes them to produce changelogs
  and version bumps automatically.
* **Rust**: `cargo fmt && cargo clippy --workspace --all-targets -- -D warnings` must pass.
* **Python**: `ruff check && ruff format --check && pyright` must pass.
* **No nightly Rust features.** The toolchain is pinned to stable.
* **Public APIs need a doc comment.** `missing_docs` is `warn` at workspace level.
* **No `unwrap` / `expect` on the public hot path.** Errors propagate via `Result`.

## Adding a new ADR

1. Copy an existing ADR from `docs/src/adr/` and increment the number.
2. Update `docs/src/SUMMARY.md`.
3. Link the ADR from the relevant source files / `ARCHITECTURE.md` entries.
4. Use the format **Status / Context / Decision / Consequences**, keep it under one screen.

## Adding a new `IndexBackend`

1. Implement `tessera_index::IndexBackend` in a new module.
2. Add unit + recall tests next to `usearch_index.rs`.
3. Update `docs/src/adr/0005-index-backend-trait.md` if the addition reveals a constraint we
   didn't anticipate.

## Adding a new `KernelBackend`

1. Add the variant to `python/tessera/kernel_dispatch.py::KernelBackend`.
2. Implement the wrapper under `python/tessera/backends/`.
3. Update `select_backend_kind` to choose it where appropriate.
4. Document the GPU requirements in `docs/src/kernel_dispatch.md`.

## License of contributions

By submitting a contribution you agree it is licensed under both MIT and Apache-2.0, as the
project itself is. See `NOTICE`.
