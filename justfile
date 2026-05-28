set windows-shell := ["pwsh.exe", "-NoLogo", "-Command"]

# Default: list available recipes.
default:
    @just --list

# --- Build ---
build:
    cargo build --workspace --all-targets

build-release:
    cargo build --workspace --release

# --- Lint ---
fmt:
    cargo fmt --all

fmt-check:
    cargo fmt --all -- --check

clippy:
    cargo clippy --workspace --all-targets -- -D warnings

lint-rust: fmt-check clippy

lint-python:
    ruff check python tests benchmarks
    ruff format --check python tests benchmarks
    pyright python tests

lint: lint-rust lint-python

# --- Test ---
test-rust:
    cargo test --workspace

test-rust-all:
    cargo test --workspace --all-features

test-python:
    pytest tests -v

test: test-rust test-python

# --- Bench ---
bench:
    cargo bench --workspace

bench-compile:
    cargo bench --workspace --no-run

# --- Docs ---
docs:
    mdbook build docs

docs-serve:
    mdbook serve docs --open

# --- Security ---
audit:
    cargo audit
    cargo deny check
    pip-audit -r pyproject.toml || true

# --- Python wheel ---
wheel:
    maturin build --release -m crates/tessera-py/Cargo.toml

develop:
    maturin develop -m crates/tessera-py/Cargo.toml

# --- Aggregate CI gate ---
ci: lint-rust test-rust bench-compile docs
    @echo "CI gate passed"

# --- Clean ---
clean:
    cargo clean
    rm -rf docs/book target python/tessera/_native*.so python/tessera/_native*.pyd
