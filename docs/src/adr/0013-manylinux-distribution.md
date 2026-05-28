# ADR-0013 — manylinux_2_28 Wheels and ghcr.io Distribution

**Status**: Accepted  
**Sprint**: 2 (WS5)  
**Closes**: TD-019  
**Update (Sprint 5, 2026-05-28)**: x86_64 manylinux_2_28 wheels (cp311 + cp312) ship per
push, as does the ghcr.io image. **ARM64 wheels and PyPI publish are now scheduled for
Sprint 6+** (was "Sprint 3" in this ADR's original deferred list); PyPI requires
governance + trusted-publishers setup that is being staged alongside the v1.0 release
preparation.

---

## Context

Sprint 1 CI produced a wheel via `ubuntu-latest` but without the manylinux ABI tag.
A wheel without the `manylinux` platform tag cannot be installed on arbitrary Linux distributions:
`pip` rejects it on systems whose glibc version predates the runner's system glibc.

Two distribution artifacts are needed for production adoption:

1. **manylinux wheel** — installable on any Linux ≥ glibc 2.28 without compiling Rust locally.
2. **Docker image** — self-contained runtime; relevant for cloud GPU pods where Tessera is
   co-deployed with vLLM inside a container.

## Decision

### Wheels — `cibuildwheel` with `manylinux_2_28`

Use `pypa/cibuildwheel@v2.21` to build inside the official `manylinux_2_28` container.
`manylinux_2_28` (glibc ≥ 2.28) is chosen over `manylinux2014` (glibc ≥ 2.17) because:

- All major cloud GPU images in 2026 ship glibc ≥ 2.28.
- `manylinux_2_28` supports more modern compiler toolchains (gcc 12+), which produce smaller
  and faster Rust artifacts via better LLVM IR optimization.
- Rust 1.82's MSRV guarantee requires glibc ≥ 2.17 anyway, so the wider `manylinux2014`
  tag would not add real portability.

We build `cp311-manylinux_x86_64` and `cp312-manylinux_x86_64`. ARM64 (`aarch64`) is deferred
to Sprint 3 pending cloud GPU availability of ARM Hopper instances.

### Container image — ghcr.io multi-stage Dockerfile

A two-stage `Dockerfile`:

1. **`builder`** stage: `rust:1.82-slim` + `maturin build --release`. This stage is never
   shipped — it is discarded after the wheel is extracted.
2. **`runtime`** stage: `python:3.11-slim` + wheel install. Only the wheel and its Python
   runtime deps are present; no Rust toolchain, no build headers, no source.

Published to `ghcr.io/<owner>/tessera` on every `main` push touching `crates/`, `python/`,
or `Dockerfile`. Tags: `sprint2`, `latest`, and `sha-<short>`.

### PyPI publication

Explicitly deferred. PyPI governance requires:
- A stable version tag (we are at `0.x.y-sprint2` — pre-release).
- Review of the `tessera` name (may be claimed on PyPI).
- Dual-license compliance review of all transitive Rust dependencies.

The wheel artifact is uploaded to GitHub Actions as `manylinux-wheels-{cp311,cp312}` and
retained for 30 days, sufficient for staged rollouts to cloud GPU pods.

## Consequences

**Positive**:
- Operators can `pip install tessera-*.whl` inside any Ubuntu 22.04+/RHEL 8+ container
  without `rustup` or `maturin`.
- The Docker image can be used as a direct `FROM` base or as an inspiration for production
  serving containers.
- GitHub Actions cache (`type=gha`) keeps Docker layer rebuilds fast on subsequent pushes.

**Negative**:
- `cibuildwheel` adds ~8 minutes to the `main` push pipeline (first-time build; subsequent
  builds use the cargo registry cache).
- ARM64 wheels are missing until Sprint 3. Operators on ARM GPU pods must build from source.
- PyPI publish is not included; operators must retrieve the wheel from the GHA artifact store.

## Relation to other ADRs

- ADR-0003: `DeviceBackend` trait — the CPU mock backend is what ships in the wheel; CUDA
  features are compile-time gated and excluded from the manylinux build.
- ADR-0007: FP8 calibration — calibrated `fp8_scales_path` is a user-supplied file; the
  wheel contains no model-specific data.
