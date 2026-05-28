# Multi-stage Tessera image.
#
# Stage 1 (builder): compiles the Rust extension with maturin inside a slim Rust image.
# Stage 2 (runtime): installs only the wheel + runtime Python deps; no build toolchain.
#
# GPU support (CUDA): use NVIDIA base images downstream; Tessera's Python layer is
# device-agnostic — only `tessera._native` calls into CUDA-specific paths when built
# with `--features cuda`. The CPU-only wheel produced here is safe on any machine.

# ──────────── Stage 1: builder ───────────────────────────────────────────────
FROM rust:1.82-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-dev \
        python3-pip \
        pkg-config \
        libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . /src

# maturin builds the PyO3 extension and produces a wheel in /src/dist/.
RUN pip install maturin --break-system-packages
RUN maturin build --release -m crates/tessera-py/Cargo.toml --out /dist

# ──────────── Stage 2: runtime ───────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="Tessera"
LABEL org.opencontainers.image.description="MLA-aware KV block manager for multi-agent inference"
LABEL org.opencontainers.image.licenses="MIT OR Apache-2.0"
LABEL org.opencontainers.image.source="https://github.com/angelnicolasc/tessera"

COPY --from=builder /dist/*.whl /tmp/

RUN pip install --no-cache-dir \
        /tmp/tessera-*.whl \
        pydantic>=2.7 \
        numpy>=1.26 \
        xxhash>=3.4 \
        usearch>=2.12 \
    && rm /tmp/tessera-*.whl

# Smoke-test: verify the native module loads cleanly.
RUN python -c "from tessera import _native; print('tessera', _native.__version__, '— OK')"

ENTRYPOINT ["python"]
