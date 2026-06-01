//! Device backend abstraction. The block manager is parameterised over this trait so the same
//! code path runs against:
//!
//! * [`CpuMockBackend`] — a host-memory implementation used by default in tests and on
//!   machines without a CUDA-capable GPU. Deterministic; no GPU runtime required.
//! * `CudaBackend` (feature `cuda`) — a real CUDA implementation backed by `cudarc`.
//!
//! The abstraction is intentionally narrow: only the operations the block manager actually
//! needs are exposed. See `docs/src/adr/0003-trait-device-backend.md`.

use std::fmt::Debug;

mod cpu_mock;
#[cfg(feature = "cuda")]
pub mod cuda;
pub mod disk;

pub use cpu_mock::CpuMockBackend;
#[cfg(feature = "cuda")]
pub use cuda::CudaBackend;
pub use disk::{DiskBackend, SwaCachingStrategy};

/// Opaque device pointer wrapped with its allocation length. The block manager treats this as
/// a handle; backends interpret it however they like (host pointer, CUDA device pointer, etc).
#[derive(Debug, Clone, Copy)]
pub struct DevicePtr {
    /// Raw address; backend-specific meaning.
    pub raw: usize,
    /// Length of the region in bytes (the backend is responsible for honouring this on
    /// memcpy and readback).
    pub len: u64,
}

impl DevicePtr {
    /// Compute a pointer offset within an allocation. Used by the block manager to slice a
    /// single contiguous backing allocation into per-block subregions.
    pub fn offset(self, by: u64) -> Self {
        debug_assert!(by <= self.len, "DevicePtr offset out of bounds");
        Self {
            raw: self.raw + by as usize,
            len: self.len - by,
        }
    }
}

/// Region kind for diagnostics and metric labels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegionKind {
    /// The primary KV region (`c_kv` for MLA).
    Primary,
    /// Position-dependent secondary region (`k_rope` for MLA).
    Rope,
    /// Per-layer FP8 scale factors.
    Fp8Scales,
}

impl RegionKind {
    /// Short name for logs.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Primary => "primary",
            Self::Rope => "rope",
            Self::Fp8Scales => "fp8_scales",
        }
    }
}

/// Operations a device backend must implement.
///
/// Implementations should be cheap to clone-handle (e.g. wrap an `Arc<Inner>`); the block
/// manager keeps one backend per manager and clones it into worker threads if needed.
pub trait DeviceBackend: Debug + Send + Sync + 'static {
    /// Allocate a contiguous region of `bytes` bytes. Returns a handle to it.
    fn alloc_region(&self, bytes: u64, kind: RegionKind) -> anyhow::Result<DevicePtr>;

    /// Copy `bytes` from `src` to `dst`. Both must point into regions previously returned by
    /// [`alloc_region`].
    fn memcpy(&self, src: DevicePtr, dst: DevicePtr, bytes: u64) -> anyhow::Result<()>;

    /// Read `len` bytes starting at `ptr` into host memory. Used by the block manager to
    /// compute content hashes; on real devices this is a `dtoh` copy. ADR-0008 tracks moving
    /// hashing onto the device.
    fn read_bytes(&self, ptr: DevicePtr, len: usize) -> anyhow::Result<Vec<u8>>;

    /// Overwrite `len` bytes starting at `ptr` with `byte`. Used by tests to plant
    /// deterministic content patterns. Production code never calls this on the hot path.
    fn fill_pattern(&self, ptr: DevicePtr, byte: u8, len: usize) -> anyhow::Result<()>;

    /// Write `bytes` to the device region starting at `ptr`. Used by
    /// [`crate::block_manager::TesseraBlockManager::import_payload`] when receiving a block
    /// from a peer rank (PD-disaggregation push path). On real devices this is `htod`.
    fn write_bytes(&self, ptr: DevicePtr, bytes: &[u8]) -> anyhow::Result<()>;

    /// Name of this backend for diagnostics ("cpu_mock", "cuda", etc).
    fn name(&self) -> &'static str;
}
