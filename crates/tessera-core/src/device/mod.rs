//! Device backend abstraction. The block manager is parameterised over this trait so the same
//! code path runs against:
//!
//! * [`CpuMockBackend`] — a host-memory implementation used by default in tests and on
//!   machines without a CUDA-capable GPU. Deterministic; no GPU runtime required.
//! * `CudaBackend` (feature `cuda`) — a real CUDA implementation backed by `cudarc`.
//! * [`DiskBackend`] — filesystem-backed; persists across process restarts.
//!
//! The abstraction is intentionally narrow: only the operations the block manager actually
//! needs are exposed. See `docs/src/adr/0003-trait-device-backend.md` for the rationale and
//! `docs/src/adr/0025-handle-based-device-ptr.md` for the Sprint 5.1 hardening refactor that
//! replaced raw `usize` pointers with `(region, offset, len)` handles.

use std::fmt::Debug;

mod cpu_mock;
#[cfg(feature = "cuda")]
pub mod cuda;
pub mod disk;

pub use cpu_mock::CpuMockBackend;
#[cfg(feature = "cuda")]
pub use cuda::CudaBackend;
pub use disk::{DiskBackend, SwaCachingStrategy};

/// Opaque handle into a backend allocation.
///
/// **Sprint 5.1 hardening**: previously a `(raw: usize, len: u64)` carrying a raw address.
/// That model required every backend's `locate()` to do an O(N) linear search comparing
/// `usize` ranges, which (a) silently aliased two regions whose addresses happened to fall
/// adjacent in the host allocator's bins, and (b) made it impossible to validate that a
/// pointer travelling through the FFI boundary actually originated from this backend.
///
/// The new shape is purely a handle: `(region, offset, len)`. Backends look up the region
/// by index in O(1) and reject offsets/lengths that exceed the region size. Cross-region
/// aliasing is structurally impossible. The trade-off — raw GPU device addresses are no
/// longer carried inside the handle — is paid via [`DeviceBackend::device_address`] when a
/// kernel-side pointer is genuinely required.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DevicePtr {
    region: u32,
    offset: u64,
    len: u64,
}

impl DevicePtr {
    /// Construct a fresh region handle. Only backends should call this from
    /// `alloc_region`; downstream code derives sub-pointers via [`Self::offset`].
    #[inline]
    pub const fn from_region(region: u32, len: u64) -> Self {
        Self {
            region,
            offset: 0,
            len,
        }
    }

    /// Derive a sub-pointer at `by` bytes from the start of this handle, with the trailing
    /// length adjusted. Panics in debug builds if `by` exceeds `self.len`; in release builds
    /// the offset is computed unconditionally (the block manager validates upstream).
    #[inline]
    pub fn offset(self, by: u64) -> Self {
        debug_assert!(by <= self.len, "DevicePtr offset out of bounds");
        Self {
            region: self.region,
            offset: self.offset + by,
            len: self.len.saturating_sub(by),
        }
    }

    /// Region index this handle belongs to. Backends index their internal `regions` vec
    /// with this.
    #[inline]
    pub const fn region(self) -> u32 {
        self.region
    }

    /// Byte offset from the start of the region.
    #[inline]
    pub const fn region_offset(self) -> u64 {
        self.offset
    }

    /// Remaining length addressable from this handle.
    #[inline]
    pub const fn len(self) -> u64 {
        self.len
    }

    /// `true` when the handle still addresses bytes.
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.len == 0
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

    /// **Sprint 5.1**: return the backend-native device address (e.g. CUDA device pointer)
    /// for handles that have one. CPU mock and Disk return `None` because their handles do
    /// not correspond to a kernel-callable address.
    ///
    /// This is the *only* place the raw address leaves the backend. GPU kernel call paths
    /// (FlashMLA, FlashInfer wrappers) consume it; downstream Rust code uses the [`DevicePtr`]
    /// handle. Returning `Option` makes "no native address" explicit at every call site.
    fn device_address(&self, _ptr: DevicePtr) -> Option<usize> {
        None
    }

    /// Name of this backend for diagnostics ("cpu_mock", "cuda", "disk").
    fn name(&self) -> &'static str;
}
