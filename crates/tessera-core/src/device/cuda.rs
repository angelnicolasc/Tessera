//! CUDA device backend.
//!
//! Compiled only with `--features cuda`. The implementation is intentionally a thin shim over
//! `cudarc`: it owns one [`cudarc::driver::CudaDevice`] per backend and a parking-lot mutex
//! around the per-region allocations. The block manager treats CUDA addresses identically to
//! mock addresses — the [`DevicePtr`] is an opaque handle (Sprint 5.1 hardening: previously
//! a raw GPU device address; now `(region, offset, len)`).
//!
//! Kernel-call sites that genuinely need the raw GPU pointer (e.g. FlashMLA dispatch) go
//! through [`DeviceBackend::device_address`].
//!
//! Production hardening notes (tracked in ADR-0008 and ADR-0003):
//!
//! * `read_bytes` performs a synchronous `dtoh` copy. At production block-rates the round-trip
//!   dominates `seal()` latency. A future on-device hashing kernel removes this transfer.
//! * `memcpy` uses `dtod` and is fully asynchronous; the lock-free CudaStream lives on the
//!   inner state. Sprint 0 keeps it synchronous to match the mock backend's semantics.

#![allow(missing_docs)]

use std::sync::Arc;

use anyhow::{anyhow, bail, Context, Result};
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr as CdarcPtr};
use parking_lot::Mutex;

/// Wrap a `cudarc` `DriverError` into an `anyhow::Error`. cudarc 0.12.1 dropped the
/// `std::error::Error` impl on `DriverError`, so the usual `?` / `.context()` chain
/// no longer compiles — adapt via `.map_err` at every call site.
fn cuda_err(e: cudarc::driver::DriverError) -> anyhow::Error {
    anyhow!("cudarc driver error: {e:?}")
}

use super::{DeviceBackend, DevicePtr, RegionKind};

#[derive(Debug)]
struct Region {
    slice: CudaSlice<u8>,
    /// Native device address (`*mut c_void` as `usize`). Used only when the kernel-call path
    /// asks for it via [`CudaBackend::device_address`]; the [`DevicePtr`] handle does not
    /// carry this any more.
    device_addr: usize,
    len: u64,
}

#[derive(Debug)]
struct Inner {
    device: Arc<CudaDevice>,
    regions: Vec<Region>,
}

/// CUDA-backed device implementation. Behind feature `cuda`.
#[derive(Debug, Clone)]
pub struct CudaBackend {
    inner: Arc<Mutex<Inner>>,
}

impl CudaBackend {
    /// Initialise CUDA on the given device ordinal.
    pub fn new(device_ordinal: usize) -> Result<Self> {
        let device = CudaDevice::new(device_ordinal)
            .map_err(cuda_err)
            .with_context(|| format!("CUDA init failed for device {device_ordinal}"))?;
        Ok(Self {
            inner: Arc::new(Mutex::new(Inner {
                device,
                regions: Vec::new(),
            })),
        })
    }

    /// Resolve a [`DevicePtr`] to `(region_index, byte_offset)` after validating bounds.
    fn locate(inner: &Inner, ptr: DevicePtr) -> Result<(usize, usize)> {
        let idx = ptr.region() as usize;
        if idx >= inner.regions.len() {
            bail!(
                "cuda: DevicePtr targets region {idx} but only {} exist",
                inner.regions.len()
            );
        }
        let off =
            usize::try_from(ptr.region_offset()).context("cuda: region_offset overflows usize")?;
        if (off as u64) > inner.regions[idx].len {
            bail!(
                "cuda: offset {off} beyond region size {} (region {idx})",
                inner.regions[idx].len
            );
        }
        Ok((idx, off))
    }
}

impl DeviceBackend for CudaBackend {
    fn alloc_region(&self, bytes: u64, kind: RegionKind) -> Result<DevicePtr> {
        let mut inner = self.inner.lock();
        let slice: CudaSlice<u8> = inner
            .device
            .alloc_zeros::<u8>(bytes as usize)
            .map_err(cuda_err)
            .with_context(|| format!("CUDA alloc {bytes} bytes ({kind:?}) failed"))?;
        // cudarc 0.12.1: device_ptr() returns `&CUdeviceptr` (`&u64`); dereference
        // before casting.
        let device_addr = *slice.device_ptr() as usize;
        let idx =
            u32::try_from(inner.regions.len()).context("cuda: region count exceeds u32::MAX")?;
        inner.regions.push(Region {
            slice,
            device_addr,
            len: bytes,
        });
        tracing::debug!(?kind, bytes, region = idx, device_addr, "cuda alloc_region");
        Ok(DevicePtr::from_region(idx, bytes))
    }

    fn memcpy(&self, src: DevicePtr, dst: DevicePtr, bytes: u64) -> Result<()> {
        let mut inner = self.inner.lock();
        let (si, so) = Self::locate(&inner, src).context("cuda memcpy: src")?;
        let (di, dof) = Self::locate(&inner, dst).context("cuda memcpy: dst")?;
        let n = bytes as usize;
        // `device` is `Arc<CudaDevice>` — clone the handle once so we can hold the
        // immutable device reference and a `&mut Region.slice` concurrently without
        // borrow-checker conflicts on `inner`.
        let device = Arc::clone(&inner.device);
        if si == di {
            // Same region: stage through host (cudarc dtod_copy requires distinct slices).
            let host: Vec<u8> = {
                let view = inner.regions[si].slice.slice(so..so + n);
                device.dtoh_sync_copy(&view).map_err(cuda_err)?
            };
            let mut dst_view = inner.regions[di].slice.slice_mut(dof..dof + n);
            device
                .htod_sync_copy_into(&host, &mut dst_view)
                .map_err(cuda_err)
                .context("cuda dtod self-copy via host failed")?;
        } else {
            // Distinct regions: split the vec to get independent mutable + immutable borrows.
            let (lo, hi) = if si < di { (si, di) } else { (di, si) };
            let (left, right) = inner.regions.split_at_mut(hi);
            let (src_region, dst_region) = if si < di {
                (&left[lo], &mut right[0])
            } else {
                (&right[0], &mut left[lo])
            };
            let src_view = src_region.slice.slice(so..so + n);
            let mut dst_view = dst_region.slice.slice_mut(dof..dof + n);
            device
                .dtod_copy(&src_view, &mut dst_view)
                .map_err(cuda_err)
                .context("cuda dtod copy failed")?;
        }
        Ok(())
    }

    fn read_bytes(&self, ptr: DevicePtr, len: usize) -> Result<Vec<u8>> {
        let inner = self.inner.lock();
        let (i, off) = Self::locate(&inner, ptr).context("cuda read_bytes")?;
        let view = inner.regions[i].slice.slice(off..off + len);
        let host: Vec<u8> = inner.device.dtoh_sync_copy(&view).map_err(cuda_err)?;
        Ok(host)
    }

    fn fill_pattern(&self, ptr: DevicePtr, byte: u8, len: usize) -> Result<()> {
        let mut inner = self.inner.lock();
        let (i, off) = Self::locate(&inner, ptr).context("cuda fill")?;
        let host = vec![byte; len];
        let device = Arc::clone(&inner.device);
        let mut view = inner.regions[i].slice.slice_mut(off..off + len);
        device
            .htod_sync_copy_into(&host, &mut view)
            .map_err(cuda_err)?;
        Ok(())
    }

    fn write_bytes(&self, ptr: DevicePtr, bytes: &[u8]) -> Result<()> {
        if bytes.is_empty() {
            return Ok(());
        }
        let mut inner = self.inner.lock();
        let (i, off) = Self::locate(&inner, ptr).context("cuda write_bytes")?;
        let device = Arc::clone(&inner.device);
        let mut view = inner.regions[i].slice.slice_mut(off..off + bytes.len());
        device
            .htod_sync_copy_into(bytes, &mut view)
            .map_err(cuda_err)?;
        Ok(())
    }

    fn device_address(&self, ptr: DevicePtr) -> Option<usize> {
        let inner = self.inner.lock();
        let idx = ptr.region() as usize;
        let region = inner.regions.get(idx)?;
        let off = usize::try_from(ptr.region_offset()).ok()?;
        if (off as u64) > region.len {
            return None;
        }
        Some(region.device_addr + off)
    }

    fn name(&self) -> &'static str {
        "cuda"
    }
}

// ─── CudaXxh3HasherStub (Sprint 2 seam — DO NOT USE IN PRODUCTION) ────────────
//
// The stub uses a position-weighted byte-sum as a "hash". This is trivially collision-able by
// construction. It exists ONLY so the `hash_device` override path compiles and exercises in
// `cargo test`; replacing the body with a real xxh3 CUDA kernel is tracked as TD-002.
//
// Sprint 5.1 hardening: hidden from `pub use`, renamed with the `Stub` suffix, deprecated
// for any non-test use. Constructing it without the `__acknowledge_stub_collision_risk`
// flag panics — this is intentionally annoying so nobody wires it into a production block
// manager by accident.

use crate::content_hash::ContentHasher;

/// **CUDA hasher stub — do not use in production.**
///
/// Uses a position-weighted byte sum, not xxh3. Trivially collidable. Exists only to keep
/// the `hash_device` override seam exercised in tests until the real GPU kernel lands
/// (TD-002).
///
/// Construct only via [`CudaXxh3HasherStub::new_acknowledging_stub_collision_risk`]; the
/// `new` constructor panics unconditionally.
#[doc(hidden)]
#[deprecated(
    since = "0.6.1",
    note = "stub hasher — trivially colliable. Wait for the real GPU xxh3 kernel (TD-002)."
)]
pub struct CudaXxh3HasherStub {
    backend: Arc<CudaBackend>,
}

#[allow(deprecated)]
impl CudaXxh3HasherStub {
    /// Panics. Use [`Self::new_acknowledging_stub_collision_risk`] in tests if you
    /// genuinely need to exercise the override path.
    pub fn new(_backend: Arc<CudaBackend>) -> Self {
        panic!(
            "CudaXxh3HasherStub is a stub with O(1) crafted-collision difficulty. \
             Production code must use Xxh3Hasher (CPU) or the future GPU xxh3 kernel \
             (TD-002). Tests that genuinely need the override seam should call \
             new_acknowledging_stub_collision_risk()."
        )
    }

    /// Test-only constructor. The name is intentionally annoying.
    pub fn new_acknowledging_stub_collision_risk(backend: Arc<CudaBackend>) -> Self {
        Self { backend }
    }
}

#[allow(deprecated)]
impl ContentHasher for CudaXxh3HasherStub {
    fn hash(&self, bytes: &[u8]) -> u64 {
        // Position-weighted byte sum. DO NOT USE FOR CONTENT ADDRESSING.
        bytes.iter().enumerate().fold(0u64, |acc, (i, &b)| {
            acc.wrapping_add(u64::from(b).wrapping_mul(i as u64 + 1))
        })
    }

    fn name(&self) -> &'static str {
        "cuda-xxh3-stub"
    }

    fn hash_device(
        &self,
        _backend: &dyn crate::device::DeviceBackend,
        ptr: crate::device::DevicePtr,
        len: usize,
    ) -> anyhow::Result<u64> {
        let bytes = self.backend.read_bytes(ptr, len)?;
        Ok(self.hash(&bytes))
    }
}
