//! CUDA device backend.
//!
//! Compiled only with `--features cuda`. The implementation is intentionally a thin shim over
//! `cudarc`: it owns one [`cudarc::driver::CudaDevice`] per backend and a parking-lot mutex
//! around the per-region allocations. The block manager treats CUDA addresses identically to
//! mock addresses — the `DevicePtr::raw` field is the device pointer.
//!
//! Production hardening notes (tracked in ADR-0008 and ADR-0003):
//!
//! * `read_bytes` performs a synchronous `dtoh` copy. At production block-rates the round-trip
//!   dominates `seal()` latency. A future on-device hashing kernel removes this transfer.
//! * `memcpy` uses `dtod` and is fully asynchronous; the lock-free CudaStream lives on the
//!   inner state. Sprint 0 keeps it synchronous to match the mock backend's semantics.

#![allow(missing_docs)]

use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
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
    /// Cached pointer value so we can rebuild the (slice, offset) pair on lookup. cudarc
    /// returns `*mut c_void` from `device_ptr()`, but we treat it as `usize` opaquely.
    raw_base: usize,
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

    fn locate(inner: &Inner, ptr: DevicePtr) -> Option<(usize, usize)> {
        for (idx, r) in inner.regions.iter().enumerate() {
            let end = r.raw_base + r.len as usize;
            if ptr.raw >= r.raw_base && ptr.raw < end {
                return Some((idx, ptr.raw - r.raw_base));
            }
        }
        None
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
        let raw_base = *slice.device_ptr() as usize;
        inner.regions.push(Region {
            slice,
            raw_base,
            len: bytes,
        });
        tracing::debug!(?kind, bytes, raw_base, "cuda alloc_region");
        Ok(DevicePtr {
            raw: raw_base,
            len: bytes,
        })
    }

    fn memcpy(&self, src: DevicePtr, dst: DevicePtr, bytes: u64) -> Result<()> {
        let mut inner = self.inner.lock();
        let (si, so) = Self::locate(&inner, src).context("cuda memcpy: src not found")?;
        let (di, dof) = Self::locate(&inner, dst).context("cuda memcpy: dst not found")?;
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
        let (i, off) = Self::locate(&inner, ptr).context("cuda read_bytes: ptr not found")?;
        let view = inner.regions[i].slice.slice(off..off + len);
        let host: Vec<u8> = inner.device.dtoh_sync_copy(&view).map_err(cuda_err)?;
        Ok(host)
    }

    fn fill_pattern(&self, ptr: DevicePtr, byte: u8, len: usize) -> Result<()> {
        let mut inner = self.inner.lock();
        let (i, off) = Self::locate(&inner, ptr).context("cuda fill: ptr not found")?;
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
        let (i, off) = Self::locate(&inner, ptr).context("cuda write_bytes: ptr not found")?;
        let device = Arc::clone(&inner.device);
        let mut view = inner.regions[i].slice.slice_mut(off..off + bytes.len());
        device
            .htod_sync_copy_into(bytes, &mut view)
            .map_err(cuda_err)?;
        Ok(())
    }

    fn name(&self) -> &'static str {
        "cuda"
    }
}

// ─── CudaXxh3Hasher stub (WS8 / TD-002) ──────────────────────────────────────
//
// Sprint 2 stub: exposes the `ContentHasher::hash_device` override seam without implementing
// an actual GPU kernel. The stub uses a deterministic placeholder hash (byte-sum accumulation)
// so tests can exercise the override path without a real xxhash kernel. The optimised kernel
// implementation is deferred to Sprint 3 (see ADR-0003).

use crate::content_hash::ContentHasher;

/// CUDA-backed content hasher stub. Overrides `ContentHasher::hash_device` to avoid
/// the `dtoh` round-trip, using a placeholder deterministic hash (not real xxhash3).
///
/// Replace `hash` body with a cuRand/cuBLAS-free xxh3 CUDA kernel in Sprint 3.
pub struct CudaXxh3Hasher {
    backend: Arc<CudaBackend>,
}

impl CudaXxh3Hasher {
    /// Wrap an existing CUDA backend.
    pub fn new(backend: Arc<CudaBackend>) -> Self {
        Self { backend }
    }
}

impl ContentHasher for CudaXxh3Hasher {
    fn hash(&self, bytes: &[u8]) -> u64 {
        // Stub: byte-position-weighted sum. Deterministic but not xxhash3. Sprint 3 replaces.
        bytes.iter().enumerate().fold(0u64, |acc, (i, &b)| {
            acc.wrapping_add(u64::from(b).wrapping_mul(i as u64 + 1))
        })
    }

    fn name(&self) -> &'static str {
        "cuda-xxh3-stub"
    }

    /// Override: reads bytes from device, then hashes — exercises the override path.
    ///
    /// The stub still does a `dtoh` transfer internally (using `CudaBackend::read_bytes`)
    /// to keep CI green without a GPU kernel. Sprint 3 replaces this with a kernel-side
    /// `cuXxh3` call that never copies to host.
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
