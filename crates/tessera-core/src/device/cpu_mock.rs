//! Host-memory mock backend. Used by default everywhere a real GPU is not present (developer
//! workstations on Windows/macOS, CI runners, `cargo test`).
//!
//! The mock keeps all "device" memory in a single `Vec<u8>` per region. `DevicePtr::raw` is
//! the address of the underlying allocation plus an offset, exactly the same shape a real
//! pointer would take. This lets the same `unsafe`-free block-manager code paths run in tests
//! without any cfg-divergence.

use std::sync::Arc;

use anyhow::{bail, Context};
use parking_lot::Mutex;

use super::{DeviceBackend, DevicePtr, RegionKind};

/// Inner state. Wrapped in `Arc<Mutex<>>` so the backend is `Clone + Send + Sync`.
#[derive(Debug, Default)]
struct Inner {
    /// All allocated regions, owned by the mock so their addresses stay stable.
    regions: Vec<Box<[u8]>>,
}

/// CPU-backed mock backend.
#[derive(Debug, Default, Clone)]
pub struct CpuMockBackend {
    inner: Arc<Mutex<Inner>>,
}

impl CpuMockBackend {
    /// Construct a fresh, empty mock backend.
    pub fn new() -> Self {
        Self::default()
    }

    /// Helper: convert a `DevicePtr` into a `(region_base, offset)` pair by locating which
    /// owned region contains it. Returns `None` if the pointer is outside every region.
    fn locate(inner: &Inner, ptr: DevicePtr) -> Option<(usize, usize)> {
        for (idx, region) in inner.regions.iter().enumerate() {
            let base = region.as_ptr() as usize;
            let end = base + region.len();
            if ptr.raw >= base && ptr.raw < end {
                return Some((idx, ptr.raw - base));
            }
        }
        None
    }
}

impl DeviceBackend for CpuMockBackend {
    fn alloc_region(&self, bytes: u64, kind: RegionKind) -> anyhow::Result<DevicePtr> {
        let size = usize::try_from(bytes)
            .with_context(|| format!("region size {bytes} doesn't fit in usize"))?;
        let mut region = vec![0u8; size].into_boxed_slice();
        let raw = region.as_mut_ptr() as usize;
        let mut inner = self.inner.lock();
        tracing::debug!(?kind, size, raw, "cpu_mock alloc_region");
        inner.regions.push(region);
        Ok(DevicePtr {
            raw,
            len: bytes,
        })
    }

    fn memcpy(&self, src: DevicePtr, dst: DevicePtr, bytes: u64) -> anyhow::Result<()> {
        if bytes == 0 {
            return Ok(());
        }
        let n = usize::try_from(bytes).context("memcpy size overflow")?;
        let inner = self.inner.lock();
        let (si, so) =
            Self::locate(&inner, src).context("cpu_mock memcpy: src ptr not in any region")?;
        let (di, dof) =
            Self::locate(&inner, dst).context("cpu_mock memcpy: dst ptr not in any region")?;
        if so + n > inner.regions[si].len() || dof + n > inner.regions[di].len() {
            bail!("cpu_mock memcpy: out-of-range copy {n} bytes");
        }
        // Safety: we have a &mut-equivalent through the lock and have bounds-checked above.
        // We must drop and re-acquire because we need two distinct &mut slices into the Vec.
        drop(inner);
        let mut inner = self.inner.lock();
        if si == di {
            // Intra-region copy: split_at_mut to get disjoint slices.
            let region = &mut inner.regions[si];
            // Build a temporary copy of the source bytes to handle overlap.
            let buf = region[so..so + n].to_vec();
            region[dof..dof + n].copy_from_slice(&buf);
        } else {
            // Cross-region: use indices to get disjoint mutable references.
            let (lo, hi) = if si < di { (si, di) } else { (di, si) };
            let (head, tail) = inner.regions.split_at_mut(hi);
            let (src_region, dst_region) = if si < di {
                (&head[lo], &mut tail[0])
            } else {
                (&tail[0], &mut head[lo])
            };
            dst_region[dof..dof + n].copy_from_slice(&src_region[so..so + n]);
        }
        Ok(())
    }

    fn read_bytes(&self, ptr: DevicePtr, len: usize) -> anyhow::Result<Vec<u8>> {
        let inner = self.inner.lock();
        let (i, off) =
            Self::locate(&inner, ptr).context("cpu_mock read_bytes: ptr not in any region")?;
        if off + len > inner.regions[i].len() {
            bail!("cpu_mock read_bytes: out-of-range read {len} bytes");
        }
        Ok(inner.regions[i][off..off + len].to_vec())
    }

    fn fill_pattern(&self, ptr: DevicePtr, byte: u8, len: usize) -> anyhow::Result<()> {
        let mut inner = self.inner.lock();
        let (i, off) =
            Self::locate(&inner, ptr).context("cpu_mock fill_pattern: ptr not in any region")?;
        if off + len > inner.regions[i].len() {
            bail!("cpu_mock fill_pattern: out-of-range fill {len} bytes");
        }
        inner.regions[i][off..off + len].fill(byte);
        Ok(())
    }

    fn write_bytes(&self, ptr: DevicePtr, bytes: &[u8]) -> anyhow::Result<()> {
        if bytes.is_empty() {
            return Ok(());
        }
        let mut inner = self.inner.lock();
        let (i, off) =
            Self::locate(&inner, ptr).context("cpu_mock write_bytes: ptr not in any region")?;
        if off + bytes.len() > inner.regions[i].len() {
            bail!("cpu_mock write_bytes: out-of-range write {} bytes", bytes.len());
        }
        inner.regions[i][off..off + bytes.len()].copy_from_slice(bytes);
        Ok(())
    }

    fn name(&self) -> &'static str {
        "cpu_mock"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alloc_and_fill_roundtrip() {
        let be = CpuMockBackend::new();
        let p = be.alloc_region(1024, RegionKind::Primary).unwrap();
        be.fill_pattern(p, 0xAB, 1024).unwrap();
        let bytes = be.read_bytes(p, 1024).unwrap();
        assert!(bytes.iter().all(|&b| b == 0xAB));
    }

    #[test]
    fn memcpy_cross_region_isolates_src() {
        let be = CpuMockBackend::new();
        let src = be.alloc_region(256, RegionKind::Primary).unwrap();
        let dst = be.alloc_region(256, RegionKind::Primary).unwrap();
        be.fill_pattern(src, 0x11, 256).unwrap();
        be.memcpy(src, dst, 256).unwrap();
        be.fill_pattern(src, 0x22, 256).unwrap();
        // dst must still have the original 0x11 pattern.
        assert!(be.read_bytes(dst, 256).unwrap().iter().all(|&b| b == 0x11));
    }
}
