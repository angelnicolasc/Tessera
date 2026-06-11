//! Host-memory mock backend. Used by default everywhere a real GPU is not present (developer
//! workstations on Windows/macOS, CI runners, `cargo test`).
//!
//! **Sprint 5.1**: the mock no longer leaks raw `Vec<u8>` data pointers through
//! [`DevicePtr`]. Each allocation gets a region index; the backend stores its allocations in
//! a `Vec<Box<[u8]>>` indexed directly by `ptr.region()`. Lookup is O(1) and cross-region
//! aliasing is structurally impossible.

use std::sync::Arc;

use anyhow::{bail, Context};
use parking_lot::Mutex;

use super::{DeviceBackend, DevicePtr, RegionKind};

/// Inner state. Wrapped in `Arc<Mutex<>>` so the backend is `Clone + Send + Sync`.
#[derive(Debug, Default)]
struct Inner {
    /// All allocated regions, owned by the mock so their slot indices stay stable for the
    /// lifetime of the backend. Index = region id encoded in [`DevicePtr::region`].
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

    /// Resolve a [`DevicePtr`] to a `(region_index, offset)` pair after validating that the
    /// region exists and the offset (plus the caller-supplied length, if any) fits inside
    /// the region.
    fn locate(inner: &Inner, ptr: DevicePtr) -> anyhow::Result<(usize, usize)> {
        let idx = ptr.region() as usize;
        if idx >= inner.regions.len() {
            bail!(
                "cpu_mock: DevicePtr targets region {idx} but only {} regions exist",
                inner.regions.len()
            );
        }
        let off = usize::try_from(ptr.region_offset())
            .context("cpu_mock: region_offset overflows usize")?;
        if off > inner.regions[idx].len() {
            bail!(
                "cpu_mock: offset {off} beyond region size {} (region {idx})",
                inner.regions[idx].len()
            );
        }
        Ok((idx, off))
    }
}

impl DeviceBackend for CpuMockBackend {
    fn alloc_region(&self, bytes: u64, kind: RegionKind) -> anyhow::Result<DevicePtr> {
        let size = usize::try_from(bytes)
            .with_context(|| format!("region size {bytes} doesn't fit in usize"))?;
        let region = vec![0u8; size].into_boxed_slice();
        let mut inner = self.inner.lock();
        let idx = u32::try_from(inner.regions.len())
            .context("cpu_mock: region count exceeds u32::MAX")?;
        tracing::debug!(?kind, size, region = idx, "cpu_mock alloc_region");
        inner.regions.push(region);
        Ok(DevicePtr::from_region(idx, bytes))
    }

    fn memcpy(&self, src: DevicePtr, dst: DevicePtr, bytes: u64) -> anyhow::Result<()> {
        if bytes == 0 {
            return Ok(());
        }
        let n = usize::try_from(bytes).context("memcpy size overflow")?;
        let mut inner = self.inner.lock();
        let (si, so) = Self::locate(&inner, src).context("cpu_mock memcpy: src invalid")?;
        let (di, dof) = Self::locate(&inner, dst).context("cpu_mock memcpy: dst invalid")?;
        if so + n > inner.regions[si].len() || dof + n > inner.regions[di].len() {
            bail!("cpu_mock memcpy: out-of-range copy {n} bytes");
        }
        if si == di {
            // Intra-region copy: stage through a temporary to handle overlap.
            let buf = inner.regions[si][so..so + n].to_vec();
            inner.regions[si][dof..dof + n].copy_from_slice(&buf);
        } else {
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
        let (i, off) = Self::locate(&inner, ptr).context("cpu_mock read_bytes: invalid ptr")?;
        if off + len > inner.regions[i].len() {
            bail!("cpu_mock read_bytes: out-of-range read {len} bytes");
        }
        Ok(inner.regions[i][off..off + len].to_vec())
    }

    fn fill_pattern(&self, ptr: DevicePtr, byte: u8, len: usize) -> anyhow::Result<()> {
        let mut inner = self.inner.lock();
        let (i, off) = Self::locate(&inner, ptr).context("cpu_mock fill_pattern: invalid ptr")?;
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
        let (i, off) = Self::locate(&inner, ptr).context("cpu_mock write_bytes: invalid ptr")?;
        if off + bytes.len() > inner.regions[i].len() {
            bail!(
                "cpu_mock write_bytes: out-of-range write {} bytes",
                bytes.len()
            );
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
        assert!(be.read_bytes(dst, 256).unwrap().iter().all(|&b| b == 0x11));
    }

    #[test]
    fn pointer_to_nonexistent_region_is_rejected() {
        let be = CpuMockBackend::new();
        // Construct a handle to region 99 — never allocated.
        let bad = DevicePtr::from_region(99, 64);
        assert!(be.read_bytes(bad, 64).is_err());
        assert!(be.write_bytes(bad, &[0u8; 64]).is_err());
        assert!(be.fill_pattern(bad, 0x00, 64).is_err());
    }

    #[test]
    fn read_past_region_end_is_rejected() {
        let be = CpuMockBackend::new();
        let p = be.alloc_region(128, RegionKind::Primary).unwrap();
        // Reading more bytes than the region holds must error, not panic.
        assert!(be.read_bytes(p, 256).is_err());
    }
}
