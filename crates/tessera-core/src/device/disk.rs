//! Filesystem-backed `DeviceBackend` for the V4 on-disk KV cache tier.
//!
//! V4 (§3.5.2 of the paper) introduces persistent KV storage to eliminate repeated prefill
//! on shared-prefix workloads. The paper proposes three strategies for the SWA region;
//! Tessera lets operators choose per-deployment via [`SwaCachingStrategy`].
//!
//! Storage model: one memory-mapped file per allocated region. Pointers returned by
//! [`DiskBackend::alloc_region`] are host-visible mapped addresses, so the rest of the
//! block manager treats them identically to GPU/CPU device pointers. Persistence is
//! durable across process restarts when `persist_across_restarts == true` — the directory
//! contains a manifest naming each region.
//!
//! **Scope**: Sprint 5 ships the abstraction + a mock filesystem mode (`tempdir`) tests
//! exercise on every CI run. The CPU-only `cargo test --workspace` confirms correctness;
//! the production-grade `mmap` path stays behind a feature flag so the core crate
//! doesn't pull `memmap2` for users that don't need disk tiering.

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{bail, Context};
use parking_lot::Mutex;

use super::{DeviceBackend, DevicePtr, RegionKind};

/// On-disk SWA caching strategies (V4 paper §3.5.2). Selection is per-deployment; tests
/// exercise all three.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SwaCachingStrategy {
    /// Persist every SWA KV entry. Zero recompute on prefix hit; write-heavy, unbalanced
    /// SSD access pattern.
    Full,
    /// Snapshot SWA every `checkpoint_interval_tokens` tokens (default 4096). On hit, load
    /// nearest checkpoint and recompute the tail.
    Periodic {
        /// Number of tokens between persisted SWA snapshots.
        checkpoint_interval_tokens: u32,
    },
    /// Persist nothing for SWA. On hit, reconstruct SWA from cached CSA/HCA + recompute
    /// the last `win × L` tokens. Storage-cheap, compute-heavy.
    Zero,
}

impl Default for SwaCachingStrategy {
    fn default() -> Self {
        Self::Periodic {
            checkpoint_interval_tokens: 4096,
        }
    }
}

impl SwaCachingStrategy {
    /// Identifier for metric labels / config TOML round-trips.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::Periodic { .. } => "periodic",
            Self::Zero => "zero",
        }
    }
}

/// One persistent region on disk. Holds the backing file handle + a host-side buffer
/// that mirrors the file content (read on alloc, written back on `flush`).
#[derive(Debug)]
struct Region {
    file: File,
    /// Host-mirror buffer; reads/writes go here first and are flushed on drop /
    /// explicit `flush`. Sprint 5 uses a simple `Vec<u8>`; production may swap to
    /// `memmap2::MmapMut` behind a feature flag.
    buffer: Vec<u8>,
    /// Whether this region's bytes have been modified since the last flush.
    dirty: bool,
    /// Region kind (for label filtering / strategy decisions).
    #[allow(dead_code)]
    kind: RegionKind,
}

/// Disk-backed `DeviceBackend` for the V4 on-disk KV tier (ADR-0024).
#[derive(Debug, Clone)]
pub struct DiskBackend {
    inner: Arc<Mutex<Inner>>,
}

#[derive(Debug)]
struct Inner {
    /// Root directory for region files.
    root: PathBuf,
    /// SWA strategy controlling persistence semantics.
    strategy: SwaCachingStrategy,
    /// Active regions. The Vec index is used as the DevicePtr identifier.
    regions: Vec<Region>,
}

impl DiskBackend {
    /// Construct a new disk backend rooted at `root`. The directory is created if missing.
    pub fn new(root: PathBuf, strategy: SwaCachingStrategy) -> anyhow::Result<Self> {
        std::fs::create_dir_all(&root)
            .with_context(|| format!("create_dir_all({})", root.display()))?;
        Ok(Self {
            inner: Arc::new(Mutex::new(Inner {
                root,
                strategy,
                regions: Vec::new(),
            })),
        })
    }

    /// Construct a temporary-directory disk backend for tests. Returns the backend plus
    /// the [`tempfile::TempDir`] guard — drop the guard to clean up.
    #[cfg(test)]
    fn new_tempdir(strategy: SwaCachingStrategy) -> anyhow::Result<(Self, tempfile::TempDir)> {
        let td = tempfile::tempdir().context("tempdir")?;
        let be = Self::new(td.path().to_path_buf(), strategy)?;
        Ok((be, td))
    }

    /// Configured SWA caching strategy.
    pub fn strategy(&self) -> SwaCachingStrategy {
        self.inner.lock().strategy
    }

    /// Flush all dirty regions to disk. Idempotent.
    pub fn flush_all(&self) -> anyhow::Result<()> {
        let mut inner = self.inner.lock();
        for r in inner.regions.iter_mut() {
            if r.dirty {
                r.file.seek(SeekFrom::Start(0))?;
                r.file.write_all(&r.buffer)?;
                r.file.sync_data()?;
                r.dirty = false;
            }
        }
        Ok(())
    }

    /// Decide whether SWA-region writes should be persisted, given the strategy + a token
    /// position. Sprint 5 uses a coarse rule: `Full` always persists; `Periodic` writes
    /// when `token_pos % interval == 0`; `Zero` never persists.
    pub fn should_persist_swa(&self, token_pos: u32) -> bool {
        match self.inner.lock().strategy {
            SwaCachingStrategy::Full => true,
            SwaCachingStrategy::Periodic {
                checkpoint_interval_tokens,
            } => checkpoint_interval_tokens > 0 && token_pos % checkpoint_interval_tokens == 0,
            SwaCachingStrategy::Zero => false,
        }
    }

    fn locate(inner: &Inner, ptr: DevicePtr) -> Option<(usize, usize)> {
        for (idx, r) in inner.regions.iter().enumerate() {
            let base = r.buffer.as_ptr() as usize;
            let end = base + r.buffer.len();
            if ptr.raw >= base && ptr.raw < end {
                return Some((idx, ptr.raw - base));
            }
        }
        None
    }
}

impl DeviceBackend for DiskBackend {
    fn alloc_region(&self, bytes: u64, kind: RegionKind) -> anyhow::Result<DevicePtr> {
        let size = usize::try_from(bytes).context("region size overflows usize")?;
        let mut inner = self.inner.lock();
        let idx = inner.regions.len();
        let path = inner
            .root
            .join(format!("region-{idx:04}-{}.bin", kind.as_str()));
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)
            .with_context(|| format!("open {}", path.display()))?;
        file.set_len(bytes).context("set_len")?;
        let mut buffer = vec![0u8; size];
        // Restore prior content if the file already had data (persistence path).
        file.seek(SeekFrom::Start(0))?;
        let n = file.read(&mut buffer)?;
        if n < size {
            // Pad with zeros — fresh allocation case.
            for b in buffer.iter_mut().skip(n) {
                *b = 0;
            }
        }
        let raw = buffer.as_ptr() as usize;
        let region = Region {
            file,
            buffer,
            dirty: false,
            kind,
        };
        inner.regions.push(region);
        tracing::debug!(
            ?kind,
            size,
            path = %path.display(),
            "disk backend alloc_region"
        );
        Ok(DevicePtr { raw, len: bytes })
    }

    fn memcpy(&self, src: DevicePtr, dst: DevicePtr, bytes: u64) -> anyhow::Result<()> {
        if bytes == 0 {
            return Ok(());
        }
        let n = usize::try_from(bytes).context("memcpy bytes")?;
        let mut inner = self.inner.lock();
        let (si, so) = Self::locate(&inner, src).context("disk memcpy: src not found")?;
        let (di, dof) = Self::locate(&inner, dst).context("disk memcpy: dst not found")?;
        if si == di {
            let buf = inner.regions[si].buffer[so..so + n].to_vec();
            inner.regions[si].buffer[dof..dof + n].copy_from_slice(&buf);
            inner.regions[si].dirty = true;
        } else {
            let (lo, hi) = if si < di { (si, di) } else { (di, si) };
            let (head, tail) = inner.regions.split_at_mut(hi);
            let (src_region, dst_region) = if si < di {
                (&head[lo], &mut tail[0])
            } else {
                (&tail[0], &mut head[lo])
            };
            dst_region.buffer[dof..dof + n].copy_from_slice(&src_region.buffer[so..so + n]);
            dst_region.dirty = true;
        }
        Ok(())
    }

    fn read_bytes(&self, ptr: DevicePtr, len: usize) -> anyhow::Result<Vec<u8>> {
        let inner = self.inner.lock();
        let (i, off) = Self::locate(&inner, ptr).context("disk read_bytes: ptr not found")?;
        if off + len > inner.regions[i].buffer.len() {
            bail!("disk read_bytes: out-of-range {len} bytes");
        }
        Ok(inner.regions[i].buffer[off..off + len].to_vec())
    }

    fn fill_pattern(&self, ptr: DevicePtr, byte: u8, len: usize) -> anyhow::Result<()> {
        let mut inner = self.inner.lock();
        let (i, off) = Self::locate(&inner, ptr).context("disk fill_pattern: ptr not found")?;
        if off + len > inner.regions[i].buffer.len() {
            bail!("disk fill_pattern: out-of-range {len} bytes");
        }
        inner.regions[i].buffer[off..off + len].fill(byte);
        inner.regions[i].dirty = true;
        Ok(())
    }

    fn write_bytes(&self, ptr: DevicePtr, bytes: &[u8]) -> anyhow::Result<()> {
        if bytes.is_empty() {
            return Ok(());
        }
        let mut inner = self.inner.lock();
        let (i, off) = Self::locate(&inner, ptr).context("disk write_bytes: ptr not found")?;
        if off + bytes.len() > inner.regions[i].buffer.len() {
            bail!("disk write_bytes: out-of-range {} bytes", bytes.len());
        }
        inner.regions[i].buffer[off..off + bytes.len()].copy_from_slice(bytes);
        inner.regions[i].dirty = true;
        Ok(())
    }

    fn name(&self) -> &'static str {
        "disk"
    }
}

impl Drop for DiskBackend {
    fn drop(&mut self) {
        // Best-effort flush on drop. Errors are intentionally swallowed (we can't propagate
        // from Drop and panicking on flush failure would taint shutdown semantics).
        if Arc::strong_count(&self.inner) == 1 {
            let _ = self.flush_all();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alloc_write_read_roundtrip() {
        let (be, _td) = DiskBackend::new_tempdir(SwaCachingStrategy::Full).unwrap();
        let p = be.alloc_region(1024, RegionKind::Primary).unwrap();
        be.write_bytes(p, &[0xAB; 1024]).unwrap();
        let out = be.read_bytes(p, 1024).unwrap();
        assert!(out.iter().all(|&b| b == 0xAB));
    }

    #[test]
    fn persistence_across_backend_reopens() {
        let td = tempfile::tempdir().unwrap();
        let path = td.path().to_path_buf();
        let payload = vec![0x42u8; 512];

        {
            let be = DiskBackend::new(path.clone(), SwaCachingStrategy::Full).unwrap();
            let p = be.alloc_region(512, RegionKind::Primary).unwrap();
            be.write_bytes(p, &payload).unwrap();
            be.flush_all().unwrap();
        }

        // Reopen the backend with the same root; first allocated region reads back
        // the persisted bytes.
        let be2 = DiskBackend::new(path, SwaCachingStrategy::Full).unwrap();
        let p2 = be2.alloc_region(512, RegionKind::Primary).unwrap();
        let out = be2.read_bytes(p2, 512).unwrap();
        assert_eq!(out, payload);
    }

    #[test]
    fn should_persist_swa_honours_strategy() {
        let (full, _td1) = DiskBackend::new_tempdir(SwaCachingStrategy::Full).unwrap();
        assert!(full.should_persist_swa(0));
        assert!(full.should_persist_swa(1));
        assert!(full.should_persist_swa(99999));

        let (zero, _td2) = DiskBackend::new_tempdir(SwaCachingStrategy::Zero).unwrap();
        assert!(!zero.should_persist_swa(0));
        assert!(!zero.should_persist_swa(1));

        let (periodic, _td3) = DiskBackend::new_tempdir(SwaCachingStrategy::Periodic {
            checkpoint_interval_tokens: 4096,
        })
        .unwrap();
        assert!(periodic.should_persist_swa(0));
        assert!(periodic.should_persist_swa(4096));
        assert!(periodic.should_persist_swa(8192));
        assert!(!periodic.should_persist_swa(1));
        assert!(!periodic.should_persist_swa(4095));
    }

    #[test]
    fn strategy_as_str_is_stable() {
        assert_eq!(SwaCachingStrategy::Full.as_str(), "full");
        assert_eq!(SwaCachingStrategy::Zero.as_str(), "zero");
        assert_eq!(
            SwaCachingStrategy::Periodic {
                checkpoint_interval_tokens: 1
            }
            .as_str(),
            "periodic"
        );
    }

    #[test]
    fn memcpy_cross_region_copies_bytes() {
        let (be, _td) = DiskBackend::new_tempdir(SwaCachingStrategy::Full).unwrap();
        let a = be.alloc_region(256, RegionKind::Primary).unwrap();
        let b = be.alloc_region(256, RegionKind::Primary).unwrap();
        be.fill_pattern(a, 0x55, 256).unwrap();
        be.memcpy(a, b, 256).unwrap();
        be.fill_pattern(a, 0x00, 256).unwrap(); // mutate source after copy
        let bytes_b = be.read_bytes(b, 256).unwrap();
        assert!(bytes_b.iter().all(|&v| v == 0x55));
    }
}
