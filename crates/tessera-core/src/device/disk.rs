//! Filesystem-backed `DeviceBackend` for the V4 on-disk KV cache tier.
//!
//! V4 (§3.5.2 of the paper) introduces persistent KV storage to eliminate repeated prefill
//! on shared-prefix workloads. The paper proposes three strategies for the SWA region;
//! Tessera lets operators choose per-deployment via [`SwaCachingStrategy`].
//!
//! Storage model:
//!
//! * **One file per region** under the configured `root`.
//! * **Host-mirror buffer**: reads/writes hit a [`Vec<u8>`] first and flush back on
//!   `flush_all` or `Drop`. Sprint 6 swaps this for `memmap2` (TD-035).
//! * **Manifest** (Sprint 5.1): `tessera-disk-manifest.json` records `(region_index, kind,
//!   size, sha256)` for every region. On reopen, region files whose checksum disagrees with
//!   the manifest are quarantined rather than silently re-used. This neutralises the
//!   prior cross-process aliasing risk where two processes pointing at the same `root` with
//!   distinct allocation orders would read each other's content (see audit C4).
//! * **Mode 0o600** on Unix (Sprint 5.1): region files are owner-only. KV cache content
//!   shadows the prompt; world-readable on a shared host is a multi-tenant leak.
//! * **Canonicalised root** (Sprint 5.1): `DiskBackend::new` rejects `root` paths that fall
//!   outside an explicit allowlist when one is configured via
//!   [`DiskBackend::with_allowed_roots`].
//!
//! Scope: Sprint 5 ships the abstraction + a tempdir-tested mock filesystem mode. Production-
//! grade `mmap` stays behind a feature flag (TD-035).

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{anyhow, bail, Context};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use xxhash_rust::xxh3::xxh3_64;

use super::{DeviceBackend, DevicePtr, RegionKind};

const MANIFEST_NAME: &str = "tessera-disk-manifest.json";

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

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ManifestEntry {
    index: u32,
    kind: String,
    size: u64,
    /// xxh3-64 of the file's bytes at last flush. Verified on reopen; mismatches quarantine
    /// the region.
    checksum: u64,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
struct Manifest {
    entries: Vec<ManifestEntry>,
}

impl Manifest {
    fn load(root: &Path) -> anyhow::Result<Self> {
        let path = root.join(MANIFEST_NAME);
        if !path.exists() {
            return Ok(Self::default());
        }
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("read manifest {}", path.display()))?;
        let parsed: Self = serde_json::from_str(&text)
            .with_context(|| format!("parse manifest {}", path.display()))?;
        Ok(parsed)
    }

    fn save(&self, root: &Path) -> anyhow::Result<()> {
        let path = root.join(MANIFEST_NAME);
        let text = serde_json::to_string_pretty(self).context("serialise manifest")?;
        // Atomic-ish write: write to a tempfile in the same directory then rename.
        let tmp = root.join(format!("{MANIFEST_NAME}.tmp"));
        std::fs::write(&tmp, text)
            .with_context(|| format!("write manifest tmp {}", tmp.display()))?;
        std::fs::rename(&tmp, &path)
            .with_context(|| format!("rename manifest tmp -> {}", path.display()))?;
        Ok(())
    }
}

/// One persistent region on disk. Holds the backing file handle + a host-side buffer
/// that mirrors the file content (read on alloc, written back on `flush`).
#[derive(Debug)]
struct Region {
    file: File,
    /// Host-mirror buffer; reads/writes go here first and are flushed on drop /
    /// explicit `flush`. Sprint 5 uses a simple `Vec<u8>`; production may swap to
    /// `memmap2::MmapMut` behind a feature flag (TD-035).
    buffer: Vec<u8>,
    /// Whether this region's bytes have been modified since the last flush.
    dirty: bool,
    /// Region kind (for label filtering / strategy decisions).
    kind: RegionKind,
    /// `true` when manifest verification failed at reopen — region is fresh-zeroed and
    /// will overwrite the old file on next flush. Retained for debug introspection; the
    /// `quarantined_count` on `Inner` is the authoritative observability surface.
    #[allow(dead_code)]
    quarantined: bool,
}

/// Disk-backed `DeviceBackend` for the V4 on-disk KV tier (ADR-0024).
#[derive(Debug, Clone)]
pub struct DiskBackend {
    inner: Arc<Mutex<Inner>>,
}

#[derive(Debug)]
struct Inner {
    /// Canonicalised root directory for region files. Sprint 5.1 stores this AFTER
    /// `canonicalize` so symlink redirection is decided once, at construction, and any
    /// subsequent symlink edit cannot redirect writes.
    root: PathBuf,
    /// SWA strategy controlling persistence semantics.
    strategy: SwaCachingStrategy,
    /// Active regions. The Vec index is used as the [`DevicePtr::region`] identifier.
    regions: Vec<Region>,
    /// Manifest of regions known at construction time. Updated on every flush.
    manifest: Manifest,
    /// Counter of regions whose checksum failed verification on reopen. Surfaced via
    /// [`DiskBackend::quarantined_regions`].
    quarantined_count: u32,
}

impl DiskBackend {
    /// Construct a new disk backend rooted at `root`. The directory is created if missing.
    /// On Unix the root is created with mode 0o700 (owner-only).
    ///
    /// Sprint 5.1 hardening:
    /// * Canonicalises `root` after creating it; subsequent symlink swaps cannot redirect
    ///   region writes.
    /// * Loads `tessera-disk-manifest.json` and verifies any pre-existing region file's
    ///   xxh3 checksum against the manifest. Mismatches quarantine the region (the buffer
    ///   is zeroed; the old file will be overwritten on first flush). Missing manifest is
    ///   tolerated (fresh root or pre-5.1 root).
    pub fn new(root: PathBuf, strategy: SwaCachingStrategy) -> anyhow::Result<Self> {
        Self::new_with_allowed_roots(root, strategy, None)
    }

    /// Like [`Self::new`] but rejects `root` paths that don't resolve inside one of the
    /// allowlisted prefixes. Useful for multi-tenant deployments — pass per-tenant roots.
    pub fn with_allowed_roots(
        root: PathBuf,
        strategy: SwaCachingStrategy,
        allowed: &[PathBuf],
    ) -> anyhow::Result<Self> {
        Self::new_with_allowed_roots(root, strategy, Some(allowed))
    }

    fn new_with_allowed_roots(
        root: PathBuf,
        strategy: SwaCachingStrategy,
        allowed: Option<&[PathBuf]>,
    ) -> anyhow::Result<Self> {
        Self::create_root(&root)?;
        let canonical = root
            .canonicalize()
            .with_context(|| format!("canonicalize({})", root.display()))?;
        if let Some(prefixes) = allowed {
            let ok = prefixes.iter().any(|p| {
                p.canonicalize()
                    .map(|c| canonical.starts_with(&c))
                    .unwrap_or(false)
            });
            if !ok {
                bail!(
                    "DiskBackend: root {} is not inside any allowed prefix {:?}",
                    canonical.display(),
                    prefixes
                );
            }
        }
        let manifest = Manifest::load(&canonical)?;
        Ok(Self {
            inner: Arc::new(Mutex::new(Inner {
                root: canonical,
                strategy,
                regions: Vec::new(),
                manifest,
                quarantined_count: 0,
            })),
        })
    }

    #[cfg(unix)]
    fn create_root(root: &Path) -> anyhow::Result<()> {
        use std::os::unix::fs::DirBuilderExt;
        std::fs::DirBuilder::new()
            .recursive(true)
            .mode(0o700)
            .create(root)
            .with_context(|| format!("create_dir_all({}) with mode 0o700", root.display()))
    }

    #[cfg(not(unix))]
    fn create_root(root: &Path) -> anyhow::Result<()> {
        std::fs::create_dir_all(root).with_context(|| format!("create_dir_all({})", root.display()))
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

    /// Number of regions that failed manifest verification on reopen.
    pub fn quarantined_regions(&self) -> u32 {
        self.inner.lock().quarantined_count
    }

    /// Flush all dirty regions to disk and rewrite the manifest with current checksums.
    /// Idempotent.
    pub fn flush_all(&self) -> anyhow::Result<()> {
        let mut inner = self.inner.lock();
        let mut entries: Vec<ManifestEntry> = Vec::with_capacity(inner.regions.len());
        for (idx, r) in inner.regions.iter_mut().enumerate() {
            if r.dirty {
                r.file.seek(SeekFrom::Start(0))?;
                r.file.write_all(&r.buffer)?;
                r.file.sync_data()?;
                r.dirty = false;
            }
            let checksum = xxh3_64(&r.buffer);
            entries.push(ManifestEntry {
                index: u32::try_from(idx).map_err(|_| anyhow!("region count exceeds u32::MAX"))?,
                kind: r.kind.as_str().to_string(),
                size: r.buffer.len() as u64,
                checksum,
            });
        }
        inner.manifest.entries = entries;
        let root = inner.root.clone();
        let manifest = inner.manifest.clone();
        drop(inner);
        manifest.save(&root)?;
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

    fn locate(inner: &Inner, ptr: DevicePtr) -> anyhow::Result<(usize, usize)> {
        let idx = ptr.region() as usize;
        if idx >= inner.regions.len() {
            bail!(
                "disk: DevicePtr targets region {idx} but only {} exist",
                inner.regions.len()
            );
        }
        let off =
            usize::try_from(ptr.region_offset()).context("disk: region_offset overflows usize")?;
        if off > inner.regions[idx].buffer.len() {
            bail!(
                "disk: offset {off} beyond region size {} (region {idx})",
                inner.regions[idx].buffer.len()
            );
        }
        Ok((idx, off))
    }
}

#[cfg(unix)]
fn open_region_file(path: &Path) -> anyhow::Result<File> {
    use std::os::unix::fs::OpenOptionsExt;
    OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .mode(0o600)
        .open(path)
        .with_context(|| format!("open {} mode 0o600", path.display()))
}

#[cfg(not(unix))]
fn open_region_file(path: &Path) -> anyhow::Result<File> {
    OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)
        .with_context(|| format!("open {}", path.display()))
}

impl DeviceBackend for DiskBackend {
    fn alloc_region(&self, bytes: u64, kind: RegionKind) -> anyhow::Result<DevicePtr> {
        let size = usize::try_from(bytes).context("region size overflows usize")?;
        let mut inner = self.inner.lock();
        let idx =
            u32::try_from(inner.regions.len()).context("disk: region count exceeds u32::MAX")?;
        let path = inner
            .root
            .join(format!("region-{idx:04}-{}.bin", kind.as_str()));
        let mut file = open_region_file(&path)?;
        file.set_len(bytes).context("set_len")?;
        let mut buffer = vec![0u8; size];
        file.seek(SeekFrom::Start(0))?;
        let n = file.read(&mut buffer)?;
        if n < size {
            for b in buffer.iter_mut().skip(n) {
                *b = 0;
            }
        }
        // Manifest verification: if we have a prior entry for this index, check the
        // checksum + size + kind. Mismatches quarantine.
        let mut quarantined = false;
        let prior = inner
            .manifest
            .entries
            .iter()
            .find(|e| e.index == idx)
            .cloned();
        if let Some(prior_entry) = prior {
            let checksum = xxh3_64(&buffer);
            let same_kind = prior_entry.kind == kind.as_str();
            let same_size = prior_entry.size == bytes;
            let same_checksum = prior_entry.checksum == checksum;
            if !(same_kind && same_size && same_checksum) {
                tracing::warn!(
                    region = idx,
                    expected_kind = %prior_entry.kind,
                    got_kind = %kind.as_str(),
                    expected_size = prior_entry.size,
                    got_size = bytes,
                    expected_checksum = prior_entry.checksum,
                    got_checksum = checksum,
                    "disk: region quarantined — manifest mismatch",
                );
                buffer.fill(0);
                quarantined = true;
                inner.quarantined_count = inner.quarantined_count.saturating_add(1);
            }
        }
        let region = Region {
            file,
            buffer,
            dirty: quarantined, // force overwrite of stale file on next flush
            kind,
            quarantined,
        };
        inner.regions.push(region);
        tracing::debug!(
            ?kind,
            size,
            region = idx,
            path = %path.display(),
            quarantined,
            "disk backend alloc_region"
        );
        Ok(DevicePtr::from_region(idx, bytes))
    }

    fn memcpy(&self, src: DevicePtr, dst: DevicePtr, bytes: u64) -> anyhow::Result<()> {
        if bytes == 0 {
            return Ok(());
        }
        let n = usize::try_from(bytes).context("memcpy bytes")?;
        let mut inner = self.inner.lock();
        let (si, so) = Self::locate(&inner, src).context("disk memcpy: src")?;
        let (di, dof) = Self::locate(&inner, dst).context("disk memcpy: dst")?;
        if so + n > inner.regions[si].buffer.len() || dof + n > inner.regions[di].buffer.len() {
            bail!("disk memcpy: out-of-range copy {n} bytes");
        }
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
        let (i, off) = Self::locate(&inner, ptr).context("disk read_bytes")?;
        if off + len > inner.regions[i].buffer.len() {
            bail!("disk read_bytes: out-of-range {len} bytes");
        }
        Ok(inner.regions[i].buffer[off..off + len].to_vec())
    }

    fn fill_pattern(&self, ptr: DevicePtr, byte: u8, len: usize) -> anyhow::Result<()> {
        let mut inner = self.inner.lock();
        let (i, off) = Self::locate(&inner, ptr).context("disk fill_pattern")?;
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
        let (i, off) = Self::locate(&inner, ptr).context("disk write_bytes")?;
        if off + bytes.len() > inner.regions[i].buffer.len() {
            bail!("disk write_bytes: out-of-range write {} bytes", bytes.len());
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
        // Best-effort flush on drop. Errors are logged (Sprint 5.1) rather than silently
        // swallowed — a disk-full or permission error during shutdown is a real
        // operational signal.
        if Arc::strong_count(&self.inner) == 1 {
            if let Err(e) = self.flush_all() {
                tracing::error!(error = ?e, "disk backend: flush_all on Drop failed — data may be lost");
            }
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
    fn persistence_across_backend_reopens_with_matching_manifest() {
        let td = tempfile::tempdir().unwrap();
        let path = td.path().to_path_buf();
        let payload = vec![0x42u8; 512];

        {
            let be = DiskBackend::new(path.clone(), SwaCachingStrategy::Full).unwrap();
            let p = be.alloc_region(512, RegionKind::Primary).unwrap();
            be.write_bytes(p, &payload).unwrap();
            be.flush_all().unwrap();
        }

        // Reopen with the same root + allocate in the same order → manifest matches,
        // bytes recover.
        let be2 = DiskBackend::new(path, SwaCachingStrategy::Full).unwrap();
        let p2 = be2.alloc_region(512, RegionKind::Primary).unwrap();
        assert_eq!(be2.quarantined_regions(), 0);
        let out = be2.read_bytes(p2, 512).unwrap();
        assert_eq!(out, payload);
    }

    #[test]
    fn manifest_mismatch_quarantines_region() {
        let td = tempfile::tempdir().unwrap();
        let path = td.path().to_path_buf();

        {
            let be = DiskBackend::new(path.clone(), SwaCachingStrategy::Full).unwrap();
            let p = be.alloc_region(256, RegionKind::Primary).unwrap();
            be.write_bytes(p, &[0xCC; 256]).unwrap();
            be.flush_all().unwrap();
        }

        // Tamper with the region file out-of-band.
        let region_path = path.join("region-0000-primary.bin");
        let mut f = std::fs::OpenOptions::new()
            .write(true)
            .open(&region_path)
            .unwrap();
        f.write_all(&[0xEE; 256]).unwrap();
        f.sync_data().unwrap();
        drop(f);

        // Reopen: same alloc, but the bytes don't match the manifest's checksum.
        let be2 = DiskBackend::new(path, SwaCachingStrategy::Full).unwrap();
        let p2 = be2.alloc_region(256, RegionKind::Primary).unwrap();
        assert_eq!(
            be2.quarantined_regions(),
            1,
            "tampered region must quarantine"
        );
        // Quarantined region is zeroed in memory.
        let out = be2.read_bytes(p2, 256).unwrap();
        assert!(out.iter().all(|&b| b == 0));
    }

    #[test]
    fn kind_mismatch_quarantines_region() {
        let td = tempfile::tempdir().unwrap();
        let path = td.path().to_path_buf();

        {
            let be = DiskBackend::new(path.clone(), SwaCachingStrategy::Full).unwrap();
            let p = be.alloc_region(128, RegionKind::Primary).unwrap();
            be.write_bytes(p, &[0xAA; 128]).unwrap();
            be.flush_all().unwrap();
        }

        // Reopen and allocate a different RegionKind for index 0 → manifest disagrees.
        let be2 = DiskBackend::new(path, SwaCachingStrategy::Full).unwrap();
        let _ = be2.alloc_region(128, RegionKind::Rope).unwrap();
        assert_eq!(be2.quarantined_regions(), 1);
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
        be.fill_pattern(a, 0x00, 256).unwrap();
        let bytes_b = be.read_bytes(b, 256).unwrap();
        assert!(bytes_b.iter().all(|&v| v == 0x55));
    }

    #[test]
    fn allowed_roots_accepts_inside_prefix() {
        let td = tempfile::tempdir().unwrap();
        let sub = td.path().join("tenant-a");
        // Pre-create the parent directory so canonicalize sees it.
        std::fs::create_dir_all(&sub).unwrap();
        let allowed = vec![td.path().to_path_buf()];
        let be = DiskBackend::with_allowed_roots(sub, SwaCachingStrategy::Full, &allowed);
        assert!(be.is_ok(), "{be:?}");
    }

    #[test]
    fn allowed_roots_rejects_outside_prefix() {
        let td_a = tempfile::tempdir().unwrap();
        let td_b = tempfile::tempdir().unwrap();
        let allowed = vec![td_a.path().to_path_buf()];
        let be = DiskBackend::with_allowed_roots(
            td_b.path().to_path_buf(),
            SwaCachingStrategy::Full,
            &allowed,
        );
        assert!(be.is_err(), "path outside allowlist must be rejected");
    }

    #[cfg(unix)]
    #[test]
    fn unix_region_files_are_owner_only() {
        use std::os::unix::fs::PermissionsExt;
        let (be, _td) = DiskBackend::new_tempdir(SwaCachingStrategy::Full).unwrap();
        let _ = be.alloc_region(128, RegionKind::Primary).unwrap();
        be.flush_all().unwrap();
        // Look up region-0000-primary.bin and check its mode.
        let root = be.inner.lock().root.clone();
        let entry = std::fs::read_dir(&root)
            .unwrap()
            .filter_map(Result::ok)
            .find(|e| e.file_name().to_string_lossy().starts_with("region-"))
            .expect("region file");
        let mode = entry.metadata().unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600, "region file must be 0o600, got {mode:o}");
    }
}
