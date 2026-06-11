//! Content hashing.
//!
//! The block manager uses xxhash3 to address blocks by their `c_kv` content. xxhash3 is chosen
//! because it is non-cryptographic but extremely fast (>20 GB/s on a single core on modern
//! x86-64) and has good collision properties on random inputs.
//!
//! **Sprint 5.1 hardening note**: xxh3 is **not collision-resistant against adversaries** —
//! crafted-collision attacks are well-known. Cross-agent dedup (`TesseraBlockManager::seal`)
//! therefore verifies bytes after a hash match before deduping; the hash only fast-paths the
//! lookup, it does not authorise the dedup. See ADR-0026 and `block_manager::seal`.
//!
//! The [`ContentHasher`] trait is provided as a seam so a future on-device hash kernel can be
//! plugged in without touching `block_manager.rs`: the cost of `dtoh` for a 4 MB block on
//! every seal is non-trivial at production block-rates. ADR-0008 tracks this.
//!
//! **WS10 seam**: `hash_device` is a default method that falls back to `read_bytes + hash`.
//! A future on-device xxh3 kernel can override it without changing call sites (TD-002).

use anyhow::Result;
use xxhash_rust::xxh3::xxh3_64;

use crate::device::{DeviceBackend, DevicePtr};

/// Abstraction over content hashers. Implementations must be deterministic across runs and
/// across machines for a given byte sequence — otherwise content-addressed dedup breaks.
pub trait ContentHasher: Send + Sync {
    /// Hash a byte slice.
    fn hash(&self, bytes: &[u8]) -> u64;

    /// Implementation name for diagnostics and metrics labels.
    fn name(&self) -> &'static str;

    /// Hash data at a device pointer. The default implementation reads bytes to host memory
    /// then hashes them — O(block_size) `dtoh` transfer on every `seal()` call.
    ///
    /// Override this method with an on-device kernel to eliminate the transfer entirely. The
    /// seam exists so no caller changes are required when the override lands. See ADR-0008
    /// and TD-002.
    fn hash_device(&self, backend: &dyn DeviceBackend, ptr: DevicePtr, len: usize) -> Result<u64> {
        let bytes = backend.read_bytes(ptr, len)?;
        Ok(self.hash(&bytes))
    }
}

/// Default hasher: xxhash3 64-bit.
#[derive(Debug, Default, Clone, Copy)]
pub struct Xxh3Hasher;

// Sprint 5.1: `CudaXxh3HasherStub` is no longer re-exported from this module. It is
// available behind `feature = "cuda"` as `crate::device::cuda::CudaXxh3HasherStub`,
// marked `#[doc(hidden)]` and `#[deprecated]`, and constructing it requires the
// `new_acknowledging_stub_collision_risk` constructor. See ADR-0026.

impl ContentHasher for Xxh3Hasher {
    fn hash(&self, bytes: &[u8]) -> u64 {
        xxh3_64(bytes)
    }

    fn name(&self) -> &'static str {
        "xxh3"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xxh3_is_deterministic() {
        let h = Xxh3Hasher;
        let data = b"the quick brown fox";
        assert_eq!(h.hash(data), h.hash(data));
    }

    #[test]
    fn different_inputs_differ() {
        let h = Xxh3Hasher;
        assert_ne!(h.hash(b"abc"), h.hash(b"abd"));
    }
}
