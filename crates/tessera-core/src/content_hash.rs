//! Content hashing.
//!
//! The block manager uses xxhash3 to address blocks by their `c_kv` content. xxhash3 is chosen
//! because it is non-cryptographic but extremely fast (>20 GB/s on a single core on modern
//! x86-64) and has good collision properties for the relatively small block sizes Tessera
//! operates on (~4 MB BF16 / ~2 MB FP8 per block).
//!
//! The [`ContentHasher`] trait is provided as a seam so a future on-device hash kernel can be
//! plugged in without touching `block_manager.rs`: the cost of `dtoh` for a 4 MB block on
//! every seal is non-trivial at production block-rates. ADR-0008 tracks this.
//!
//! **WS10 seam**: `hash_device` is a default method that falls back to `read_bytes + hash`.
//! A future `CudaXxh3Hasher` can override it with a kernel-side hash, eliminating the `dtoh`
//! round-trip documented as TD-002.

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
    /// Override this method with an on-device kernel (e.g. `CudaXxh3Hasher`) to eliminate
    /// the transfer entirely. The seam exists so no caller changes are required when the
    /// override lands. See ADR-0008 and TD-002.
    fn hash_device(
        &self,
        backend: &dyn DeviceBackend,
        ptr: DevicePtr,
        len: usize,
    ) -> Result<u64> {
        let bytes = backend.read_bytes(ptr, len)?;
        Ok(self.hash(&bytes))
    }
}

/// Default hasher: xxhash3 64-bit.
#[derive(Debug, Default, Clone, Copy)]
pub struct Xxh3Hasher;

/// CUDA stub hasher — Sprint 2 seam, Sprint 3 kernel. Feature-gated.
#[cfg(feature = "cuda")]
pub use crate::device::cuda::CudaXxh3Hasher;

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
