//! `usearch`-backed implementation of [`IndexBackend`].
//!
//! `usearch` is chosen over FAISS for one reason: it supports incremental `remove` without
//! rebuilding the index. FAISS HNSW requires a full rebuild on deletion, which is fundamentally
//! incompatible with the eviction pattern of a KV-cache block manager. See ADR-0005.

use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result};
use parking_lot::{Mutex, RwLock};
use usearch::{Index, IndexOptions, MetricKind, ScalarKind};

use crate::{IndexBackend, IndexMatch};

/// HNSW configuration for a [`UsearchIndex`]. Defaults match the playbook §6.1 numbers.
#[derive(Debug, Clone, Copy)]
pub struct UsearchConfig {
    /// Vector dimensionality (d_c — usually 512 for DeepSeek-V3).
    pub dimensions: usize,
    /// `M`: number of bidirectional links per node.
    pub connectivity: usize,
    /// `ef_construction`: build-time search depth.
    pub expansion_add: usize,
    /// `ef`: query-time search depth. Raise for recall, lower for latency.
    pub expansion_search: usize,
}

impl UsearchConfig {
    /// Defaults matching the playbook recommendation (M=32, efC=200, ef=64).
    pub const fn default_for_dim(dimensions: usize) -> Self {
        Self {
            dimensions,
            connectivity: 32,
            expansion_add: 200,
            expansion_search: 64,
        }
    }
}

/// `usearch`-backed segment index. Thread-safe.
pub struct UsearchIndex {
    index: Mutex<Index>,
    dimensions: usize,
    /// Monotonically incrementing label generator. We map `block_id -> label` so that
    /// `usearch::add(label, vec)` does not collide when the same block id is re-added.
    next_label: AtomicU64,
    label_for_block: RwLock<std::collections::HashMap<u32, u64>>,
    block_for_label: RwLock<std::collections::HashMap<u64, u32>>,
}

// `usearch::Index` does not implement `Debug` (the upstream crate omits it on purpose —
// the C++ shim holds raw FFI handles). Custom impl skips the inner index and reports
// only the metadata fields, which is all callers need.
impl std::fmt::Debug for UsearchIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UsearchIndex")
            .field("dimensions", &self.dimensions)
            .field("next_label", &self.next_label)
            .finish_non_exhaustive()
    }
}

impl UsearchIndex {
    /// Construct an empty index from a config.
    pub fn new(config: UsearchConfig) -> Result<Self> {
        let options = IndexOptions {
            dimensions: config.dimensions,
            metric: MetricKind::Cos,
            quantization: ScalarKind::F32,
            connectivity: config.connectivity,
            expansion_add: config.expansion_add,
            expansion_search: config.expansion_search,
            multi: false,
        };
        let index = Index::new(&options).context("failed to construct usearch index")?;
        // Reserve a modest initial capacity; grows automatically.
        index
            .reserve(1024)
            .context("failed to reserve usearch capacity")?;
        Ok(Self {
            index: Mutex::new(index),
            dimensions: config.dimensions,
            next_label: AtomicU64::new(1), // 0 is reserved as "no label"
            label_for_block: RwLock::new(std::collections::HashMap::new()),
            block_for_label: RwLock::new(std::collections::HashMap::new()),
        })
    }

    fn check_dim(&self, descriptor: &[f32]) -> Result<()> {
        if descriptor.len() != self.dimensions {
            anyhow::bail!(
                "descriptor length {} does not match index dimensions {}",
                descriptor.len(),
                self.dimensions
            );
        }
        Ok(())
    }
}

impl IndexBackend for UsearchIndex {
    fn add(&self, block_id: u32, descriptor: &[f32]) -> Result<()> {
        self.check_dim(descriptor)?;
        let label = self.next_label.fetch_add(1, Ordering::Relaxed);
        {
            let index = self.index.lock();
            // Grow if at capacity.
            if index.size() + 1 > index.capacity() {
                index.reserve(index.capacity().saturating_mul(2).max(1024))?;
            }
            index.add(label, descriptor).context("usearch add failed")?;
        }
        self.label_for_block.write().insert(block_id, label);
        self.block_for_label.write().insert(label, block_id);
        Ok(())
    }

    fn query(&self, descriptor: &[f32], k: usize) -> Result<Vec<IndexMatch>> {
        self.check_dim(descriptor)?;
        let matches = {
            let index = self.index.lock();
            if index.size() == 0 {
                return Ok(Vec::new());
            }
            index
                .search(descriptor, k)
                .context("usearch search failed")?
        };
        let block_for_label = self.block_for_label.read();
        let mut out = Vec::with_capacity(matches.keys.len());
        for (label, dist) in matches.keys.iter().zip(matches.distances.iter()) {
            if let Some(&block_id) = block_for_label.get(label) {
                // usearch's Cos metric returns DISTANCE = 1 - cosine_similarity. Convert.
                let similarity = 1.0 - *dist;
                out.push(IndexMatch {
                    block_id,
                    similarity,
                });
            }
        }
        Ok(out)
    }

    fn remove(&self, block_id: u32) -> Result<()> {
        let label = self.label_for_block.write().remove(&block_id);
        if let Some(label) = label {
            self.block_for_label.write().remove(&label);
            let index = self.index.lock();
            // usearch `remove` returns the number of items removed; treat 0 as a no-op.
            let _ = index.remove(label);
        }
        Ok(())
    }

    fn len(&self) -> usize {
        self.label_for_block.read().len()
    }

    fn name(&self) -> &'static str {
        "usearch"
    }
}

#[cfg(test)]
mod tests {
    use rand::Rng;
    use rand::SeedableRng;

    use super::*;

    fn rand_vec(rng: &mut impl Rng, dim: usize) -> Vec<f32> {
        (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect()
    }

    #[test]
    fn add_then_query_returns_inserted_block() {
        let idx = UsearchIndex::new(UsearchConfig::default_for_dim(16)).unwrap();
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0xC0FFEE);
        let v = rand_vec(&mut rng, 16);
        idx.add(42, &v).unwrap();
        let results = idx.query(&v, 1).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].block_id, 42);
        assert!(
            results[0].similarity > 0.99,
            "similarity should be ~1 for the inserted vector"
        );
    }

    #[test]
    fn remove_drops_block() {
        let idx = UsearchIndex::new(UsearchConfig::default_for_dim(16)).unwrap();
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1);
        let v = rand_vec(&mut rng, 16);
        idx.add(7, &v).unwrap();
        assert_eq!(idx.len(), 1);
        idx.remove(7).unwrap();
        assert_eq!(idx.len(), 0);
        // Removing again is a no-op.
        idx.remove(7).unwrap();
    }

    #[test]
    fn empty_index_returns_no_matches() {
        let idx = UsearchIndex::new(UsearchConfig::default_for_dim(16)).unwrap();
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(2);
        let q = rand_vec(&mut rng, 16);
        assert!(idx.query(&q, 5).unwrap().is_empty());
    }
}
