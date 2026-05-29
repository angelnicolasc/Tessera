//! PyO3 module `tessera._native`.
//!
//! Exposes the Rust core to Python in a deliberately small surface: the Python layer wraps
//! these primitives with higher-level orchestration (segment index pipeline, vLLM plugin,
//! observability) so that the Rust ↔ Python boundary stays narrow and stable.

#![allow(non_local_definitions)] // PyO3 generates impl blocks inside the macros
#![warn(missing_docs)]

use std::sync::Arc;
use std::time::Duration;

use numpy::PyReadonlyArray1;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyModule;
use tessera_core::{
    block::BlockId, transport::MockTransport, transport::RankTransport, CkvDtype,
    CompressionScheme, CrossAgentShareTable, MlaBlockConfig, NodeId, RankId, TesseraBlockManager,
    TesseraError, TokenRange, Topology, World,
};
use tessera_index::{DistributedSegmentIndex, IndexBackend, UsearchConfig, UsearchIndex};

/// Shared tokio runtime used to drive async Rust APIs from the synchronous Python boundary.
/// One process-wide multithread runtime keeps the cost of `block_on` calls amortised across
/// many invocations. See `docs/src/adr/0014-multi-rank-architecture.md`.
static TOKIO_RT: std::sync::LazyLock<tokio::runtime::Runtime> = std::sync::LazyLock::new(|| {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .worker_threads(2)
        .thread_name("tessera-py-rt")
        .build()
        .expect("failed to build tessera tokio runtime")
});

// ------------------------------ error glue --------------------------------

fn map_err(e: TesseraError) -> PyErr {
    match e {
        TesseraError::OutOfBlocks { used, total } => PyRuntimeError::new_err(format!(
            "tessera: out of blocks (used={used}, total={total})"
        )),
        TesseraError::InvalidConfig(msg) => PyValueError::new_err(format!("tessera: {msg}")),
        TesseraError::UnknownBlock(id) => {
            PyValueError::new_err(format!("tessera: unknown block id {id}"))
        }
        TesseraError::Backend(err) => PyRuntimeError::new_err(format!("tessera backend: {err}")),
    }
}

fn map_anyhow(e: anyhow::Error) -> PyErr {
    PyRuntimeError::new_err(format!("{e:?}"))
}

// ------------------------------ CkvDtype ----------------------------------

/// Python wrapper for [`tessera_core::CkvDtype`].
#[pyclass(name = "CkvDtype", eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyCkvDtype {
    /// BF16 storage.
    Bf16,
    /// FP8 E4M3 storage (requires calibration).
    Fp8E4m3,
    /// FP4 E2M1 storage (Sprint 5 / V4 Lightning Indexer).
    Fp4E2m1,
    /// Mixed BF16+FP8+FP4 layout, scheme-driven (Sprint 5 / V4 hybrid).
    MixedBf16Fp8Fp4,
}

impl From<PyCkvDtype> for CkvDtype {
    fn from(d: PyCkvDtype) -> Self {
        match d {
            PyCkvDtype::Bf16 => Self::Bf16,
            PyCkvDtype::Fp8E4m3 => Self::Fp8E4m3,
            PyCkvDtype::Fp4E2m1 => Self::Fp4E2m1,
            PyCkvDtype::MixedBf16Fp8Fp4 => Self::MixedBf16Fp8Fp4,
        }
    }
}

#[pymethods]
impl PyCkvDtype {
    fn __repr__(&self) -> &'static str {
        match self {
            PyCkvDtype::Bf16 => "CkvDtype.Bf16",
            PyCkvDtype::Fp8E4m3 => "CkvDtype.Fp8E4m3",
            PyCkvDtype::Fp4E2m1 => "CkvDtype.Fp4E2m1",
            PyCkvDtype::MixedBf16Fp8Fp4 => "CkvDtype.MixedBf16Fp8Fp4",
        }
    }
}

// ------------------------------ CompressionScheme -------------------------

/// Python wrapper for [`tessera_core::CompressionScheme`].
#[pyclass(name = "CompressionScheme")]
#[derive(Clone)]
pub struct PyCompressionScheme {
    inner: CompressionScheme,
}

#[pymethods]
impl PyCompressionScheme {
    /// Construct an MLA latent scheme (standard DeepSeek MLA).
    #[staticmethod]
    fn mla_latent(latent_dim: u32, rope_key_dim: u32) -> Self {
        Self {
            inner: CompressionScheme::MlaLatent {
                latent_dim,
                rope_key_dim,
            },
        }
    }

    /// Construct an MHA fallback scheme (no compression).
    #[staticmethod]
    fn mha_full(num_heads: u32, head_dim: u32) -> Self {
        Self {
            inner: CompressionScheme::MhaFull {
                num_heads,
                head_dim,
            },
        }
    }

    /// **Deprecated (Sprint 5)**: DeepSeek-V4's real architecture is CSA + HCA + SWA. Use
    /// `v4_csa`, `v4_hca`, `v4_swa` instead. Construction is still permitted for legacy
    /// callers; `MlaBlockConfig` validation now rejects it. See ADR-0020.
    #[staticmethod]
    fn dsa_hierarchical(coarse_dim: u32, fine_dim: u32, swa_window: u32) -> Self {
        #[allow(deprecated)]
        Self {
            inner: CompressionScheme::DsaHierarchical {
                coarse_dim,
                fine_dim,
                swa_window,
            },
        }
    }

    /// **Sprint 5 / V4** — Compressed Sparse Attention layer. Per the V4 paper §2.3.1.
    ///
    /// Args:
    ///     k1: token compression ratio (4 in both V4 models).
    ///     head_dim: per-head dimension (512).
    ///     num_heads: total query heads (64 Flash / 128 Pro).
    ///     rope_dim: trailing RoPE BF16 dimensions (64).
    ///     indexer_head_dim: Lightning Indexer head dimension (128).
    ///     num_indexer_heads: indexer query heads (64).
    ///     top_k: sparse-selection top-k (512 Flash / 1024 Pro).
    #[staticmethod]
    #[pyo3(signature = (k1, head_dim, num_heads, rope_dim, indexer_head_dim, num_indexer_heads, top_k))]
    fn v4_csa(
        k1: u32,
        head_dim: u32,
        num_heads: u32,
        rope_dim: u32,
        indexer_head_dim: u32,
        num_indexer_heads: u32,
        top_k: u32,
    ) -> Self {
        Self {
            inner: CompressionScheme::V4Csa {
                k1,
                head_dim,
                num_heads,
                rope_dim,
                indexer_head_dim,
                num_indexer_heads,
                top_k,
            },
        }
    }

    /// **Sprint 5 / V4** — Heavily Compressed Attention layer. Per the V4 paper §2.3.2.
    #[staticmethod]
    fn v4_hca(k2: u32, head_dim: u32, num_heads: u32, rope_dim: u32) -> Self {
        Self {
            inner: CompressionScheme::V4Hca {
                k2,
                head_dim,
                num_heads,
                rope_dim,
            },
        }
    }

    /// **Sprint 5 / V4** — Sliding Window Attention branch / pure-SWA layer. Per §2.3.3.
    #[staticmethod]
    fn v4_swa(window: u32, head_dim: u32, num_heads: u32, rope_dim: u32) -> Self {
        Self {
            inner: CompressionScheme::V4Swa {
                window,
                head_dim,
                num_heads,
                rope_dim,
            },
        }
    }

    /// Optional MLA latent dim.
    fn mla_latent_dim(&self) -> Option<u32> {
        self.inner.mla_latent_dim()
    }

    /// Optional MLA rope key dim.
    fn mla_rope_key_dim(&self) -> Option<u32> {
        self.inner.mla_rope_key_dim()
    }

    /// Whether the scheme is in the V4 hybrid family.
    fn is_v4(&self) -> bool {
        self.inner.is_v4()
    }

    /// Self-describing per-token-per-layer byte cost (V4 schemes only; 0 otherwise).
    fn bytes_per_token_per_layer(&self) -> u64 {
        self.inner.bytes_per_token_per_layer()
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }
}

// ------------------------------ MlaBlockConfig ----------------------------

/// Python wrapper for [`tessera_core::MlaBlockConfig`].
#[pyclass(name = "MlaBlockConfig")]
#[derive(Clone)]
pub struct PyMlaBlockConfig {
    inner: MlaBlockConfig,
}

#[pymethods]
impl PyMlaBlockConfig {
    #[new]
    fn new(
        scheme: PyCompressionScheme,
        num_layers: u32,
        block_size_tokens: u32,
        ckv_dtype: PyCkvDtype,
        device: i32,
    ) -> PyResult<Self> {
        let inner = MlaBlockConfig::new(
            scheme.inner,
            num_layers,
            block_size_tokens,
            ckv_dtype.into(),
            device,
        )
        .map_err(map_err)?;
        Ok(Self { inner })
    }

    #[getter]
    fn num_layers(&self) -> u32 {
        self.inner.num_layers
    }

    #[getter]
    fn block_size_tokens(&self) -> u32 {
        self.inner.block_size_tokens
    }

    fn primary_block_bytes(&self) -> u64 {
        self.inner.primary_block_bytes()
    }

    fn rope_block_bytes(&self) -> u64 {
        self.inner.rope_block_bytes()
    }

    fn fp8_scale_block_bytes(&self) -> u64 {
        self.inner.fp8_scale_block_bytes()
    }

    fn total_block_bytes(&self) -> u64 {
        self.inner.total_block_bytes()
    }

    fn compression_ratio_vs_mha_bf16(&self) -> f64 {
        self.inner.compression_ratio_vs_mha_bf16()
    }

    /// **Sprint 5 / V4** — Per-layer scheme resolution. Returns the scheme for the given
    /// layer index. For homogeneous configs returns the primary scheme.
    fn scheme_for_layer(&self, layer_idx: u32) -> PyCompressionScheme {
        PyCompressionScheme {
            inner: self.inner.scheme_for_layer(layer_idx),
        }
    }

    /// Whether this config carries an explicit per-layer schemes vector.
    fn has_per_layer_schemes(&self) -> bool {
        self.inner.has_per_layer_schemes()
    }

    /// `lcm(k1, k2)` over the active V4 schemes. Returns 1 when no V4 layer present.
    fn v4_block_size_lcm(&self) -> u32 {
        self.inner.v4_block_size_lcm()
    }

    /// **Sprint 5 / V4** — Construct a per-layer hybrid config from an ordered Python list
    /// of `CompressionScheme` (one per transformer layer). The list length determines
    /// `num_layers`; `block_size_tokens` must be a multiple of every layer's k.
    #[staticmethod]
    fn with_per_layer_schemes(
        schemes: Vec<PyCompressionScheme>,
        block_size_tokens: u32,
        ckv_dtype: PyCkvDtype,
        device: i32,
    ) -> PyResult<Self> {
        let raw: Vec<CompressionScheme> = schemes.into_iter().map(|s| s.inner).collect();
        let inner = MlaBlockConfig::with_per_layer_schemes(
            raw,
            block_size_tokens,
            ckv_dtype.into(),
            device,
        )
        .map_err(map_err)?;
        Ok(Self { inner })
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }
}

// ------------------------------ BlockManager ------------------------------

/// CPU-mock-backed block manager. The CUDA-backed variant lives in `tessera._native` under
/// `BlockManagerCuda` when the crate is built with `--features cuda`.
///
/// `inner` is `Arc<...>` so cross-rank adapters (`MockTransport` peer wiring) can hold a
/// shared handle without forcing PyO3 to materialise a clone of the underlying state.
#[pyclass(name = "BlockManager")]
pub struct PyBlockManager {
    inner: Arc<TesseraBlockManager>,
}

#[pymethods]
impl PyBlockManager {
    #[new]
    fn new(config: PyMlaBlockConfig, memory_bytes: u64) -> PyResult<Self> {
        let inner = TesseraBlockManager::new(config.inner, memory_bytes).map_err(map_err)?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    /// Multi-rank constructor. `world` is a [`PyWorld`] describing the surrounding
    /// deployment topology and `rank` is this process's id within it. Equivalent to
    /// `BlockManager(config, memory_bytes)` when `world.size == 1`.
    #[staticmethod]
    fn with_world(
        config: PyMlaBlockConfig,
        memory_bytes: u64,
        rank: u32,
        world: &PyWorld,
    ) -> PyResult<Self> {
        let inner = TesseraBlockManager::new_with_world(
            config.inner,
            memory_bytes,
            RankId(rank),
            world.inner.clone(),
        )
        .map_err(map_err)?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    fn allocate(&self, req_id: u64, token_start: u32, token_end: u32) -> PyResult<u32> {
        self.inner
            .allocate(req_id, TokenRange::new(token_start, token_end))
            .map(BlockId::raw)
            .map_err(map_err)
    }

    fn seal(&self, block_id: u32) -> PyResult<(u32, u64, bool)> {
        let out = self.inner.seal(BlockId(block_id)).map_err(map_err)?;
        Ok((out.canonical_block.raw(), out.content_hash, out.was_dedup))
    }

    fn cow_fork(&self, block_id: u32, new_req_id: u64) -> PyResult<u32> {
        self.inner
            .cow_fork(BlockId(block_id), new_req_id)
            .map(BlockId::raw)
            .map_err(map_err)
    }

    fn free(&self, block_id: u32) -> PyResult<()> {
        self.inner.free(BlockId(block_id)).map_err(map_err)
    }

    /// Free all private blocks owned by `req_id` and return the count freed.
    /// This is the per-request teardown path (ADR-0009 / WS1).
    fn release_request(&self, req_id: u64) -> u32 {
        self.inner.release_request(req_id)
    }

    fn increment_ref(&self, block_id: u32) -> PyResult<u32> {
        self.inner.increment_ref(BlockId(block_id)).map_err(map_err)
    }

    fn fill_primary_test_pattern(&self, block_id: u32, byte: u8) -> PyResult<()> {
        self.inner
            .fill_primary_test_pattern(BlockId(block_id), byte)
            .map_err(map_err)
    }

    #[getter]
    fn used_blocks(&self) -> u32 {
        self.inner.used_blocks()
    }

    #[getter]
    fn total_blocks(&self) -> u32 {
        self.inner.total_blocks()
    }

    fn utilization(&self) -> f64 {
        self.inner.utilization()
    }

    /// Return the raw device pointer (as `usize`) to the FP8 scale region for `block_id`.
    /// Returns `None` if FP8 is not active for this config or the block is unknown.
    fn fp8_scales_ptr(&self, block_id: u32) -> Option<usize> {
        self.inner
            .fp8_scales_ptr(BlockId(block_id))
            .map(|ptr| ptr.raw)
    }

    /// This manager's rank within its world (Sprint 3 / ADR-0014). For singleton-world
    /// managers this is always 0.
    #[getter]
    fn rank(&self) -> u32 {
        self.inner.rank().raw()
    }

    /// World size this manager belongs to. Singleton worlds report 1.
    #[getter]
    fn world_size(&self) -> u32 {
        self.inner.world().size
    }

    /// Lift a local `block_id` into a global `(rank, block_id)` tuple. Used by callers that
    /// span multiple ranks (distributed segment index, share table).
    fn global_id(&self, block_id: u32) -> (u32, u32) {
        let g = self.inner.global_id(BlockId(block_id));
        (g.rank.raw(), g.block.raw())
    }

    /// PD-disaggregation: transfer every private block owned by `req_id` to rank `target`
    /// using `transport`. Returns the number of blocks transferred. See ADR-0016.
    fn transfer_request_to_rank(
        &self,
        req_id: u64,
        target: u32,
        transport: &PyMockTransport,
    ) -> PyResult<u32> {
        let dyn_transport = transport.inner.clone();
        TOKIO_RT
            .block_on(
                self.inner
                    .transfer_request_to_rank(req_id, RankId(target), &dyn_transport),
            )
            .map_err(map_err)
    }

    fn __repr__(&self) -> String {
        format!(
            "BlockManager(used={}, total={}, util={:.2}%)",
            self.inner.used_blocks(),
            self.inner.total_blocks(),
            self.inner.utilization() * 100.0,
        )
    }

    fn __len__(&self) -> usize {
        self.inner.used_blocks() as usize
    }

    fn __sizeof__(&self) -> usize {
        std::mem::size_of_val(&self.inner)
    }
}

// ------------------------------ ShareTable --------------------------------

/// Python wrapper for [`CrossAgentShareTable`].
#[pyclass(name = "ShareTable")]
pub struct PyShareTable {
    inner: CrossAgentShareTable,
}

#[pymethods]
impl PyShareTable {
    #[new]
    fn new() -> Self {
        Self {
            inner: CrossAgentShareTable::new(),
        }
    }

    fn add_share(&self, req_id: u64, block_id: u32) {
        self.inner.add_share(req_id, BlockId(block_id));
    }

    fn release_request(&self, req_id: u64) -> Vec<u32> {
        self.inner
            .release_request(req_id)
            .into_iter()
            .map(BlockId::raw)
            .collect()
    }

    fn owners(&self, block_id: u32) -> Option<Vec<u64>> {
        self.inner.owners(BlockId(block_id))
    }

    fn shared_block_count(&self) -> usize {
        self.inner.shared_block_count()
    }

    fn sharing_rate(&self) -> f64 {
        self.inner.sharing_rate()
    }

    fn __repr__(&self) -> String {
        format!(
            "ShareTable(shared_blocks={}, sharing_rate={:.4})",
            self.inner.shared_block_count(),
            self.inner.sharing_rate(),
        )
    }

    fn __len__(&self) -> usize {
        self.inner.shared_block_count()
    }
}

// ------------------------------ UsearchIndex ------------------------------

/// Python wrapper for [`tessera_index::UsearchIndex`].
#[pyclass(name = "UsearchIndex")]
pub struct PyUsearchIndex {
    inner: UsearchIndex,
}

#[pymethods]
impl PyUsearchIndex {
    #[new]
    #[pyo3(signature = (dimensions, connectivity=32, expansion_add=200, expansion_search=64))]
    fn new(
        dimensions: usize,
        connectivity: usize,
        expansion_add: usize,
        expansion_search: usize,
    ) -> PyResult<Self> {
        let cfg = UsearchConfig {
            dimensions,
            connectivity,
            expansion_add,
            expansion_search,
        };
        let inner = UsearchIndex::new(cfg).map_err(map_anyhow)?;
        Ok(Self { inner })
    }

    fn add(&self, block_id: u32, descriptor: PyReadonlyArray1<'_, f32>) -> PyResult<()> {
        let slice = descriptor
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("{e}")))?;
        self.inner.add(block_id, slice).map_err(map_anyhow)
    }

    fn query(&self, descriptor: PyReadonlyArray1<'_, f32>, k: usize) -> PyResult<Vec<(u32, f32)>> {
        let slice = descriptor
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("{e}")))?;
        let matches = self.inner.query(slice, k).map_err(map_anyhow)?;
        Ok(matches
            .into_iter()
            .map(|m| (m.block_id, m.similarity))
            .collect())
    }

    fn remove(&self, block_id: u32) -> PyResult<()> {
        self.inner.remove(block_id).map_err(map_anyhow)
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn name(&self) -> &'static str {
        self.inner.name()
    }

    fn __repr__(&self) -> String {
        format!(
            "UsearchIndex(len={}, name={})",
            self.inner.len(),
            self.inner.name(),
        )
    }
}

// ────────────────────── Sprint 3: rank / world / transport ────────────────

/// Rank within a multi-rank world. Lightweight newtype around `u32`.
#[pyclass(name = "RankId")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PyRankId(pub u32);

#[pymethods]
impl PyRankId {
    #[new]
    fn new(value: u32) -> Self {
        Self(value)
    }

    #[getter]
    fn value(&self) -> u32 {
        self.0
    }

    fn __repr__(&self) -> String {
        format!("RankId({})", self.0)
    }

    fn __int__(&self) -> u32 {
        self.0
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.0 == other.0
    }

    fn __hash__(&self) -> u64 {
        u64::from(self.0)
    }
}

/// Deployment topology description.
#[pyclass(name = "World")]
#[derive(Clone)]
pub struct PyWorld {
    inner: Arc<World>,
}

#[pymethods]
impl PyWorld {
    /// Single-node, multi-rank world (NVLink P2P territory). `local` ∈ `[0, size)`.
    #[staticmethod]
    fn single_node(local: u32, size: u32) -> PyResult<Self> {
        let w = World::new(RankId(local), size, Topology::SingleNode)
            .ok_or_else(|| PyValueError::new_err("invalid (local, size) for World::single_node"))?;
        Ok(Self { inner: Arc::new(w) })
    }

    /// Convenience: world of size 1, rank 0. Equivalent to `World::singleton()` in Rust.
    #[staticmethod]
    fn singleton() -> Self {
        Self {
            inner: Arc::new(World::singleton()),
        }
    }

    /// Multi-node world (NCCL territory; Sprint 4 runtime impl). `node_of[i]` is the
    /// node id hosting rank `i`. Length must equal `size`.
    #[staticmethod]
    fn multi_node(local: u32, size: u32, node_of: Vec<u32>) -> PyResult<Self> {
        let node_of: Vec<NodeId> = node_of.into_iter().map(NodeId).collect();
        let w =
            World::new(RankId(local), size, Topology::MultiNode { node_of }).ok_or_else(|| {
                PyValueError::new_err("invalid (local, size, node_of) for World::multi_node")
            })?;
        Ok(Self { inner: Arc::new(w) })
    }

    #[getter]
    fn local(&self) -> u32 {
        self.inner.local.raw()
    }

    #[getter]
    fn size(&self) -> u32 {
        self.inner.size
    }

    fn is_multi_node(&self) -> bool {
        self.inner.topology.is_multi_node()
    }

    fn has_peers(&self) -> bool {
        self.inner.has_peers()
    }

    fn peers(&self) -> Vec<u32> {
        self.inner.peers().map(RankId::raw).collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "World(local=r{}, size={}, topology={})",
            self.inner.local.raw(),
            self.inner.size,
            if self.inner.topology.is_multi_node() {
                "MultiNode"
            } else {
                "SingleNode"
            }
        )
    }
}

/// Adapter that lets a `MockTransport` peer slot be fulfilled by a [`PyBlockManager`]. Used
/// by tests + the `MultiRankCoordinator` in `python/tessera/multi_rank.py`.
struct BlockManagerPeerAdapter {
    manager: Arc<TesseraBlockManager>,
    accept_req_id: u64,
    token_range: TokenRange,
}

impl tessera_core::transport::mock::MockPeer for BlockManagerPeerAdapter {
    fn provide_block(
        &self,
        block_id: BlockId,
    ) -> anyhow::Result<tessera_core::transport::BlockPayload> {
        self.manager
            .export_payload(block_id)
            .map_err(|e| anyhow::anyhow!("export_payload({block_id:?}): {e}"))
    }

    fn accept_pushed(
        &self,
        payload: tessera_core::transport::BlockPayload,
    ) -> anyhow::Result<BlockId> {
        self.manager
            .import_payload(self.accept_req_id, self.token_range, &payload)
            .map_err(|e| anyhow::anyhow!("import_payload: {e}"))
    }

    fn lookup_hash(&self, _content_hash: u64) -> anyhow::Result<Option<BlockId>> {
        // Content-hash → block_id lookup lives in Python's SegmentIndex (Layer 1). We don't
        // duplicate it here. Returning None means "ask elsewhere".
        Ok(None)
    }

    fn reserve(
        &self,
        req_id: u64,
        count: u32,
    ) -> anyhow::Result<tessera_core::transport::ReservationToken> {
        // Forward to the destination's block manager so the reservation enforces real
        // capacity. Returning a non-zero token instructs the MockTransport to use this
        // value verbatim instead of minting one from its internal counter (see
        // mock.rs::reserve_slots).
        self.manager
            .reserve_incoming(req_id, count)
            .map_err(|e| anyhow::anyhow!("reserve_incoming: {e}"))
    }

    fn release_reservation(
        &self,
        token: tessera_core::transport::ReservationToken,
    ) -> anyhow::Result<()> {
        self.manager
            .release_reservation_local(token)
            .map_err(|e| anyhow::anyhow!("release_reservation_local: {e}"))
    }
}

/// In-process mock transport. Construct sets of N interconnected handles via
/// [`PyMockTransport::new_world`]; wire each handle's peer to a block manager via
/// [`PyMockTransport::register_block_manager_peer`].
#[pyclass(name = "MockTransport")]
pub struct PyMockTransport {
    /// Underlying `Arc<dyn RankTransport>` (the trait object form so we can pass it to
    /// `transfer_request_to_rank`).
    inner: Arc<dyn RankTransport>,
    /// Concrete handle kept separately so `register_block_manager_peer` and `events`
    /// remain accessible without trait downcasts.
    mock: MockTransport,
}

#[pymethods]
impl PyMockTransport {
    /// Construct N interconnected mock transports, one per rank. Returns them in rank order.
    #[staticmethod]
    fn new_world(size: u32) -> Vec<Self> {
        MockTransport::new_world(size)
            .into_iter()
            .map(|m| {
                let arc: Arc<dyn RankTransport> = Arc::new(m.clone());
                Self {
                    inner: arc,
                    mock: m,
                }
            })
            .collect()
    }

    /// Singleton mock transport (size=1, no peers). Useful for retrocompat.
    #[staticmethod]
    fn singleton() -> Self {
        let m = MockTransport::singleton();
        let arc: Arc<dyn RankTransport> = Arc::new(m.clone());
        Self {
            inner: arc,
            mock: m,
        }
    }

    /// Wire a `BlockManager` as the peer answering for `rank` on this transport handle. The
    /// adapter uses `accept_req_id` as the destination-side request id when blocks arrive
    /// via `push_block`. The destination's token range is currently fixed at `[0, 64)` for
    /// Sprint 3; this matches FlashMLA's block size and is sufficient for the multirank
    /// tests.
    fn register_block_manager_peer(&self, rank: u32, manager: &PyBlockManager, accept_req_id: u64) {
        let adapter = BlockManagerPeerAdapter {
            manager: Arc::clone(&manager.inner),
            accept_req_id,
            token_range: TokenRange::new(0, 64),
        };
        self.mock.register_peer(RankId(rank), Arc::new(adapter));
    }

    /// Snapshot the event log (one entry per cross-rank op invoked on this handle).
    fn events(&self) -> Vec<String> {
        self.mock
            .events()
            .into_iter()
            .map(|e| format!("{e:?}"))
            .collect()
    }

    /// Number of events recorded on this handle since construction or last `clear_events`.
    fn event_count(&self) -> usize {
        self.mock.event_count()
    }

    /// Clear the event log.
    fn clear_events(&self) {
        self.mock.clear_events();
    }

    /// Local rank assigned to this handle.
    #[getter]
    fn local(&self) -> u32 {
        self.mock.local().raw()
    }

    fn name(&self) -> &'static str {
        self.inner.name()
    }

    fn __repr__(&self) -> String {
        format!(
            "MockTransport(local=r{}, events={})",
            self.mock.local().raw(),
            self.mock.event_count(),
        )
    }
}

/// Distributed segment index — fan-out across ranks for content-hash lookups. Sprint 3 ships
/// the hash-only path; descriptor-similarity fan-out is Sprint 4.
#[pyclass(name = "DistributedSegmentIndex")]
pub struct PyDistributedSegmentIndex {
    inner: DistributedSegmentIndex,
}

#[pymethods]
impl PyDistributedSegmentIndex {
    /// Construct a distributed index. `dimensions` configures the local `UsearchIndex`.
    /// `budget_us` is the total wall-clock budget for fan-out lookups (exceeded budget →
    /// safe miss returning `None`).
    #[new]
    #[pyo3(signature = (dimensions, world, transport, budget_us=1000))]
    fn new(
        dimensions: usize,
        world: &PyWorld,
        transport: &PyMockTransport,
        budget_us: u64,
    ) -> PyResult<Self> {
        let local: Arc<dyn IndexBackend> = Arc::new(
            UsearchIndex::new(UsearchConfig::default_for_dim(dimensions)).map_err(map_anyhow)?,
        );
        let inner = DistributedSegmentIndex::new(
            local,
            world.inner.clone(),
            transport.inner.clone(),
            Duration::from_micros(budget_us),
        );
        Ok(Self { inner })
    }

    /// Look up a content hash. Returns `(rank, block_id)` if any peer holds the hash,
    /// otherwise `None`. Always safe to call; a miss simply means the caller computes the
    /// block fresh.
    fn lookup_hash(&self, py: Python<'_>, content_hash: u64) -> PyResult<Option<(u32, u32)>> {
        py.allow_threads(|| {
            TOKIO_RT
                .block_on(self.inner.lookup_hash(content_hash))
                .map_err(map_anyhow)
                .map(|opt| opt.map(|hit| (hit.global.rank.raw(), hit.global.block.raw())))
        })
    }

    /// Record a local hit (the caller's local Layer 1 resolved the query without fan-out).
    /// Bumps the `tessera_distributed_index_local_hits_total` counter.
    fn record_local_hit(&self) {
        self.inner.record_local_hit();
    }
}

// ------------------------------ metrics snapshot --------------------------

#[pyfunction]
fn metrics_snapshot_text() -> String {
    tessera_core::metrics::snapshot_text()
}

// ------------------------------ module ------------------------------------

/// Native bindings for the Tessera Rust core. Imported by `python/tessera/__init__.py`.
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCkvDtype>()?;
    m.add_class::<PyCompressionScheme>()?;
    m.add_class::<PyMlaBlockConfig>()?;
    m.add_class::<PyBlockManager>()?;
    m.add_class::<PyShareTable>()?;
    m.add_class::<PyUsearchIndex>()?;
    // Sprint 3: multi-rank surface.
    m.add_class::<PyRankId>()?;
    m.add_class::<PyWorld>()?;
    m.add_class::<PyMockTransport>()?;
    m.add_class::<PyDistributedSegmentIndex>()?;
    m.add_function(wrap_pyfunction!(metrics_snapshot_text, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add(
        "REQUIRED_BLOCK_SIZE_TOKENS",
        tessera_core::config::REQUIRED_BLOCK_SIZE_TOKENS,
    )?;
    Ok(())
}
