//! Basic tests for the `rank` module + the rank-aware constructor on `TesseraBlockManager`.
//!
//! These tests run on the CPU mock backend; no GPU required.

use std::sync::Arc;

use tessera_core::{
    rank::{NodeId, RankId, Topology, World},
    CkvDtype, CompressionScheme, GlobalBlockId, MlaBlockConfig, TesseraBlockManager, TokenRange,
};

fn small_cfg() -> MlaBlockConfig {
    MlaBlockConfig::new(
        CompressionScheme::MlaLatent { latent_dim: 32, rope_key_dim: 8 },
        4,
        64,
        CkvDtype::Bf16,
        0,
    )
    .unwrap()
}

#[test]
fn singleton_world_matches_legacy_new() {
    // The Sprint-0 `new()` path is preserved; it constructs a singleton-world manager.
    let mgr = TesseraBlockManager::new(small_cfg(), 16 * 1024 * 1024).unwrap();
    assert_eq!(mgr.rank(), RankId::ZERO);
    assert_eq!(mgr.world().size, 1);
    assert!(matches!(mgr.world().topology, Topology::SingleNode));
}

#[test]
fn new_with_world_stores_rank_and_world() {
    let world = Arc::new(World::new(RankId(2), 4, Topology::SingleNode).unwrap());
    let mgr = TesseraBlockManager::new_with_world(
        small_cfg(),
        16 * 1024 * 1024,
        RankId(2),
        Arc::clone(&world),
    )
    .unwrap();
    assert_eq!(mgr.rank(), RankId(2));
    assert_eq!(mgr.world().size, 4);
    assert!(Arc::ptr_eq(mgr.world(), &world));
}

#[test]
fn rejects_rank_out_of_range() {
    let world = Arc::new(World::new(RankId(0), 4, Topology::SingleNode).unwrap());
    let err = TesseraBlockManager::new_with_world(
        small_cfg(),
        16 * 1024 * 1024,
        RankId(99),
        world,
    )
    .expect_err("rank 99 in a world of size 4 must fail");
    assert!(matches!(err, tessera_core::TesseraError::InvalidConfig(_)));
}

#[test]
fn global_id_combines_rank_and_block() {
    let world = Arc::new(World::new(RankId(3), 4, Topology::SingleNode).unwrap());
    let mgr = TesseraBlockManager::new_with_world(
        small_cfg(),
        16 * 1024 * 1024,
        RankId(3),
        world,
    )
    .unwrap();
    let bid = mgr.allocate(1, TokenRange::new(0, 64)).unwrap();
    let gid: GlobalBlockId = mgr.global_id(bid);
    assert_eq!(gid.rank, RankId(3));
    assert_eq!(gid.block, bid);
    assert!(gid.to_string().starts_with("r3:b"));
}

#[test]
fn multi_node_topology_validates_mapping() {
    // MultiNode is only declared today; this test confirms World validation rejects bad maps.
    let good = World::new(
        RankId(0),
        4,
        Topology::MultiNode {
            node_of: vec![NodeId(0), NodeId(0), NodeId(1), NodeId(1)],
        },
    );
    assert!(good.is_some());

    let bad_len = World::new(
        RankId(0),
        4,
        Topology::MultiNode { node_of: vec![NodeId(0), NodeId(1)] },
    );
    assert!(bad_len.is_none());
}
