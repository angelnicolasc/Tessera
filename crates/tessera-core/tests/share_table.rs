//! Cross-agent share table accounting under multi-owner / partial release.

use tessera_core::{block::BlockId, CrossAgentShareTable};

#[test]
fn release_of_only_owner_drops_the_block() {
    let st = CrossAgentShareTable::new();
    st.add_share(7, BlockId(99));
    assert_eq!(st.shared_block_count(), 1);
    let to_free = st.release_request(7);
    assert_eq!(to_free, vec![BlockId(99)]);
    assert!(st.owners(BlockId(99)).is_none());
}

#[test]
fn partial_release_keeps_block_alive() {
    let st = CrossAgentShareTable::new();
    st.add_share(1, BlockId(5));
    st.add_share(2, BlockId(5));
    let to_free = st.release_request(1);
    assert_eq!(to_free, vec![BlockId(5)]);
    // Still owned by req 2.
    assert_eq!(st.owners(BlockId(5)).unwrap(), vec![2]);
}

#[test]
fn sharing_rate_is_bounded_and_monotone() {
    let st = CrossAgentShareTable::new();
    assert_eq!(st.sharing_rate(), 0.0);
    st.add_share(1, BlockId(1));
    assert!(st.sharing_rate() > 0.0 && st.sharing_rate() <= 1.0);
}
