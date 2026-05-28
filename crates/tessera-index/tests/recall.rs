//! Coarse recall test: inserting N random vectors and querying with each one back must return
//! the original block as the top match. This is a baseline sanity check; the production
//! recall target is tuned per-deployment via `expansion_search`.

use rand::{Rng, SeedableRng};
use tessera_index::{IndexBackend, UsearchConfig, UsearchIndex};

fn rand_vec(rng: &mut impl Rng, dim: usize) -> Vec<f32> {
    (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect()
}

#[test]
fn top1_recall_is_perfect_on_inserted_vectors() {
    let dim = 64;
    let n = 256;
    let idx = UsearchIndex::new(UsearchConfig::default_for_dim(dim)).unwrap();
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0xBEEF);
    let mut vecs = Vec::with_capacity(n);
    for i in 0..n {
        let v = rand_vec(&mut rng, dim);
        idx.add(i as u32, &v).unwrap();
        vecs.push(v);
    }
    let mut hits = 0;
    for (i, v) in vecs.iter().enumerate() {
        let results = idx.query(v, 1).unwrap();
        if !results.is_empty() && results[0].block_id == i as u32 {
            hits += 1;
        }
    }
    let recall = hits as f64 / n as f64;
    assert!(recall >= 0.95, "top-1 self-recall too low: {recall}");
}
