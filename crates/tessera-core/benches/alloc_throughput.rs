//! Microbenchmark: allocate/free throughput on the CPU mock backend. Establishes a floor for
//! the per-block bookkeeping cost; real-CUDA numbers will be lower because of the device
//! allocator, but the relative shape of the curve should match.

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use tessera_core::{
    CkvDtype, CompressionScheme, MlaBlockConfig, TesseraBlockManager, TokenRange,
};

fn cfg() -> MlaBlockConfig {
    MlaBlockConfig::new(
        CompressionScheme::MlaLatent {
            latent_dim: 64,
            rope_key_dim: 8,
        },
        4,
        64,
        CkvDtype::Bf16,
        0,
    )
    .unwrap()
}

fn bench_alloc_free(c: &mut Criterion) {
    let mut group = c.benchmark_group("block_manager/alloc_free");
    group.throughput(Throughput::Elements(1));
    let mgr = TesseraBlockManager::new(cfg(), 256 * 1024 * 1024).unwrap();
    group.bench_function("alloc_then_free", |b| {
        b.iter(|| {
            let id = mgr.allocate(1, TokenRange::new(0, 64)).unwrap();
            mgr.free(id).unwrap();
        });
    });
    group.finish();
}

criterion_group!(benches, bench_alloc_free);
criterion_main!(benches);
