//! Microbenchmark: xxhash3 throughput across representative block sizes. The headline number
//! to watch is GB/s; on modern x86-64 we expect >10 GB/s on a single core.

// criterion's `criterion_group!` / `criterion_main!` macros expand to undocumented
// `pub fn` items; suppress the workspace-level missing_docs lint for this bench.
#![allow(missing_docs)]

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use tessera_core::content_hash::{ContentHasher, Xxh3Hasher};

fn bench_hash(c: &mut Criterion) {
    let mut group = c.benchmark_group("content_hash/xxh3");
    for size in [4 * 1024, 64 * 1024, 1024 * 1024, 4 * 1024 * 1024] {
        let buf = vec![0xABu8; size];
        let hasher = Xxh3Hasher;
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_function(format!("{} KiB", size / 1024), |b| {
            b.iter(|| {
                let h = hasher.hash(black_box(&buf));
                black_box(h);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_hash);
criterion_main!(benches);
