//! Microbenchmark: cross-agent share-table throughput under multi-threaded add_share.

use std::sync::Arc;
use std::thread;

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use tessera_core::{block::BlockId, CrossAgentShareTable};

fn bench_concurrent_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("share_table/concurrent_add");
    for threads in [1usize, 4, 8] {
        group.throughput(Throughput::Elements(1));
        group.bench_function(format!("threads={threads}"), |b| {
            b.iter_custom(|iters| {
                let st = Arc::new(CrossAgentShareTable::new());
                let start = std::time::Instant::now();
                let per_thread = iters / threads as u64;
                let mut handles = Vec::with_capacity(threads);
                for t in 0..threads {
                    let st = Arc::clone(&st);
                    handles.push(thread::spawn(move || {
                        for i in 0..per_thread {
                            st.add_share(t as u64, BlockId((i as u32) % 1024));
                        }
                    }));
                }
                for h in handles {
                    h.join().unwrap();
                }
                start.elapsed()
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_concurrent_add);
criterion_main!(benches);
