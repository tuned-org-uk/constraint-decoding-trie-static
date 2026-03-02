use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rand::RngExt;
use rand::{SeedableRng, rngs::SmallRng};

use transition_matrix_trie::transition::build_static_index;
use transition_matrix_trie::vntk::vntk;

// ─────────────────────────────────────────────────────────────────────────────
// Shared fixture helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Parameters that stay fixed for every benchmark in this file.
struct BenchConfig {
    vocab_size: u32,
    sid_length: u32,
    dense_depth: u32,
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            vocab_size: 2048, // |V| from the paper's YouTube setting
            sid_length: 8,    // L
            dense_depth: 2,   // d
        }
    }
}

/// Generates a deterministic random constraint set of `n` sequences.
fn make_constraints(n: usize, cfg: &BenchConfig, seed: u64) -> Vec<Vec<u32>> {
    let mut rng = SmallRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            (0..cfg.sid_length)
                .map(|_| rng.random_range(0..cfg.vocab_size))
                .collect()
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. VNTK throughput — varying constraint-set size
// ─────────────────────────────────────────────────────────────────────────────

/// Benchmarks the VNTK kernel at M=70 beams, level 2 (first sparse level),
/// across four constraint-set sizes: 10 K, 100 K, 500 K, 1 M.
///
/// Throughput is reported in beams/second so Criterion can track regression
/// independently of machine clock speed.
fn bench_vntk_scaling(c: &mut Criterion) {
    let cfg = BenchConfig::default();
    let beam_width = 70usize; // paper's M
    let level = 2usize; // first sparse level (after d=2 dense layers)

    let sizes: &[usize] = &[10_000, 100_000, 500_000, 1_000_000];

    let mut group = c.benchmark_group("vntk/constraint_set_size");

    for &n in sizes {
        // Pre-build the index outside the timed loop.
        let constraints = make_constraints(n, &cfg, 0xDEAD_BEEF);
        let index = build_static_index(
            &constraints,
            cfg.vocab_size,
            cfg.sid_length,
            cfg.dense_depth,
        );

        // Beam nodes: spread across the trie so we exercise varied CSR rows.
        let current_nodes: Vec<u32> = (0..beam_width as u32)
            .map(|i| i % index.sparse.num_nodes)
            .collect();

        group.throughput(Throughput::Elements(beam_width as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(n),
            &(&current_nodes, &index.sparse),
            |b, (nodes, sparse)| {
                b.iter(|| {
                    vntk(
                        criterion::black_box(nodes),
                        sparse,
                        level,
                        cfg.vocab_size as usize,
                    )
                });
            },
        );
    }

    group.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. VNTK throughput — varying beam width
// ─────────────────────────────────────────────────────────────────────────────

/// Holds beam width constant at 1M constraints, varies M ∈ {1, 10, 70, 140}.
/// Useful for understanding how the kernel scales with batch dimension.
fn bench_vntk_beam_width(c: &mut Criterion) {
    let cfg = BenchConfig::default();
    let level = 2usize;
    let n = 1_000_000usize;

    let constraints = make_constraints(n, &cfg, 0xDEAD_BEEF);
    let index = build_static_index(
        &constraints,
        cfg.vocab_size,
        cfg.sid_length,
        cfg.dense_depth,
    );

    let beam_widths: &[usize] = &[1, 10, 70, 140];
    let mut group = c.benchmark_group("vntk/beam_width");

    for &m in beam_widths {
        let current_nodes: Vec<u32> = (0..m as u32).map(|i| i % index.sparse.num_nodes).collect();

        group.throughput(Throughput::Elements(m as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(m),
            &(&current_nodes, &index.sparse),
            |b, (nodes, sparse)| {
                b.iter(|| {
                    vntk(
                        criterion::black_box(nodes),
                        sparse,
                        level,
                        cfg.vocab_size as usize,
                    )
                });
            },
        );
    }

    group.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. VNTK throughput — varying trie depth (level parameter)
// ─────────────────────────────────────────────────────────────────────────────

/// Benchmarks VNTK at each sparse level (2 through L−1=7) to reveal how
/// branching factor changes across depth affect scatter-gather cost.
fn bench_vntk_depth(c: &mut Criterion) {
    let cfg = BenchConfig::default();
    let beam_width = 70usize;
    let n = 1_000_000usize;

    let constraints = make_constraints(n, &cfg, 0xDEAD_BEEF);
    let index = build_static_index(
        &constraints,
        cfg.vocab_size,
        cfg.sid_length,
        cfg.dense_depth,
    );

    // Sparse levels: dense_depth .. sid_length−1
    let sparse_levels: Vec<usize> = (cfg.dense_depth as usize..cfg.sid_length as usize).collect();

    let mut group = c.benchmark_group("vntk/trie_level");

    for level in sparse_levels {
        // Approximate realistic node distribution at each level by cycling
        // through a prefix of the trie's node range.
        let current_nodes: Vec<u32> = (0..beam_width as u32)
            .map(|i| i % index.sparse.num_nodes)
            .collect();

        group.throughput(Throughput::Elements(beam_width as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(level),
            &(&current_nodes, &index.sparse, level),
            |b, (nodes, sparse, lvl)| {
                b.iter(|| {
                    vntk(
                        std::hint::black_box(nodes),
                        sparse,
                        *lvl,
                        cfg.vocab_size as usize,
                    )
                });
            },
        );
    }

    group.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. Index construction — varying constraint-set size
// ─────────────────────────────────────────────────────────────────────────────

/// Benchmarks `build_static_index` at 10 K, 100 K, 500 K, 1 M constraints.
/// Throughput is reported in constraints/second.
fn bench_build_index_scaling(c: &mut Criterion) {
    let cfg = BenchConfig::default();
    let sizes: &[usize] = &[10_000, 100_000, 500_000, 1_000_000];

    let mut group = c.benchmark_group("build_static_index/constraint_set_size");
    // Construction is slow; reduce the sample count so the suite finishes in
    // reasonable wall time without losing statistical validity.
    group.sample_size(20);

    for &n in sizes {
        // Generate constraints once per size; they are re-used across Criterion
        // sample iterations (only the index build itself is timed).
        let constraints = make_constraints(n, &cfg, 0xCAFE_F00D);

        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &constraints, |b, c_set| {
            b.iter(|| {
                build_static_index(
                    criterion::black_box(c_set),
                    cfg.vocab_size,
                    cfg.sid_length,
                    cfg.dense_depth,
                )
            });
        });
    }

    group.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. Index construction — varying vocabulary size
// ─────────────────────────────────────────────────────────────────────────────

/// Holds constraint count at 100 K and varies |V| ∈ {256, 1024, 2048, 8192}.
/// Exercises how the dense mask allocation cost scales with |V|^d.
fn bench_build_index_vocab(c: &mut Criterion) {
    let base = BenchConfig::default();
    let n = 100_000usize;
    let vocab_sizes: &[u32] = &[256, 1024, 2048, 8192];

    let mut group = c.benchmark_group("build_static_index/vocab_size");
    group.sample_size(20);

    for &v in vocab_sizes {
        let cfg = BenchConfig {
            vocab_size: v,
            ..Default::default()
        };
        // Re-generate with matching vocab so no token exceeds v.
        let constraints = make_constraints(n, &cfg, 0xCAFE_F00D);

        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(v), &constraints, |b, c_set| {
            b.iter(|| {
                build_static_index(
                    criterion::black_box(c_set),
                    cfg.vocab_size,
                    cfg.sid_length,
                    cfg.dense_depth,
                )
            });
        });
    }

    group.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Registration
// ─────────────────────────────────────────────────────────────────────────────

criterion_group!(
    benches,
    bench_vntk_scaling,
    bench_vntk_beam_width,
    bench_vntk_depth,
    bench_build_index_scaling,
    bench_build_index_vocab,
);
criterion_main!(benches);
