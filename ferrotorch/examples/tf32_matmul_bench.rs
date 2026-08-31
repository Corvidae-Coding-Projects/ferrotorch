//! Validates the matmul-precision patch (`ferrotorch_gpu::MatmulPrecision`).
//!
//! Two things are measured across all three precisions (Pedantic, Highest,
//! High/TF32):
//!   1. Speed: GEMM shapes taken directly from a GPT-2 (124M) block forward
//!      pass (B=4, T=1024).
//!   2. Numerical fidelity: error vs an f64 reference, so a speedup can't be
//!      reported without also reporting what it costs in precision.
//!
//! Run with: cargo run --release --features gpu --example tf32_matmul_bench

use std::hint::black_box;
use std::time::Instant;

use ferrotorch_core::*;
use ferrotorch_gpu::{MatmulPrecision, with_matmul_precision};

const PRECISIONS: [(&str, MatmulPrecision); 3] = [
    ("Pedantic  ", MatmulPrecision::Pedantic),
    ("Highest   ", MatmulPrecision::Highest),
    ("High(TF32)", MatmulPrecision::High),
];

/// Same synchronized-timing methodology as `ferrotorch_bench.rs`'s `bench_gpu`:
/// wall-clock across `iters` calls, with a `synchronize()` before starting the
/// clock and after stopping it so async kernel launch latency isn't measured
/// instead of real execution time.
fn bench_gpu<R, F>(name: &str, warmup: usize, iters: usize, mut f: F) -> f64
where
    F: FnMut() -> R,
{
    let sync = || {
        if let Some(b) = ferrotorch_core::gpu_dispatch::gpu_backend() {
            let _ = b.synchronize(0);
        }
    };
    for _ in 0..warmup {
        black_box(f());
    }
    sync();
    let start = Instant::now();
    for _ in 0..iters {
        black_box(f());
    }
    sync();
    let elapsed = start.elapsed().as_secs_f64() / iters as f64 * 1e6; // microseconds
    println!("  {name}: {elapsed:.1} us");
    elapsed
}

fn main() -> FerrotorchResult<()> {
    if ferrotorch_core::gpu_dispatch::gpu_backend().is_none() {
        ferrotorch_gpu::init_cuda_backend().expect("no CUDA device available");
    }

    println!("{}", "=".repeat(78));
    println!("GPT-2 (124M) GEMM shapes: Pedantic vs Highest (cuBLAS default) vs High (TF32)");
    println!("{}", "=".repeat(78));

    // (name, m, k, n, warmup, iters) — shapes as they occur in a B=4, T=1024
    // GPT-2 block. `m` = B*T tokens; lm_head gets fewer iters since its
    // output tensor is ~800MB at this batch size.
    let shapes: [(&str, usize, usize, usize, usize, usize); 5] = [
        ("c_attn    [4096,768]x[768,2304] ", 4 * 1024, 768, 2304, 10, 30),
        ("attn proj [4096,768]x[768,768]  ", 4 * 1024, 768, 768, 10, 30),
        ("mlp c_fc  [4096,768]x[768,3072] ", 4 * 1024, 768, 3072, 10, 30),
        ("mlp proj  [4096,3072]x[3072,768]", 4 * 1024, 3072, 768, 10, 30),
        ("lm_head   [4096,768]x[768,50257]", 4 * 1024, 768, 50257, 5, 15),
    ];

    let mut rows = Vec::new();
    for (name, m, k, n, warmup, iters) in shapes {
        let a = rand::<f32>(&[m, k])?.cuda()?;
        let b = rand::<f32>(&[k, n])?.cuda()?;

        let mut times = [0.0; 3];
        for (i, (label, prec)) in PRECISIONS.into_iter().enumerate() {
            times[i] = with_matmul_precision(prec, || {
                bench_gpu(&format!("{name} {label}"), warmup, iters, || {
                    a.matmul(&b).unwrap()
                })
            });
        }
        let speedup = times[1] / times[2]; // Highest -> High(TF32)
        println!("  -> {speedup:.2}x\n");
        rows.push((name, times, speedup));
    }

    println!("{}", "=".repeat(78));
    println!("Summary (GPT-2 124M GEMMs, B=4 T=1024)");
    println!("{}", "=".repeat(78));
    println!(
        "{:<34}{:>12}{:>12}{:>12}{:>9}",
        "shape", "Pedantic(us)", "Highest(us)", "TF32(us)", "speedup"
    );
    for (name, times, sp) in &rows {
        println!(
            "{name:<34}{:>12.1}{:>12.1}{:>12.1}{sp:>8.2}x",
            times[0], times[1], times[2]
        );
    }

    // ---- Numerical fidelity vs an f64 reference ----
    println!("\n{}", "=".repeat(78));
    println!("Numerical error vs f64 reference — matmul [1024,1024] x [1024,1024]");
    println!("{}", "=".repeat(78));

    let n = 1024;
    let a64 = rand::<f64>(&[n, n])?;
    let b64 = rand::<f64>(&[n, n])?;
    let reference = a64.matmul(&b64)?.data_vec()?;

    let a32_data: Vec<f32> = a64.data_vec()?.iter().map(|&x| x as f32).collect();
    let b32_data: Vec<f32> = b64.data_vec()?.iter().map(|&x| x as f32).collect();
    let a32 = from_vec(a32_data, &[n, n])?.cuda()?;
    let b32 = from_vec(b32_data, &[n, n])?.cuda()?;

    for (label, prec) in PRECISIONS {
        let out = with_matmul_precision(prec, || a32.matmul(&b32).unwrap());
        let out = out.cpu()?.data_vec()?;

        let mut max_abs = 0.0_f64;
        let mut sum_rel = 0.0_f64;
        let mut counted = 0usize;
        for (o, r) in out.iter().zip(reference.iter()) {
            let o = *o as f64;
            let diff = (o - r).abs();
            max_abs = max_abs.max(diff);
            if r.abs() > 1e-6 {
                sum_rel += diff / r.abs();
                counted += 1;
            }
        }
        println!(
            "  {label}  max_abs_err={max_abs:.3e}  mean_rel_err={:.3e}",
            sum_rel / counted.max(1) as f64
        );
    }

    Ok(())
}
