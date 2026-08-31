// Lint baseline mirrors `ferrotorch-core/src/lib.rs`. `missing_docs` and
// `missing_debug_implementations` are held at `warn` while the workspace-wide
// rustdoc / `Debug` pass is tracked as a follow-up issue (matches the existing
// `ferrotorch-core` precedent — diverging unilaterally from a leaf crate would
// be Step 4 architectural unilateralism). `unsafe_code` is intentionally NOT
// denied: this crate is fundamentally unsafe-using (PTX launches, raw pointer
// slices, FFI to cudarc); per-block SAFETY substantiation is tracked in the
// gpu-B..gpu-F dispatches.
#![warn(clippy::all, clippy::pedantic)]
#![deny(rust_2018_idioms)]
// `missing_debug_implementations` is held at `allow` while the workspace-wide
// `Debug` follow-up is tracked separately. `missing_docs` flipped from `allow`
// to `deny` as part of the workspace-wide rustdoc pass (#703).
#![allow(missing_debug_implementations)]
#![deny(missing_docs)]
// Host-only builds compile the same public GPU taxonomy as no-op stubs so
// downstream crates can typecheck without CUDA. Those stubs mirror the
// documented CUDA APIs and intentionally return `GpuError::NoCudaFeature`.
#![cfg_attr(
    not(feature = "cuda"),
    allow(
        missing_docs,
        dead_code,
        reason = "host-only builds expose documented CUDA APIs as NoCudaFeature stubs"
    )
)]
// Pedantic lints we explicitly accept across this crate. Each allow names a
// concrete reason — the alternative would be churn-for-zero-benefit or a
// worse API. Mirrors the ferrotorch-core baseline; add to this list only with
// a one-line justification.
#![allow(
    // `MpsDevice`/`GpuDevice`/`GpuTensor`/`GpuError`-style names intentionally
    // repeat the crate name — that's the API shape consumers expect.
    clippy::module_name_repetitions,
    // # Errors / # Panics sections will be added in the workspace-wide
    // rustdoc pass (tracked separately, not gated on this lint baseline).
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    // Long match-on-op blocks mirror the kernel taxonomy 1:1; splitting
    // reduces legibility.
    clippy::too_many_lines,
    // Numeric ML code casts pervasively between integer/float widths around
    // GPU buffer sizes, dimensions, and indexing; the explicit cast is more
    // readable than try_into/unwrap or num-traits indirection.
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::cast_lossless,
    // `#[must_use]` on every getter is churn for marginal value; callers in
    // this codebase already use the returned values.
    clippy::must_use_candidate,
    // Builder-style methods returning `Self` document their pattern in the
    // type signature; `#[must_use]` is noise.
    clippy::return_self_not_must_use,
    // Math kernels naturally use single-character names (m, k, n for matmul
    // dims; i, j for indices); requiring longer names hurts readability.
    clippy::many_single_char_names,
    clippy::similar_names,
    // Doc comments follow the standard rustdoc layout; pedantic doc-markdown
    // rules are too aggressive for technical prose with PTX assembly.
    clippy::doc_markdown,
    // Hex-encoded constants in PTX templates and Philox round constants don't
    // gain readability from the underscore separators clippy prefers.
    clippy::unreadable_literal,
    // Test/helper modules define small fns after `let`-bindings; the
    // hoisting requirement is style-only.
    clippy::items_after_statements,
    // GPU trait methods often take many `&GpuBufferHandle` parameters that
    // mirror a kernel's input signature; each refactor is its own follow-up.
    clippy::too_many_arguments,
    // `let ... else { return }` rewrites of `match { Some(x) => x, None => return }`
    // are often less readable when the match arm is the natural pattern.
    clippy::manual_let_else,
    // `.collect::<Vec<_>>()` after mapping is the idiomatic shape; rewriting
    // to `extend(map(..))` is lossier and clippy's preference is contested.
    clippy::redundant_closure_for_method_calls,
    // Match arms that wrap a single variant of an enum and re-export are
    // intentional when the variant set is documented and the wrapper is part
    // of the API.
    clippy::match_wildcard_for_single_variants,
    // Parameter names `_a`, `_b` mirror tensor naming conventions
    // (left/right operand) and are not single-letter cargo-cult.
    clippy::single_match_else,
    // `for i in 0..n { ... }` over indices is the natural shape for kernel
    // launch math; .iter().enumerate() is needlessly indirect.
    clippy::needless_range_loop,
    // Manual `Debug` impls intentionally omit non-Debug fields like
    // `Box<dyn Fn>` callbacks, `cudarc` opaque handles, and `Mutex<...>`
    // contents to keep the formatted output useful and free of lock probes.
    clippy::missing_fields_in_debug,
    // Methods that take `&self` for a uniform interface (e.g., guard accessors
    // that are conceptually about the guard but don't read state) are part of
    // the public API shape and not refactor candidates from gpu-A.
    clippy::unused_self,
    // `.map(...).unwrap_or(...)` is the documented PyTorch-style fallback
    // shape used in the OOM recovery path; rewriting to `match` is lossier.
    clippy::map_unwrap_or,
    // PTX template strings, blas/solver test code, and Box<dyn Any>-erased
    // capture pools predate gpu-A's hygiene baseline; remaining pedantic
    // warnings (raw-pointer cast styles, `cloned` vs `copied`, trailing
    // commas in `assert!(.., "msg",)`, `if x { 1 } else { 0 }` patterns,
    // wildcard enum imports, `!=` simplifications, `format!` string interp,
    // strict-float-eq in identity-matmul tests) are tracked for the
    // gpu-B..gpu-F dispatches. Keeping `-D warnings` viable now while the
    // SAFETY substantiation work decides how to phrase those sites.
    clippy::ptr_as_ptr,
    clippy::ref_as_ptr,
    clippy::borrow_as_ptr,
    clippy::cast_ptr_alignment,
    clippy::bool_to_int_with_if,
    clippy::float_cmp,
    clippy::cloned_instead_of_copied,
    clippy::single_char_pattern,
    clippy::uninlined_format_args,
    clippy::wildcard_imports,
    clippy::enum_glob_use,
    clippy::if_not_else,
    clippy::needless_pass_by_value,
    clippy::assigning_clones,
    clippy::semicolon_if_nothing_returned,
    clippy::redundant_else,
    clippy::unnecessary_trailing_comma,
)]

//! CUDA GPU backend for ferrotorch.
//!
//! This crate provides device management, memory allocation, and host/device
//! data transfers built on [`cudarc`]. It is the bridge between ferrotorch's
//! CPU tensor world and NVIDIA GPUs.
//!
//! # Feature flags
//!
//! | Feature | Default | Description |
//! |---------|---------|-------------|
//! | `cuda`  | **yes** | Links against the CUDA driver API via cudarc. Disable to compile on machines without a GPU. |
//!
//! # Quick start
//!
//! ```rust,no_run
//! use ferrotorch_gpu::{GpuDevice, GpuError, cpu_to_gpu, gpu_to_cpu};
//!
//! fn main() -> Result<(), GpuError> {
//!     let device = GpuDevice::new(0)?;
//!     let host_data = vec![1.0_f32, 2.0, 3.0];
//!     let gpu_buf = cpu_to_gpu(&host_data, &device)?;
//!     let back = gpu_to_cpu(&gpu_buf, &device)?;
//!     assert_eq!(back, host_data);
//!     Ok(())
//! }
//! ```
//!
//! ## REQ status (per `.design/ferrotorch-gpu/lib.md`)
//!
//! Full evidence rows (impl + non-test production consumer + upstream
//! cites) live in the design doc; this synopsis is a one-line summary per
//! REQ.
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (lint baseline) | SHIPPED | `#![warn(clippy::all, clippy::pedantic)] + #![deny(rust_2018_idioms, missing_docs)]` at top of `lib.rs` with per-item `#![allow(..)]` justifications; consumer: every `ferrotorch-gpu/src/*.rs` module compiles under this baseline |
//! | REQ-2 (feature-flag matrix) | SHIPPED | cfg-gated `pub mod` declarations + matching cfg-gated `pub use` re-exports in `lib.rs`; consumer `ferrotorch-distributions/src/fallback.rs` invokes `ferrotorch_gpu::init_cuda_backend()` under host-only configurations |
//! | REQ-3 (flat module taxonomy) | SHIPPED | 32 `pub mod X` lines in `lib.rs`, one-file-per-module; consumer `tooling/translate-routes.toml` enumerates one route per `ferrotorch-gpu/src/*.rs` matching the declarations |
//! | REQ-4 (ergonomic re-exports) | SHIPPED | `pub use device::GpuDevice / error::{GpuError, GpuResult} / buffer::CudaBuffer / graph::{...} / memory_guard::{...}` etc. in `lib.rs`; consumer `ferrotorch-diffusion/src/gpu/clip.rs` imports `ferrotorch_gpu::{CudaBuffer, GpuDevice, GpuError, gpu_bmm_f32, gpu_layernorm, gpu_matmul_f32, gpu_softmax}` directly off the crate root |
//! | REQ-5 (cuSPARSE symbol-resolution probe) | SHIPPED | `mod cusparse_smoke in lib.rs` binds `cudarc::cusparse::sys::cusparseHandle_t = std::ptr::null_mut()`; consumer `ferrotorch-gpu/src/sparse.rs` (the real cuSPARSE SpMM caller) requires the symbol to resolve at compile time |

pub mod allocator;
#[cfg(feature = "cuda")]
pub mod atan2;
pub mod backend_impl;
#[cfg(feature = "cuda")]
pub mod bf16;
pub mod blas;
#[cfg(feature = "cuda")]
pub mod bool_kernels;
pub mod buffer;
#[cfg(feature = "cuda")]
pub mod cast_kernels;
pub mod conv;
#[cfg(feature = "cuda")]
pub mod copysign;
#[cfg(feature = "cuda")]
pub mod cufft;
#[cfg(feature = "cuda")]
pub mod cumulative_kernels;
pub mod cusolver;
#[cfg(all(feature = "cuda", feature = "cusparselt"))]
pub mod cusparselt;
pub mod device;
pub mod diag;
pub mod distance;
pub mod error;
#[cfg(feature = "cuda")]
pub mod f16;
pub mod flash_attention;
#[cfg(feature = "cuda")]
pub mod flip;
#[cfg(feature = "cuda")]
pub mod frequency;
#[cfg(feature = "cuda")]
pub mod gather_int;
pub mod graph;
#[cfg(feature = "cuda")]
pub mod group_norm;
#[cfg(feature = "cuda")]
pub mod int_kernels;
pub mod kernels;
#[cfg(feature = "cuda")]
pub mod masked_kernels;
#[cfg(feature = "cuda")]
pub mod median;
pub mod memory_guard;
pub mod module_cache;
#[cfg(feature = "cuda")]
pub mod nan_reductions;
pub mod pool;
pub mod precision;
#[cfg(feature = "cuda")]
pub mod reduce_arg;
#[cfg(feature = "cuda")]
pub mod repeat_interleave;
pub mod rng;
#[cfg(feature = "cuda")]
pub mod roll;
#[cfg(feature = "cuda")]
pub mod scatter_gather_kernels;
#[cfg(feature = "cuda")]
pub mod search;
#[cfg(feature = "cuda")]
pub mod semi_structured;
mod shape_math;
#[cfg(feature = "cuda")]
pub mod sparse;
#[cfg(feature = "cuda")]
pub mod special;
pub mod stream;
pub mod tensor_bridge;
pub mod transcendental;
pub mod transfer;
#[cfg(feature = "cuda")]
pub mod triangular;
#[cfg(feature = "cuda")]
pub mod upsample;

// Re-exports for ergonomic use.
pub use allocator::CudaAllocator;
pub use backend_impl::{CudaBackendImpl, get_cuda_device, init_cuda_backend};
#[cfg(feature = "cuda")]
pub use bf16::{
    gpu_add_bf16, gpu_block_reduce_max_abs_bf16, gpu_causal_mask_bf16, gpu_embedding_gather_bf16,
    gpu_embedding_gather_bf16_to_f32, gpu_fatrelu_bf16, gpu_gelu_bf16, gpu_layernorm_bf16,
    gpu_mul_bf16, gpu_relu_bf16, gpu_repeat_kv_bf16, gpu_rmsnorm_bf16, gpu_rope_half_bf16,
    gpu_scale_bf16, gpu_silu_bf16, gpu_softmax_bf16, gpu_transpose_from_heads_bf16,
    gpu_transpose_to_heads_bf16,
};
pub use blas::gpu_bmm_f32;
pub use blas::{gpu_bmm_f32_into, gpu_matmul_f32_into};
#[cfg(feature = "cuda")]
pub use blas::{
    gpu_broadcast_bmm_bf16_bf16, gpu_broadcast_bmm_f16_f16, gpu_matmul_bf16_bf16,
    gpu_matmul_bf16_bf16_nt, gpu_matmul_bf16_bf16_strided_batched,
    gpu_matmul_bf16_bf16_strided_batched_nt, gpu_matmul_f16_f16_strided_batched,
};
pub use blas::{gpu_matmul_f32, gpu_matmul_f32_nt, gpu_matmul_f64, gpu_matmul_f64_nt};
#[cfg(feature = "cuda")]
pub use bool_kernels::gpu_broadcast_bool;
pub use buffer::CudaBuffer;
pub use conv::gpu_conv2d_f32;
pub use device::GpuDevice;
pub use error::{GpuError, GpuResult};
pub use flash_attention::{gpu_flash_attention_f32, gpu_flash_attention_f64};
#[cfg(feature = "cuda")]
pub use flip::{gpu_flip_f32, gpu_flip_f64, gpu_flip_u16};
pub use graph::{
    CaptureMode, CapturePool, CaptureStatus, CapturedGraph, GraphPoolHandle, PrivateMemPool,
    begin_capture, capture_into_private_pool, capture_pool_for_handle, end_capture,
    end_capture_with_pool, graph_pool_handle, make_graphed_callable, release_graph_pool_handle,
};
#[cfg(feature = "cuda")]
pub use graph::{
    GraphCaptureGuard, MemPoolScope, begin_capture_with_mode, begin_capture_with_pool,
    begin_capture_with_pool_mode, capture_status, is_stream_capturing,
};
#[cfg(feature = "cuda")]
pub use group_norm::gpu_group_norm_f32;
#[cfg(feature = "cuda")]
pub use group_norm::{
    gpu_batch_norm_backward_f32, gpu_local_response_norm_backward_f32, gpu_local_response_norm_f32,
};
#[cfg(feature = "cuda")]
pub use kernels::gpu_broadcast_div_rounding;
#[cfg(feature = "cuda")]
pub use kernels::gpu_div_rounding;
pub use kernels::{gpu_add, gpu_mul, gpu_neg, gpu_relu, gpu_sub};
#[cfg(feature = "cuda")]
pub use kernels::{
    gpu_add_into, gpu_add_into_on_stream, gpu_embed_lookup_into, gpu_gelu_into, gpu_layernorm_into,
    gpu_mul_into, gpu_permute_0213_into, gpu_scale_into, gpu_slice_read_into,
    gpu_small_matmul_into, gpu_softmax_into, gpu_transpose_2d_into,
};
#[cfg(feature = "cuda")]
pub use kernels::{gpu_add_scaled_f32, gpu_add_scaled_f64};
pub use kernels::{gpu_broadcast_add, gpu_broadcast_mul, gpu_broadcast_sub};
#[cfg(feature = "cuda")]
pub use kernels::{gpu_causal_mask_indirect, gpu_slice_write_indirect};
pub use kernels::{
    gpu_dropout, gpu_embed_lookup, gpu_gelu, gpu_layernorm, gpu_permute_0213, gpu_slice_read,
    gpu_slice_write, gpu_softmax, gpu_transpose_2d,
};
#[cfg(feature = "cuda")]
pub use kernels::{gpu_small_bmm, gpu_small_matmul};
pub use memory_guard::{
    MemoryGuard, MemoryGuardBuilder, MemoryGuardedDevice, MemoryHook, MemoryPressureListener,
    MemoryReservation, MemoryStats, MemoryWatchdog, OomPolicy, PressureLevel,
};
pub use pool::{cached_bytes, empty_cache, empty_cache_all, round_len};
pub use precision::{MatmulPrecision, matmul_precision, with_matmul_precision};
#[cfg(feature = "cuda")]
pub use repeat_interleave::{
    gpu_repeat_interleave_backward_bf16, gpu_repeat_interleave_backward_f16,
    gpu_repeat_interleave_backward_f32, gpu_repeat_interleave_backward_f64,
    gpu_repeat_interleave_f32, gpu_repeat_interleave_f64, gpu_repeat_interleave_u16,
};
pub use rng::{CudaRngManager, PhiloxGenerator, PhiloxState, cuda_rng_manager, fork_rng, join_rng};
#[cfg(feature = "cuda")]
pub use roll::{gpu_roll_f32, gpu_roll_f64};
#[cfg(feature = "cuda")]
pub use scatter_gather_kernels::{
    gpu_gather_dim_f32, gpu_gather_dim_f64, gpu_scatter_add_dim_bf16, gpu_scatter_add_dim_f16,
    gpu_scatter_add_dim_f32, gpu_scatter_add_dim_f64, gpu_scatter_add_nd_bf16,
    gpu_scatter_add_nd_f16, gpu_scatter_add_nd_f32, gpu_scatter_add_nd_f64,
    gpu_scatter_add_segments_bf16, gpu_scatter_add_segments_f16, gpu_scatter_add_segments_f32,
    gpu_scatter_add_segments_f64, gpu_scatter_dim_f32, gpu_scatter_dim_f64, gpu_scatter_dim_u16,
    gpu_scatter_nd_f32, gpu_scatter_nd_f64, gpu_scatter_nd_u16, gpu_scatter_reduce_nd_f32,
    gpu_scatter_reduce_nd_f64, gpu_scatter_value_dim_f32, gpu_scatter_value_dim_f64,
    gpu_scatter_value_nd_f32, gpu_scatter_value_nd_f64, gpu_scatter_value_nd_u16,
};
#[cfg(feature = "cuda")]
pub use search::{
    gpu_histc_f32, gpu_histc_f64, gpu_meshgrid_bf16, gpu_meshgrid_f16, gpu_meshgrid_f32,
    gpu_meshgrid_f64, gpu_searchsorted_bf16, gpu_searchsorted_f16, gpu_searchsorted_f32,
    gpu_searchsorted_f64, gpu_topk_bf16, gpu_topk_f16, gpu_topk_f32, gpu_topk_f64,
    gpu_topk_nd_bf16, gpu_topk_nd_f16, gpu_topk_nd_f32, gpu_topk_nd_f64, gpu_unique_bf16,
    gpu_unique_consecutive_bf16, gpu_unique_consecutive_f16, gpu_unique_consecutive_f32,
    gpu_unique_consecutive_f64, gpu_unique_f16, gpu_unique_f32, gpu_unique_f64,
};
#[cfg(feature = "cuda")]
pub use special::{
    gpu_airy_ai_f32, gpu_chebyshev_poly_f32, gpu_chebyshev_poly_f64, gpu_entr_f32,
    gpu_hermite_h_poly_f32, gpu_hermite_h_poly_f64, gpu_hermite_he_poly_f32,
    gpu_hermite_he_poly_f64, gpu_laguerre_poly_f32, gpu_laguerre_poly_f64, gpu_legendre_poly_f32,
    gpu_legendre_poly_f64, gpu_ndtr_f32, gpu_ndtri_f32, gpu_spherical_bessel_j0_f32, gpu_zeta_f32,
};
pub use tensor_bridge::{GpuFloat, GpuTensor, cuda, cuda_default, tensor_to_cpu, tensor_to_gpu};
#[cfg(feature = "cuda")]
pub use transfer::alloc_zeros_bf16;
pub use transfer::{alloc_zeros, alloc_zeros_f32, alloc_zeros_f64, cpu_to_gpu, gpu_to_cpu};
#[cfg(feature = "cuda")]
pub use upsample::gpu_nearest_upsample2x_f32;

// ---------------------------------------------------------------------------
// cuSPARSE feature-flag smoke test
// ---------------------------------------------------------------------------
//
// P1 wires the `cusparse` feature on the workspace `cudarc` dep so P2's
// `SparseTensor::spmm` can call into cuSPARSE. This is a type-reference-only
// test — it confirms the symbol resolves at compile time without making any
// CUDA call. P2 will exercise the actual SpMM path.

#[cfg(test)]
#[cfg(feature = "cuda")]
mod cusparse_smoke {
    /// Compile-time check that `cudarc::cusparse::sys::cusparseHandle_t` is
    /// reachable through the workspace dep's feature set. The function never
    /// runs against a real GPU; the assertion is purely that the type resolves
    /// (i.e., `cusparse` is enabled on `cudarc`).
    #[test]
    fn cusparse_handle_type_resolves() {
        // A null handle is a valid bit pattern for the raw pointer typedef and
        // proves the `sys::cusparseHandle_t` symbol is in scope without
        // touching the device.
        let handle: cudarc::cusparse::sys::cusparseHandle_t = std::ptr::null_mut();
        assert!(handle.is_null());
    }
}
