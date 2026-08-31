//! Thread-scoped controlling for matmul precision
use std::cell::Cell;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Matmul precision enum
pub enum MatmulPrecision {
    /// f32, strict IEEE, no reduced-precision reductions
    Pedantic,
    /// f32, cublas default
    Highest,
    /// TF32, pytorch 'high'
    High,
}

thread_local! {
    /// Thread local storage for matmul precision
    static MATMUL_PRECISION: Cell<MatmulPrecision> = const { Cell::new(MatmulPrecision::Highest) };
}

/// Get current precision
pub fn matmul_precision() -> MatmulPrecision { MATMUL_PRECISION.with(|c| c.get()) }

/// Runs f with precision set to p
pub fn with_matmul_precision<R>(p: MatmulPrecision, f: impl FnOnce() -> R) -> R {
    struct Guard(MatmulPrecision);

    impl Drop for Guard {
        /// To restore old when we drop
        fn drop(&mut self) { MATMUL_PRECISION.with(|c| c.set(self.0));}
    }

    let _g = Guard(matmul_precision());
    MATMUL_PRECISION.with(|c| c.set(p)); // Set to p
    f() // Immediately execute f
}