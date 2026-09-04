//! Deterministic, decomposition-invariant per-cell noise for seeding the
//! MRI at init. Moved verbatim out of problems/puffy/puffy.zig (redesign
//! step 5, 2026-09-04).

/// splitmix64 finalizer; self-contained (no std.hash dependency, so the
/// noise field is reproducible across Zig versions).
fn mix64(z0: u64) u64 {
    var z = z0 +% 0x9E3779B97F4A7C15;
    z = (z ^ (z >> 30)) *% 0xBF58476D1CE4E5B9;
    z = (z ^ (z >> 27)) *% 0x94D049BB133111EB;
    return z ^ (z >> 31);
}

/// Deterministic per-cell noise ξ ∈ [−1, 1) for the init perturbation,
/// hashed from the cell-center INTERNAL coordinates. The grid contract
/// (MPI plan gate 1) makes those coordinates bit-identical for the same
/// physical cell on every rank and thread count, so the noise field is
/// decomposition-invariant by construction.
pub fn perturbXi(x: [4]f64) f64 {
    const b1: u64 = @bitCast(x[1]);
    const b2: u64 = @bitCast(x[2]);
    const b3: u64 = @bitCast(x[3]);
    const u = mix64(mix64(mix64(b1) ^ b2) ^ b3);
    return 2.0 * (@as(f64, @floatFromInt(u >> 11)) * 0x1.0p-53) - 1.0;
}
