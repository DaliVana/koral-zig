//! Small shared numerical utilities that are not tied to any one subsystem
//! (C: misc.c). Kept here so callers across the tree (radviscosity mfp
//! limiter, dynamo radial weighting, …) share one definition.

const std = @import("std");

/// C: step_function (misc.c:1311) — a smoothed Heaviside normalized so that
/// f(x9) = 0.95 (k = 1.47222/x9).
pub fn stepFunction(x: f64, x9: f64) f64 {
    const k = 1.47222 / x9;
    return 1.0 / (1.0 + @exp(-2.0 * k * x));
}
