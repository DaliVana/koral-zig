//! Small shared numerical utilities that are not tied to any one subsystem
//! (C: misc.c). Kept here so callers across the tree (radviscosity mfp
//! limiter, dynamo radial weighting, …) share one definition.

const std = @import("std");

/// C: step_function (misc.c:1311). A smoothed Heaviside normalized so that
/// f(x9) = 0.95 (k = 1.47222/x9).
pub fn stepFunction(x: f64, x9: f64) f64 {
    const k = 1.47222 / x9;
    return 1.0 / (1.0 + @exp(-2.0 * k * x));
}

/// HARM's nrutil.c bisection (KORAL tools.c:26 rtbis): 100 halvings max,
/// exits at |dx| < xacc. `func` is any value with `eval(self, x: f64) f64`.
/// On an unbracketed root C prints a warning and marches on; we do the
/// same (silently).
pub fn rtbis(func: anytype, x1: f64, x2: f64, xacc: f64) f64 {
    const f = func.eval(x1);
    var fmid = func.eval(x2);
    var dx: f64 = undefined;
    var rtb: f64 = undefined;
    if (f < 0.0) {
        dx = x2 - x1;
        rtb = x1;
    } else {
        dx = x1 - x2;
        rtb = x2;
    }
    var j: usize = 1;
    while (j <= 100) : (j += 1) {
        dx *= 0.5;
        const xmid = rtb + dx;
        fmid = func.eval(xmid);
        if (fmid <= 0.0) rtb = xmid;
        if (@abs(dx) < xacc or fmid == 0.0) return rtb;
    }
    return 0.0; // C: "Too many bisections in rtbis"
}
