//! Velocity conversions and index gymnastics (C: relele.c, frames.c tail).
//!
//! KORAL threads three velocity representations through the code
//! (mnemonics.h): VEL4; lab-frame four-velocity u^μ; VEL3; lab
//! three-velocity u^i/u^t; VELR; relative velocity ũ^i measured by the
//! normal observer. Primitives store VELPRIM = VELR (choices.h:103).
//!
//! Scalar entry points take the whole per-point `Geometry` and pick the
//! metric block the operation needs, so no caller can hand g_μν where g^μν
//! belongs. The lane-generic `<name>G` functions have no `Geometry` to
//! select from, so they take the nominally distinct `MetricCovOf(T)` /
//! `MetricConOf(T)` views (geometry.zig) instead; build those with
//! `geom.cov()` / `geom.con()`, since a bare `.{ .m = ... }` literal
//! coerces to either view and defeats the distinction.
//!
//! The choice is enforced against callers, not inside this file: the
//! helpers below still index `geom.gg` / `geom.GG` directly, so a swap
//! within a body here compiles. The blocks keep C's 4×5 layout (column 4 =
//! extras, ignored here); the arithmetic transcribes 1:1 from relele.c and
//! every doc comment keeps the C name.

const std = @import("std");
const simd = @import("math/simd.zig");
const geometry = @import("geometry.zig");
const Geometry = geometry.Geometry;

pub const MetricCovOf = geometry.MetricCovOf;
pub const MetricConOf = geometry.MetricConOf;
pub const MetricCov = geometry.MetricCov;
pub const MetricCon = geometry.MetricCon;

// The `<name>G(comptime T, ...)` functions are the comptime-T-generic cores
// of the radiative-source chain (parallelization plan §2.2): T is f64 or
// @Vector(W, f64), the plain `<name>` scalar API delegates to T = f64, and
// the vector instantiation is bit-identical per lane (simd_tests.zig).

/// C: mnemonics.h:16-18 (numeric values matter for golden records).
pub const VelType = enum(u8) {
    vel4 = 1,
    vel3 = 2,
    velr = 3,
};

/// Whether `convert` may trust the input's u^t. C exposes the choice as two
/// functions: conv_vels (recompute) and conv_vels_ut (trust).
pub const UtMode = enum {
    recompute_ut,
    trust_ut,
};

pub const Error = error{
    /// C prints "ut.nan in conv_vels" and returns -1 (relele.c:199).
    VelocityConversionFailed,
    /// C calls my_err("delta.lt.0 in fill_utinucon") and would continue
    /// into sqrt(<0) = NaN (relele.c:394); we refuse instead.
    SpacelikeVelocity,
    /// A NaN reached the assembled face flux (physics/flux.zig). Same spirit
    /// as the two above; we refuse to propagate an unphysical state rather
    /// than let it flow into the conserved update, where cell_fixup could
    /// silently neighbour-average it away and finish the run with quietly
    /// wrong physics. C: f_flux_prime's isnan → my_err + exit(-1)
    /// (physics.c:1230-1247). It lives in this shared evolution error set so
    /// it threads through the sweep, the ChunkResult reduction, and the
    /// driver's step-failure path unchanged.
    NanInFlux,
};

pub inline fn dot(a: [4]f64, b: [4]f64) f64 {
    return dotG(f64, a, b);
}

pub inline fn dotG(comptime T: type, a: [4]T, b: [4]T) T {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
}

pub inline fn kron(i: usize, j: usize) f64 {
    return if (i == j) 1.0 else 0.0;
}

//
// ---- index raising / lowering (C: frames.c:1518, :1546) ------------------
//

/// A_μ -> A^μ (C: indices_12). Safe to call with aliased in/out.
pub fn raiseVec(a1: [4]f64, geom: *const Geometry) [4]f64 {
    return raiseVecG(f64, a1, geom.con());
}

pub fn raiseVecG(comptime T: type, a1: [4]T, GG: MetricConOf(T)) [4]T {
    var a2: [4]T = @splat(simd.splat(T, 0));
    for (0..4) |i| {
        for (0..4) |k| {
            a2[i] += a1[k] * GG.m[i][k];
        }
    }
    return a2;
}

/// A^μ -> A_μ (C: indices_21).
pub fn lowerVec(a1: [4]f64, geom: *const Geometry) [4]f64 {
    return lowerVecG(f64, a1, geom.cov());
}

pub fn lowerVecG(comptime T: type, a1: [4]T, gg: MetricCovOf(T)) [4]T {
    var a2: [4]T = @splat(simd.splat(T, 0));
    for (0..4) |i| {
        for (0..4) |j| {
            a2[i] += a1[j] * gg.m[i][j];
        }
    }
    return a2;
}

/// T_μν -> T^μν (C: indices_1122).
pub fn raiseBoth(t1: [4][4]f64, geom: *const Geometry) [4][4]f64 {
    const GG = &geom.GG;
    var t2: [4][4]f64 = @splat(@splat(0));
    for (0..4) |i| {
        for (0..4) |j| {
            for (0..4) |k| {
                for (0..4) |l| {
                    t2[i][j] += t1[k][l] * GG[i][k] * GG[j][l];
                }
            }
        }
    }
    return t2;
}

/// T^μν -> T_μν (C: indices_2211).
pub fn lowerBoth(t1: [4][4]f64, geom: *const Geometry) [4][4]f64 {
    const gg = &geom.gg;
    var t2: [4][4]f64 = @splat(@splat(0));
    for (0..4) |i| {
        for (0..4) |j| {
            for (0..4) |k| {
                for (0..4) |l| {
                    t2[i][j] += t1[k][l] * gg[i][k] * gg[j][l];
                }
            }
        }
    }
    return t2;
}

/// T^μν -> T^μ_ν (C: indices_2221).
pub fn lowerSecond(t1: [4][4]f64, geom: *const Geometry) [4][4]f64 {
    const gg = &geom.gg;
    var t2: [4][4]f64 = @splat(@splat(0));
    for (0..4) |i| {
        for (0..4) |j| {
            for (0..4) |k| {
                t2[i][j] += t1[i][k] * gg[k][j];
            }
        }
    }
    return t2;
}

/// A single row of lowerSecond: T^row_j = Σ_k T^{row k} g_{kj}, for j=0..3.
/// Same inner k-summation as lowerSecond, so the result is bitwise-identical
/// to `lowerSecond(t1, geom)[row]`; 16 madds instead of 64 when a caller
/// reads only one lowered row (flux.fFluxPrime consumes row idim+1 of both
/// stress tensors, P2 #7).
pub fn lowerSecondRow(t1: [4][4]f64, geom: *const Geometry, row: usize) [4]f64 {
    const gg = &geom.gg;
    var out: [4]f64 = @splat(0);
    for (0..4) |j| {
        for (0..4) |k| {
            out[j] += t1[row][k] * gg[k][j];
        }
    }
    return out;
}

/// uout^μ = A^μ_ν uin^ν (C: frames.c multiply2). Alias-safe.
pub fn transformVec(uin: [4]f64, a: [4][4]f64) [4]f64 {
    return transformVecG(f64, uin, a);
}

pub fn transformVecG(comptime T: type, uin: [4]T, a: [4][4]T) [4]T {
    var uout: [4]T = @splat(simd.splat(T, 0));
    for (0..4) |i| {
        for (0..4) |j| {
            uout[i] += a[i][j] * uin[j];
        }
    }
    return uout;
}

/// T2^μν = A^μ_κ A^ν_λ T1^κλ (C: frames.c multiply22). Alias-safe.
pub fn transformTensor(t1: [4][4]f64, a: [4][4]f64) [4][4]f64 {
    var t2: [4][4]f64 = @splat(@splat(0));
    for (0..4) |i| {
        for (0..4) |j| {
            for (0..4) |k| {
                for (0..4) |l| {
                    t2[i][j] += a[i][k] * a[j][l] * t1[k][l];
                }
            }
        }
    }
    return t2;
}

//
// ---- u^t solvers (C: relele.c:342, :373) ----------------------------------
//

/// u^t from spatial VEL3 components: u^t = sqrt(-1/(g00 + 2 g0i v^i + gij v^i v^j))
/// (C: fill_utinvel3).
pub fn utFromVel3(v: [4]f64, geom: *const Geometry) f64 {
    const gg = &geom.gg;
    var b: f64 = 0;
    var c: f64 = 0;
    for (1..4) |i| {
        b += v[i] * gg[0][i];
        for (1..4) |j| {
            c += v[i] * v[j] * gg[i][j];
        }
    }
    return @sqrt(-1.0 / (gg[0][0] + 2.0 * b + c));
}

/// u^t from spatial four-velocity components (C: fill_utinucon); solves
/// g00 ut² + 2b ut + c = 0 taking the (-b - sqrt(Δ))/g00 root in both signs
/// of g00 (C keeps the minus root in the ergoregion too, relele.c:407).
pub fn utFromSpatialUcon(u: [4]f64, geom: *const Geometry) Error!f64 {
    const gg = &geom.gg;
    const a = gg[0][0];
    var b: f64 = 0;
    var c: f64 = 1;
    for (1..4) |i| {
        b += u[i] * gg[0][i];
        for (1..4) |j| {
            c += u[i] * u[j] * gg[i][j];
        }
    }
    const delta = b * b - a * c;
    if (delta < 0) return Error.SpacelikeVelocity;
    return (-b - @sqrt(delta)) / a;
}

/// α·γ for a VELR vector (C: calc_alpgam, relele.c:309). On alpgam² < 0
/// C prints and returns 1; mirrored.
pub fn alphaGamma(u: [4]f64, geom: *const Geometry) f64 {
    return alphaGammaG(f64, u, geom.cov(), geom.con());
}

pub fn alphaGammaG(comptime T: type, u: [4]T, gg: MetricCovOf(T), GG: MetricConOf(T)) T {
    const sp = simd.splat;
    var qsq: T = sp(T, 0);
    for (1..4) |i| {
        for (1..4) |j| {
            qsq += u[i] * u[j] * gg.m[i][j];
        }
    }
    const alpgam2 = (sp(T, -1.0) / GG.m[0][0]) * (sp(T, 1.0) + qsq);
    return simd.select(T, alpgam2 < sp(T, 0), sp(T, 1.0), @sqrt(alpgam2));
}

/// The VELR → VEL4 conversion (convert's velr branch); infallible:
/// alphaGamma absorbs alpgam² < 0 by returning 1, C-faithfully. This is
/// the only conv_vels direction the T-generic radiative chain needs.
pub fn velrToVel4G(comptime T: type, uin: [4]T, gg: MetricCovOf(T), GG: MetricConOf(T)) [4]T {
    const alpgam = alphaGammaG(T, uin, gg, GG);
    return .{
        -alpgam * GG.m[0][0],
        uin[1] - alpgam * GG.m[0][1],
        uin[2] - alpgam * GG.m[0][2],
        uin[3] - alpgam * GG.m[0][3],
    };
}

//
// ---- conv_vels (C: relele.c:136 conv_vels_core) ---------------------------
//

/// Contravariant velocity conversion (C: conv_vels_core, exposed as
/// conv_vels for `.recompute_ut` and conv_vels_ut for `.trust_ut`); u[0] of
/// the input is ignored unless `.trust_ut`. Returns the converted 4-vector.
pub fn convert(
    uin: [4]f64,
    from: VelType,
    to: VelType,
    geom: *const Geometry,
    ut: UtMode,
) Error![4]f64 {
    const GG = &geom.GG;
    const ut_known = ut == .trust_ut;

    if (from == to) {
        // VEL4 -> VEL4 recomputes u^t when unknown; VEL3/VELR copy through.
        var uout = uin;
        if (from == .vel4 and !ut_known) uout[0] = try utFromSpatialUcon(uin, geom);
        return uout;
    }

    // from != to: an exhaustive (from, to) matrix so the compiler proves every
    // VelType pair is handled. The diagonal arms are `unreachable` (caught by
    // the from == to fast path above); this restores the my_err fallback the
    // port dropped from C's conv_vels_core if-chain (relele.c:136).
    var uout: [4]f64 = undefined;
    switch (from) {
        .vel4 => switch (to) {
            .vel4 => unreachable,
            .vel3 => {
                const ut_ = if (ut_known) uin[0] else try utFromSpatialUcon(uin, geom);
                uout = .{ 1.0, uin[1] / ut_, uin[2] / ut_, uin[3] / ut_ };
            },
            .velr => {
                const ut_ = if (ut_known) uin[0] else try utFromSpatialUcon(uin, geom);
                uout[0] = ut_;
                for (1..4) |i| {
                    uout[i] = uin[i] - ut_ * GG[0][i] / GG[0][0];
                }
            },
        },
        .vel3 => switch (to) {
            .vel3 => unreachable,
            .vel4 => {
                const ut_ = utFromVel3(uin, geom);
                if (ut_ < 1.0 or std.math.isNan(ut_)) return Error.VelocityConversionFailed;
                uout = .{ ut_, uin[1] * ut_, uin[2] * ut_, uin[3] * ut_ };
            },
            .velr => {
                const ut_ = utFromVel3(uin, geom);
                if (ut_ < 1.0 or std.math.isNan(ut_)) return Error.VelocityConversionFailed;
                uout = .{ ut_, uin[1] * ut_, uin[2] * ut_, uin[3] * ut_ };
                for (1..4) |i| {
                    uout[i] = uout[i] - uout[0] * GG[0][i] / GG[0][0];
                }
            },
        },
        .velr => switch (to) {
            .velr => unreachable,
            .vel4 => uout = velrToVel4G(f64, uin, geom.cov(), geom.con()),
            .vel3 => {
                const alpgam = alphaGamma(uin, geom);
                uout[0] = -alpgam * GG[0][0];
                uout[1] = uin[1] / uout[0] + GG[0][1] / GG[0][0];
                uout[2] = uin[2] / uout[0] + GG[0][2] / GG[0][0];
                uout[3] = uin[3] / uout[0] + GG[0][3] / GG[0][0];
            },
        },
    }
    return uout;
}

pub fn ConCovOf(comptime T: type) type {
    return struct { con: [4]T, cov: [4]T };
}
pub const ConCov = ConCovOf(f64);

/// C: conv_vels_both. Only to == VEL4 is supported; u^t is recomputed.
pub fn convertBoth(uin: [4]f64, from: VelType, geom: *const Geometry) Error!ConCov {
    const con = try convert(uin, from, .vel4, geom, .recompute_ut);
    return .{ .con = con, .cov = lowerVec(con, geom) };
}

/// conv_vels_both for VELR input over lane type T (infallible, see
/// velrToVel4G).
pub fn convertBothVelrG(comptime T: type, uin: [4]T, gg: MetricCovOf(T), GG: MetricConOf(T)) ConCovOf(T) {
    const con = velrToVel4G(T, uin, gg, GG);
    return .{ .con = con, .cov = lowerVecG(T, con, gg) };
}

/// Gas u^μ, u_μ from primitives' velocity slots (C: calc_ucon_ucov_from_prims;
/// VELPRIM == VELR). `v` are the three VELPRIM components pp[VX..VZ].
pub fn uconUcovFromPrims(v: [3]f64, geom: *const Geometry) Error!ConCov {
    return convertBoth(.{ 0, v[0], v[1], v[2] }, .velr, geom);
}

/// uconUcovFromPrims over lane type T.
pub fn uconUcovFromPrimsG(comptime T: type, v: [3]T, gg: MetricCovOf(T), GG: MetricConOf(T)) ConCovOf(T) {
    return convertBothVelrG(T, .{ simd.splat(T, 0), v[0], v[1], v[2] }, gg, GG);
}

//
// ---- normal observer (C: relele.c:447-509) --------------------------------
//

/// n^μ = -α g^{μ0} with the precomputed lapse (C: calc_normalobs_ncon).
/// Kept separate from normalObs4vel: Geometry.alpha is @sqrt(-1.0/g^tt)
/// while normalObs4vel derives 1.0/@sqrt(-g^tt); the two round differently.
pub fn normalObsCon(geom: *const Geometry) [4]f64 {
    var ncon: [4]f64 = undefined;
    for (0..4) |i| ncon[i] = -geom.alpha * geom.GG[i][0];
    return ncon;
}

/// n^μ with α computed from g^tt (C: calc_normalobs_4vel).
pub fn normalObs4vel(geom: *const Geometry) [4]f64 {
    const alp = 1.0 / @sqrt(-geom.GG[0][0]);
    return raiseVec(.{ -alp, 0, 0, 0 }, geom);
}

/// VELR of the normal observer (C: calc_normalobs_relvel); identically 0
/// in exact arithmetic; kept for parity.
pub fn normalObsVelr(geom: *const Geometry) [4]f64 {
    const GG = &geom.GG;
    const ucon = normalObs4vel(geom);
    var ncon: [4]f64 = ucon;
    for (1..4) |i| ncon[i] = ucon[i] - ucon[0] * GG[0][i] / GG[0][0];
    return ncon;
}
