//! Velocity conversions and index gymnastics (C: relele.c, frames.c tail).
//!
//! KORAL threads three velocity representations through the code
//! (mnemonics.h): VEL4 — lab-frame four-velocity u^μ; VEL3 — lab
//! three-velocity u^i/u^t; VELR — relative velocity ũ^i measured by the
//! normal observer. Primitives store VELPRIM = VELR (choices.h:103).
//!
//! All functions take the metric as C's 4×5 blocks (`gg`, `GG`) so the
//! arithmetic transcribes 1:1 from relele.c; the 5th column is ignored.

const std = @import("std");
const simd = @import("math/simd.zig");
const Geometry = @import("geometry.zig").Geometry;

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

pub const Error = error{
    /// C prints "ut.nan in conv_vels" and returns -1 (relele.c:199).
    VelocityConversionFailed,
    /// C calls my_err("delta.lt.0 in fill_utinucon") and would continue
    /// into sqrt(<0) = NaN (relele.c:394); we refuse instead.
    SpacelikeVelocity,
    /// A NaN reached the assembled face flux (physics/flux.zig). Same spirit
    /// as the two above — we refuse to propagate an unphysical state rather
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
pub fn indices12(a1: [4]f64, GG: *const [4][5]f64) [4]f64 {
    return indices12G(f64, a1, GG);
}

pub fn indices12G(comptime T: type, a1: [4]T, GG: *const [4][5]T) [4]T {
    var a2: [4]T = @splat(simd.splat(T, 0));
    for (0..4) |i| {
        for (0..4) |k| {
            a2[i] += a1[k] * GG[i][k];
        }
    }
    return a2;
}

/// A^μ -> A_μ (C: indices_21).
pub fn indices21(a1: [4]f64, gg: *const [4][5]f64) [4]f64 {
    return indices21G(f64, a1, gg);
}

pub fn indices21G(comptime T: type, a1: [4]T, gg: *const [4][5]T) [4]T {
    var a2: [4]T = @splat(simd.splat(T, 0));
    for (0..4) |i| {
        for (0..4) |j| {
            a2[i] += a1[j] * gg[i][j];
        }
    }
    return a2;
}

/// T_μν -> T^μν (C: indices_1122).
pub fn indices1122(t1: [4][4]f64, GG: *const [4][5]f64) [4][4]f64 {
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
pub fn indices2211(t1: [4][4]f64, gg: *const [4][5]f64) [4][4]f64 {
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

/// T^μ_ν -> T^μν (C: indices_2122).
pub fn indices2122(t1: [4][4]f64, GG: *const [4][5]f64) [4][4]f64 {
    var t2: [4][4]f64 = @splat(@splat(0));
    for (0..4) |i| {
        for (0..4) |j| {
            for (0..4) |k| {
                t2[i][j] += t1[i][k] * GG[k][j];
            }
        }
    }
    return t2;
}

/// T^μν -> T^μ_ν (C: indices_2221).
pub fn indices2221(t1: [4][4]f64, gg: *const [4][5]f64) [4][4]f64 {
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

/// A single row of indices2221: T^row_j = Σ_k T^{row k} g_{kj}, for j=0..3.
/// Same inner k-summation as indices2221, so the result is bitwise-identical to
/// `indices2221(t1, gg)[row]` — 16 madds instead of 64 when a caller reads only
/// one lowered row (flux.fFluxPrime consumes row idim+1 of both stress tensors,
/// P2 #7).
pub fn indices2221Row(t1: [4][4]f64, gg: *const [4][5]f64, row: usize) [4]f64 {
    var out: [4]f64 = @splat(0);
    for (0..4) |j| {
        for (0..4) |k| {
            out[j] += t1[row][k] * gg[k][j];
        }
    }
    return out;
}

/// uout^μ = A^μ_ν uin^ν (C: frames.c multiply2). Alias-safe.
pub fn multiply2(uin: [4]f64, a: [4][4]f64) [4]f64 {
    return multiply2G(f64, uin, a);
}

pub fn multiply2G(comptime T: type, uin: [4]T, a: [4][4]T) [4]T {
    var uout: [4]T = @splat(simd.splat(T, 0));
    for (0..4) |i| {
        for (0..4) |j| {
            uout[i] += a[i][j] * uin[j];
        }
    }
    return uout;
}

/// T2^μν = A^μ_κ A^ν_λ T1^κλ (C: frames.c multiply22). Alias-safe.
pub fn multiply22(t1: [4][4]f64, a: [4][4]f64) [4][4]f64 {
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

/// u^t from spatial VEL3 components: u^t = sqrt(-1/(g00 + 2 g0i v^i + gij v^i v^j)).
pub fn utInVel3(v: [4]f64, gg: *const [4][5]f64) f64 {
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

/// u^t from spatial four-velocity components; solves g00 ut² + 2b ut + c = 0
/// taking the (-b - sqrt(Δ))/g00 root in both signs of g00 (C keeps the
/// minus root in the ergoregion too, relele.c:407).
pub fn utInUcon(u: [4]f64, gg: *const [4][5]f64) Error!f64 {
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
/// C prints and returns 1 — mirrored.
pub fn calcAlpgam(u: [4]f64, gg: *const [4][5]f64, GG: *const [4][5]f64) f64 {
    return calcAlpgamG(f64, u, gg, GG);
}

pub fn calcAlpgamG(comptime T: type, u: [4]T, gg: *const [4][5]T, GG: *const [4][5]T) T {
    const sp = simd.splat;
    var qsq: T = sp(T, 0);
    for (1..4) |i| {
        for (1..4) |j| {
            qsq += u[i] * u[j] * gg[i][j];
        }
    }
    const alpgam2 = (sp(T, -1.0) / GG[0][0]) * (sp(T, 1.0) + qsq);
    return simd.select(T, alpgam2 < sp(T, 0), sp(T, 1.0), @sqrt(alpgam2));
}

/// The VELR → VEL4 conversion (convVelsCore's velr branch) — infallible:
/// calcAlpgam absorbs alpgam² < 0 by returning 1, C-faithfully. This is
/// the only conv_vels direction the T-generic radiative chain needs.
pub fn velrToVel4G(comptime T: type, uin: [4]T, gg: *const [4][5]T, GG: *const [4][5]T) [4]T {
    const alpgam = calcAlpgamG(T, uin, gg, GG);
    return .{
        -alpgam * GG[0][0],
        uin[1] - alpgam * GG[0][1],
        uin[2] - alpgam * GG[0][2],
        uin[3] - alpgam * GG[0][3],
    };
}

//
// ---- conv_vels (C: relele.c:136 conv_vels_core) ---------------------------
//

/// Contravariant velocity conversion; u[0] of the input is ignored unless
/// `ut_known` (C: conv_vels vs conv_vels_ut). Returns the converted 4-vector.
pub fn convVelsCore(
    uin: [4]f64,
    from: VelType,
    to: VelType,
    gg: *const [4][5]f64,
    GG: *const [4][5]f64,
    ut_known: bool,
) Error![4]f64 {
    if (from == to) {
        // VEL4 -> VEL4 recomputes u^t when unknown; VEL3/VELR copy through.
        var uout = uin;
        if (from == .vel4 and !ut_known) uout[0] = try utInUcon(uin, gg);
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
                const ut = if (ut_known) uin[0] else try utInUcon(uin, gg);
                uout = .{ 1.0, uin[1] / ut, uin[2] / ut, uin[3] / ut };
            },
            .velr => {
                const ut = if (ut_known) uin[0] else try utInUcon(uin, gg);
                uout[0] = ut;
                for (1..4) |i| {
                    uout[i] = uin[i] - ut * GG[0][i] / GG[0][0];
                }
            },
        },
        .vel3 => switch (to) {
            .vel3 => unreachable,
            .vel4 => {
                const ut = utInVel3(uin, gg);
                if (ut < 1.0 or std.math.isNan(ut)) return Error.VelocityConversionFailed;
                uout = .{ ut, uin[1] * ut, uin[2] * ut, uin[3] * ut };
            },
            .velr => {
                const ut = utInVel3(uin, gg);
                if (ut < 1.0 or std.math.isNan(ut)) return Error.VelocityConversionFailed;
                uout = .{ ut, uin[1] * ut, uin[2] * ut, uin[3] * ut };
                for (1..4) |i| {
                    uout[i] = uout[i] - uout[0] * GG[0][i] / GG[0][0];
                }
            },
        },
        .velr => switch (to) {
            .velr => unreachable,
            .vel4 => uout = velrToVel4G(f64, uin, gg, GG),
            .vel3 => {
                const alpgam = calcAlpgam(uin, gg, GG);
                uout[0] = -alpgam * GG[0][0];
                uout[1] = uin[1] / uout[0] + GG[0][1] / GG[0][0];
                uout[2] = uin[2] / uout[0] + GG[0][2] / GG[0][0];
                uout[3] = uin[3] / uout[0] + GG[0][3] / GG[0][0];
            },
        },
    }
    return uout;
}

/// C: conv_vels — u^t of the input not trusted.
pub fn convVels(uin: [4]f64, from: VelType, to: VelType, gg: *const [4][5]f64, GG: *const [4][5]f64) Error![4]f64 {
    return convVelsCore(uin, from, to, gg, GG, false);
}

/// C: conv_vels_ut — u^t of the input is already correct.
pub fn convVelsUt(uin: [4]f64, from: VelType, to: VelType, gg: *const [4][5]f64, GG: *const [4][5]f64) Error![4]f64 {
    return convVelsCore(uin, from, to, gg, GG, true);
}

pub fn ConCovOf(comptime T: type) type {
    return struct { con: [4]T, cov: [4]T };
}
pub const ConCov = ConCovOf(f64);

/// C: conv_vels_both — only to == VEL4 is supported.
pub fn convVelsBoth(uin: [4]f64, from: VelType, gg: *const [4][5]f64, GG: *const [4][5]f64) Error!ConCov {
    const con = try convVelsCore(uin, from, .vel4, gg, GG, false);
    return .{ .con = con, .cov = indices21(con, gg) };
}

/// conv_vels_both for VELR input over lane type T (infallible, see
/// velrToVel4G).
pub fn convVelsBothVelrG(comptime T: type, uin: [4]T, gg: *const [4][5]T, GG: *const [4][5]T) ConCovOf(T) {
    const con = velrToVel4G(T, uin, gg, GG);
    return .{ .con = con, .cov = indices21G(T, con, gg) };
}

/// Gas u^μ, u_μ from primitives' velocity slots (C: calc_ucon_ucov_from_prims;
/// VELPRIM == VELR). `v` are the three VELPRIM components pp[VX..VZ].
pub fn uconUcovFromPrims(v: [3]f64, geom: *const Geometry) Error!ConCov {
    return convVelsBoth(.{ 0, v[0], v[1], v[2] }, .velr, &geom.gg, &geom.GG);
}

/// uconUcovFromPrims over lane type T.
pub fn uconUcovFromPrimsG(comptime T: type, v: [3]T, gg: *const [4][5]T, GG: *const [4][5]T) ConCovOf(T) {
    return convVelsBothVelrG(T, .{ simd.splat(T, 0), v[0], v[1], v[2] }, gg, GG);
}

//
// ---- normal observer (C: relele.c:447-509) --------------------------------
//

/// n^μ = -α g^{μ0} (C: calc_normalobs_ncon).
pub fn normalObsNcon(GG: *const [4][5]f64, alpha: f64) [4]f64 {
    var ncon: [4]f64 = undefined;
    for (0..4) |i| ncon[i] = -alpha * GG[i][0];
    return ncon;
}

/// n^μ with α computed from g^tt (C: calc_normalobs_4vel).
pub fn normalObs4vel(GG: *const [4][5]f64) [4]f64 {
    const alp = 1.0 / @sqrt(-GG[0][0]);
    return indices12(.{ -alp, 0, 0, 0 }, GG);
}

/// VELR of the normal observer (C: calc_normalobs_relvel) — identically 0
/// in exact arithmetic; kept for parity.
pub fn normalObsRelvel(GG: *const [4][5]f64) [4]f64 {
    const ucon = normalObs4vel(GG);
    var ncon: [4]f64 = ucon;
    for (1..4) |i| ncon[i] = ucon[i] - ucon[0] * GG[0][i] / GG[0][0];
    return ncon;
}
