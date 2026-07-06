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
const Geometry = @import("geometry.zig").Geometry;

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
};

pub inline fn dot(a: [4]f64, b: [4]f64) f64 {
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
    var a2: [4]f64 = @splat(0);
    for (0..4) |i| {
        for (0..4) |k| {
            a2[i] += a1[k] * GG[i][k];
        }
    }
    return a2;
}

/// A^μ -> A_μ (C: indices_21).
pub fn indices21(a1: [4]f64, gg: *const [4][5]f64) [4]f64 {
    var a2: [4]f64 = @splat(0);
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

/// uout^μ = A^μ_ν uin^ν (C: frames.c multiply2). Alias-safe.
pub fn multiply2(uin: [4]f64, a: [4][4]f64) [4]f64 {
    var uout: [4]f64 = @splat(0);
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
    var qsq: f64 = 0;
    for (1..4) |i| {
        for (1..4) |j| {
            qsq += u[i] * u[j] * gg[i][j];
        }
    }
    const alpgam2 = (-1.0 / GG[0][0]) * (1.0 + qsq);
    if (alpgam2 < 0) return 1.0;
    return @sqrt(alpgam2);
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
    var uout: [4]f64 = undefined;

    if (from == to) {
        // VEL4 -> VEL4 recomputes u^t when unknown; VEL3/VELR copy through.
        uout = uin;
        if (from == .vel4 and !ut_known) uout[0] = try utInUcon(uin, gg);
    } else if (from == .vel4 and to == .vel3) {
        const ut = if (ut_known) uin[0] else try utInUcon(uin, gg);
        uout = .{ 1.0, uin[1] / ut, uin[2] / ut, uin[3] / ut };
    } else if (from == .vel3 and to == .vel4) {
        const ut = utInVel3(uin, gg);
        if (ut < 1.0 or std.math.isNan(ut)) return Error.VelocityConversionFailed;
        uout = .{ ut, uin[1] * ut, uin[2] * ut, uin[3] * ut };
    } else if (from == .vel3 and to == .velr) {
        const ut = utInVel3(uin, gg);
        if (ut < 1.0 or std.math.isNan(ut)) return Error.VelocityConversionFailed;
        uout = .{ ut, uin[1] * ut, uin[2] * ut, uin[3] * ut };
        for (1..4) |i| {
            uout[i] = uout[i] - uout[0] * GG[0][i] / GG[0][0];
        }
    } else if (from == .vel4 and to == .velr) {
        const ut = if (ut_known) uin[0] else try utInUcon(uin, gg);
        uout[0] = ut;
        for (1..4) |i| {
            uout[i] = uin[i] - ut * GG[0][i] / GG[0][0];
        }
    } else if (from == .velr and to == .vel4) {
        const alpgam = calcAlpgam(uin, gg, GG);
        uout[0] = -alpgam * GG[0][0];
        uout[1] = uin[1] - alpgam * GG[0][1];
        uout[2] = uin[2] - alpgam * GG[0][2];
        uout[3] = uin[3] - alpgam * GG[0][3];
    } else { // VELR -> VEL3
        const alpgam = calcAlpgam(uin, gg, GG);
        uout[0] = -alpgam * GG[0][0];
        uout[1] = uin[1] / uout[0] + GG[0][1] / GG[0][0];
        uout[2] = uin[2] / uout[0] + GG[0][2] / GG[0][0];
        uout[3] = uin[3] / uout[0] + GG[0][3] / GG[0][0];
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

pub const ConCov = struct { con: [4]f64, cov: [4]f64 };

/// C: conv_vels_both — only to == VEL4 is supported.
pub fn convVelsBoth(uin: [4]f64, from: VelType, gg: *const [4][5]f64, GG: *const [4][5]f64) Error!ConCov {
    const con = try convVelsCore(uin, from, .vel4, gg, GG, false);
    return .{ .con = con, .cov = indices21(con, gg) };
}

/// Gas u^μ, u_μ from primitives' velocity slots (C: calc_ucon_ucov_from_prims;
/// VELPRIM == VELR). `v` are the three VELPRIM components pp[VX..VZ].
pub fn uconUcovFromPrims(v: [3]f64, geom: *const Geometry) Error!ConCov {
    return convVelsBoth(.{ 0, v[0], v[1], v[2] }, .velr, &geom.gg, &geom.GG);
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
