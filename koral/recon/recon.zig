//! Cell-average → face reconstruction (C: avg2point, finite.c:97).
//!
//! Orders (C: INT_ORDER − reconstr_par, clamped at 0):
//!   0 — donor cell
//!   1 — linear with the minmod-θ limiter (C: FLUXLIMITER == 0; θ = 1 is
//!       MinMod, θ = 2 is MC; PUFFY runs θ = 1.5). The MC/Superbee variants
//!       of FLUXLIMITER are not used by any current problem and are not
//!       implemented.
//!   2 — PPM (Colella & Woodward 1984, eqs 1.6–1.10) on a five-point
//!       stencil with per-cell widths.
//!
//! `reconstr_par` reduces the order near boundaries at runtime exactly like C
//! (avg2point's reconstrpar): effective order = base − reconstr_par.

const std = @import("std");

pub const Faces = struct { ul: f64, ur: f64 };

/// Scalar reconstruction of one variable; the C routine loops this body
/// over NV. `u` is the five-point stencil {i-2 … i+2}, `dx` the five cell
/// widths, `order` the effective integration order (already reduced).
pub fn avg2pointScalar(u: [5]f64, dx: [5]f64, order: u8, theta: f64) Faces {
    return switch (order) {
        0 => .{ .ul = u[2], .ur = u[2] }, // donor cell
        1 => linearMinmod(u, theta),
        2 => ppm(u, dx),
        // `eff` is base − reconstr_par, both clamped to the 3-member
        // config.Reconstruction enum's 0/1/2 orders (sim.zig switches it
        // exhaustively), so nothing else can reach here.
        else => unreachable,
    };
}

/// Vector version over an NV-sized stencil (matches C's call shape).
pub fn avg2point(
    comptime nv: usize,
    um2: [nv]f64,
    um1: [nv]f64,
    uc0: [nv]f64,
    up1: [nv]f64,
    up2: [nv]f64,
    dx: [5]f64,
    base_order: u8,
    reconstr_par: u8,
    theta: f64,
) struct { ul: [nv]f64, ur: [nv]f64 } {
    const eff: u8 = if (reconstr_par >= base_order) 0 else base_order - reconstr_par;
    var ul: [nv]f64 = undefined;
    var ur: [nv]f64 = undefined;
    for (0..nv) |iv| {
        const f = avg2pointScalar(
            .{ um2[iv], um1[iv], uc0[iv], up1[iv], up2[iv] },
            dx,
            eff,
            theta,
        );
        ul[iv] = f.ul;
        ur[iv] = f.ur;
    }
    return .{ .ul = ul, .ur = ur };
}

/// C: finite.c:126-208 (Ramesh's rewrite: no divisions), FLUXLIMITER == 0.
fn linearMinmod(u: [5]f64, theta: f64) Faces {
    const deltam = u[2] - u[1];
    const deltap = u[3] - u[2];

    if (deltam * deltap <= 0.0) {
        // local extremum → donor cell
        return .{ .ul = u[2], .ur = u[2] };
    }
    const slope = if (deltam > 0.0)
        @min(@min(theta * deltam, 0.5 * (deltam + deltap)), theta * deltap)
    else
        @max(@max(theta * deltam, 0.5 * (deltam + deltap)), theta * deltap);
    return .{ .ul = u[2] - 0.5 * slope, .ur = u[2] + 0.5 * slope };
}

fn sign(x: f64) f64 {
    // C: my_sign — returns ±1 (0 → +1 is never hit: guarded by monotonicity)
    return if (x >= 0.0) 1.0 else -1.0;
}

/// C: finite.c:210-332 — PPM with non-uniform cell widths.
fn ppm(u: [5]f64, dx: [5]f64) Faces {
    const dxm2 = dx[0];
    const dxm1 = dx[1];
    const dx0 = dx[2];
    const dxp1 = dx[3];
    const dxp2 = dx[4];

    const dxp2_plus_dxp1 = dxp2 + dxp1;
    const dxp2_plus_dxp1_inv = 1.0 / dxp2_plus_dxp1;
    const dxp1_plus_dx0 = dxp1 + dx0;
    const dxp1_plus_dx0_inv = 1.0 / dxp1_plus_dx0;
    const dx0_plus_dxm1 = dx0 + dxm1;
    const dx0_plus_dxm1_inv = 1.0 / dx0_plus_dxm1;
    const dxm1_plus_dxm2 = dxm1 + dxm2;
    const dxm1_plus_dxm2_inv = 1.0 / dxm1_plus_dxm2;

    const dxm1_plus_dx0_plus_dxp1_inv = 1.0 / (dxm1 + dx0 + dxp1);
    const dx0_plus_dxp1_plus_dxp2_inv = 1.0 / (dx0 + dxp1 + dxp2);
    const dxm2_plus_dxm1_plus_dx0_inv = 1.0 / (dxm2 + dxm1 + dx0);

    const dx0_plus_twodxm1 = dx0 + 2.0 * dxm1;
    const dxp1_plus_twodx0 = dxp1 + 2.0 * dx0;
    const dxm1_plus_twodxm2 = dxm1 + 2.0 * dxm2;
    const twodxp1_plus_dx0 = 2.0 * dxp1 + dx0;
    const twodxp2_plus_dxp1 = 2.0 * dxp2 + dxp1;
    const twodx0_plus_dxm1 = 2.0 * dx0 + dxm1;

    // slopes Δa_{j-1}, Δa_j, Δa_{j+1} (C&W eq 1.7)
    var dri = dx0 * dxm1_plus_dx0_plus_dxp1_inv *
        (dx0_plus_twodxm1 * dxp1_plus_dx0_inv * (u[3] - u[2]) +
            twodxp1_plus_dx0 * dx0_plus_dxm1_inv * (u[2] - u[1]));
    var drip1 = dxp1 * dx0_plus_dxp1_plus_dxp2_inv *
        (dxp1_plus_twodx0 * dxp2_plus_dxp1_inv * (u[4] - u[3]) +
            twodxp2_plus_dxp1 * dxp1_plus_dx0_inv * (u[3] - u[2]));
    var drim1 = dxm1 * dxm2_plus_dxm1_plus_dx0_inv *
        (dxm1_plus_twodxm2 * dx0_plus_dxm1_inv * (u[2] - u[1]) +
            twodx0_plus_dxm1 * dxm1_plus_dxm2_inv * (u[1] - u[0]));

    // monotonize the slopes (C&W eq 1.8)
    if ((u[3] - u[2]) * (u[2] - u[1]) > 0.0) {
        dri = @min(@abs(dri), @min(2.0 * @abs(u[2] - u[1]), 2.0 * @abs(u[2] - u[3]))) * sign(dri);
    } else {
        dri = 0.0;
    }
    if ((u[4] - u[3]) * (u[3] - u[2]) > 0.0) {
        drip1 = @min(@abs(drip1), @min(2.0 * @abs(u[3] - u[2]), 2.0 * @abs(u[3] - u[4]))) * sign(drip1);
    } else {
        drip1 = 0.0;
    }
    if ((u[2] - u[1]) * (u[1] - u[0]) > 0.0) {
        drim1 = @min(@abs(drim1), @min(2.0 * @abs(u[1] - u[0]), 2.0 * @abs(u[1] - u[2]))) * sign(drim1);
    } else {
        drim1 = 0.0;
    }

    // right face a_{j+1/2} (C&W eq 1.6)
    var z1 = dx0_plus_dxm1 / dxp1_plus_twodx0;
    var z2 = dxp2_plus_dxp1 / twodxp1_plus_dx0;
    var dx_inv = 1.0 / (dxm1 + dx0 + dxp1 + dxp2);
    var ur = u[2] + dx0 * dxp1_plus_dx0_inv * (u[3] - u[2]) +
        dx_inv * ((2.0 * dxp1 * dx0) * dxp1_plus_dx0_inv * (z1 - z2) * (u[3] - u[2]) -
            dx0 * z1 * drip1 + dxp1 * z2 * dri);

    // left face a_{j-1/2}
    z1 = dxm1_plus_dxm2 / dx0_plus_twodxm1;
    z2 = dxp1_plus_dx0 / twodx0_plus_dxm1;
    dx_inv = 1.0 / (dxm2 + dxm1 + dx0 + dxp1);
    var ul = u[1] + dxm1 * dx0_plus_dxm1_inv * (u[2] - u[1]) +
        dx_inv * ((2.0 * dx0 * dxm1) * dx0_plus_dxm1_inv * (z1 - z2) * (u[2] - u[1]) -
            dxm1 * z1 * dri + dx0 * z2 * drim1);

    // keep the parabola monotonic (equivalent of C&W eq 1.10)
    if ((ur - u[2]) * (u[2] - ul) <= 0.0) {
        ul = u[2];
        ur = u[2];
    }
    if ((ur - ul) * (ul - (3.0 * u[2] - 2.0 * ur)) < 0.0) {
        ul = 3.0 * u[2] - 2.0 * ur;
    }
    if ((ur - ul) * ((3.0 * u[2] - 2.0 * ul) - ur) < 0.0) {
        ur = 3.0 * u[2] - 2.0 * ul;
    }
    return .{ .ul = ul, .ur = ur };
}
