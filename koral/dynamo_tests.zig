//! M12 theory gates for the mean-field dynamo (koral/magn/dynamo.zig).
//!
//! Physics identities that must hold regardless of the exact numerics (the
//! C golden in golden_dynamo_test.zig carries the formula-exactness):
//!
//!  * DAMPBETA (dampBphi): dB^φ opposes B^φ, is zero below β_sat, grows with
//!    β above it, and is clamped against overshoot — so |B^φ| never grows and
//!    its sign never flips (saturation + monotonicity toward 0). Tested
//!    directly on the pure formula (the t=0 torus starts at β=1/20 with the
//!    b²/ρ floor capping β<β_sat, so it does not itself trigger damping —
//!    physically correct: the dynamo saturates MRI-amplified fields *later*).
//!  * ALPHAFLIPSSIGN: ΔA_φ ∝ effalpha·B^φ with effalpha = −(π/2−θ)·(…>0), so
//!    sign(ΔA_φ) = sign(−(π/2−θ)·B^φ) — the α-effect flips across the equator.
//!  * The dynamo acts through a vector potential, so the added poloidal field
//!    is div-free: divB stays at its (≈0) initial level.
//!  * calcScaleHeight reproduces a density-weighted RMS on a known profile.

const std = @import("std");
const config = @import("config.zig");
const sim_mod = @import("sim.zig");
const puffy = @import("problems/puffy.zig");
const dynamo = @import("magn/dynamo.zig");
const ct = @import("magn/ct.zig");
const coco = @import("metric/coco.zig");
const p2u_mod = @import("p2u.zig");
const invert = @import("solve/invert.zig");
const invert_rad = @import("solve/invert_rad.zig");
const radforce = @import("physics/radforce.zig");

const SimP = sim_mod.Sim(config.puffy);
const LP = SimP.Layout;
const pi = std.math.pi;

/// injected toroidal field B³ = factor·B² (gives the dynamo a field to act on
/// and drives the ΔA_φ sign-flip); the C harness uses the same factor.
const inject_factor: f64 = 10.0;

fn dynamoOptions() SimP.Options {
    return .{
        .coords = .mks2,
        .mp = puffy.mp,
        .gam = puffy.gam,
        .floors = invert.FloorParams.puffy,
        .rad = invert_rad.RadParams.puffy,
        .opac = radforce.Params.puffy(),
        .correct_polaraxis = true,
        .nccorrectpolar = 2,
        .dynamo = true,
        .bc_x = .specific,
        .bc_y = .specific,
        .specific_bc = &puffy.Bc(SimP).calc,
    };
}

fn thetaBL(s: *const SimP, ix: i64, iy: i64) f64 {
    const geom = s.cache.fillGeometry(ix, iy, 0);
    return coco.cocoN(geom.xxvec, .mks2, .bl, s.opt.mp)[2];
}

test "M12 dynamo: DAMPBETA saturation + monotonicity + no-overshoot (formula)" {
    const dp = dynamo.Params{};
    const ab = dp.alphabeta;
    const bsat = dp.betasaturated;

    // below saturation → no damping at all
    try std.testing.expectEqual(@as(f64, 0), dynamo.dampBphi(ab, 1, 1, 1e-3, 0.05, bsat, 2.0));
    try std.testing.expectEqual(@as(f64, 0), dynamo.dampBphi(ab, 1, 1, 1e-3, bsat, bsat, 2.0));

    // above saturation → dB^φ opposes B^φ, shrinks |B^φ|, preserves sign
    {
        const bphi = 2.0;
        const d = dynamo.dampBphi(ab, 1, 1, 1e-4, 0.3, bsat, bphi);
        try std.testing.expect(d < 0); // opposes positive B^φ
        try std.testing.expect(@abs(bphi + d) < @abs(bphi)); // |B^φ| shrinks
        try std.testing.expect((bphi + d) * bphi >= 0); // sign preserved
    }
    // negative B^φ → dB^φ positive (still opposing)
    try std.testing.expect(dynamo.dampBphi(ab, 1, 1, 1e-4, 0.3, bsat, -2.0) > 0);

    // strong forcing overshoots → clamped exactly to −B^φ (B^φ → 0, no flip)
    {
        const bphi = 2.0;
        const d = dynamo.dampBphi(ab, 1, 1, 1.0, 5.0, bsat, bphi); // huge dt/Pk
        try std.testing.expectEqual(-bphi, d);
    }
    // zero radius/height weight → no damping regardless of β
    try std.testing.expectEqual(@as(f64, 0), dynamo.dampBphi(ab, 0, 1, 1e-3, 5.0, bsat, 2.0));
    try std.testing.expectEqual(@as(f64, 0), dynamo.dampBphi(ab, 1, 0, 1e-3, 5.0, bsat, 2.0));

    // monotone in β above saturation (stronger β → stronger damping)
    const d1 = dynamo.dampBphi(ab, 1, 1, 1e-5, 0.2, bsat, 1.0);
    const d2 = dynamo.dampBphi(ab, 1, 1, 1e-5, 0.5, bsat, 1.0);
    try std.testing.expect(@abs(d2) > @abs(d1));
}

test "M12 dynamo: ΔA_φ equatorial sign flip + |B³| non-increasing + divB" {
    const a = std.testing.allocator;
    var s = try SimP.init(a, puffy.makeGrid(48, 44), dynamoOptions());
    defer s.deinit();
    _ = try puffy.initAll(SimP, &s);

    const b2 = LP.index(.b2);
    const b3 = LP.index(.b3);
    const nx = s.nxi();
    const ny = s.nyi();

    // inject B³ = factor·B², keep u consistent
    var b3_before = try a.alloc(f64, @intCast(nx * ny));
    defer a.free(b3_before);
    {
        var iy: i64 = 0;
        while (iy < ny) : (iy += 1) {
            var ix: i64 = 0;
            while (ix < nx) : (ix += 1) {
                var pp: [SimP.nv]f64 = undefined;
                s.p.load(ix, iy, 0, &pp);
                pp[b3] = inject_factor * pp[b2];
                b3_before[@intCast(iy * nx + ix)] = pp[b3];
                const geom = s.cache.fillGeometry(ix, iy, 0);
                const uu = try p2u_mod.p2u(config.puffy, pp, &geom, s.opt.gam);
                s.p.store(ix, iy, 0, &pp);
                s.u.store(ix, iy, 0, &uu);
            }
        }
    }

    // divB before injection response (toroidal B³ doesn't change 2D divB)
    var divb_before: f64 = 0;
    {
        var iy: i64 = 1;
        while (iy < ny) : (iy += 1) {
            var ix: i64 = 1;
            while (ix < nx) : (ix += 1) divb_before = @max(divb_before, @abs(ct.calcDivB(SimP, &s, ix, iy, 0)));
        }
    }

    // run the dynamo core directly (no final calc_u2p round-trip on B³)
    dynamo.calcScaleHeight(SimP, &s);
    try s.setBc(0.0, false);
    try dynamo.mimicDynamo(SimP, &s, 10.0);

    // -- DAMPBETA property: |B³| never grows, sign never flips --
    {
        var iy: i64 = 0;
        while (iy < ny) : (iy += 1) {
            var ix: i64 = 0;
            while (ix < nx) : (ix += 1) {
                const before = b3_before[@intCast(iy * nx + ix)];
                const after = s.p.get(b3, ix, iy, 0);
                try std.testing.expect(@abs(after) <= @abs(before) + 1e-30);
                try std.testing.expect(after * before >= 0.0);
            }
        }
    }

    // -- ALPHAFLIPSSIGN: sign(ΔA_φ) == sign(−(π/2−θ)·B^φ) where it acts --
    var checked: usize = 0;
    {
        var iy: i64 = 0;
        while (iy < ny) : (iy += 1) {
            var ix: i64 = 0;
            while (ix < nx) : (ix += 1) {
                const aphi = s.dynA.get(2, ix, iy, 0);
                const bphi = b3_before[@intCast(iy * nx + ix)];
                const expected = -(pi / 2.0 - thetaBL(&s, ix, iy)) * bphi;
                if (@abs(aphi) > 1e-40 and @abs(expected) > 1e-40) {
                    try std.testing.expect(aphi * expected > 0.0);
                    checked += 1;
                }
            }
        }
    }
    try std.testing.expect(checked > 0);

    // -- divB preservation: curl(ΔA_φ) is div-free --
    var divb_after: f64 = 0;
    {
        var iy: i64 = 1;
        while (iy < ny) : (iy += 1) {
            var ix: i64 = 1;
            while (ix < nx) : (ix += 1) divb_after = @max(divb_after, @abs(ct.calcDivB(SimP, &s, ix, iy, 0)));
        }
    }
    std.debug.print("dynamo: sign-flip checked {d} cells, divB before {e:.3} after {e:.3}\n", .{ checked, divb_before, divb_after });
    try std.testing.expect(divb_after <= @max(divb_before, 1e-10) * 100.0);
}

test "M12 dynamo: calcScaleHeight = density-weighted RMS |π/2−θ|" {
    const a = std.testing.allocator;
    var s = try SimP.init(a, puffy.makeGrid(32, 40), dynamoOptions());
    defer s.deinit();
    _ = try puffy.initAll(SimP, &s);

    dynamo.calcScaleHeight(SimP, &s);

    const nx = s.nxi();
    const ny = s.nyi();
    const ix: i64 = @divTrunc(nx, 2);
    var num: f64 = 0;
    var den: f64 = 0;
    var iy: i64 = 0;
    while (iy < ny) : (iy += 1) {
        const geom = s.cache.fillGeometry(ix, iy, 0);
        const xxbl = coco.cocoN(geom.xxvec, .mks2, .bl, s.opt.mp);
        const rho = s.p.get(LP.index(.rho), ix, iy, 0);
        const dth = pi / 2.0 - xxbl[2];
        num += rho * geom.gdet * dth * dth;
        den += rho * geom.gdet;
    }
    try std.testing.expectApproxEqRel(@sqrt(num / den), s.scaleth[@intCast(ix)], 1e-12);
    try std.testing.expect(s.scaleth[@intCast(ix)] > 0);
}
