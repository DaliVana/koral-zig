//! M12 theory gates for the mean-field dynamo (koral/sim/dynamo.zig).
//!
//! Physics identities that must hold regardless of the exact numerics (the
//! C golden in golden_dynamo_test.zig carries the formula-exactness):
//!
//!  * DAMPBETA (dampBphi): dB^φ opposes B^φ, is zero below β_sat, grows with
//!    β above it, and is clamped against overshoot, so |B^φ| never grows and
//!    its sign never flips (saturation + monotonicity toward 0). Tested
//!    directly on the pure formula (the t=0 torus starts at β=1/20 with the
//!    b²/ρ floor capping β<β_sat, so it does not itself trigger damping;
//!    physically correct: the dynamo saturates MRI-amplified fields *later*).
//!  * ALPHAFLIPSSIGN: ΔA_φ ∝ effalpha·B^φ with effalpha = −(π/2−θ)·(…>0), so
//!    sign(ΔA_φ) = sign(−(π/2−θ)·B^φ); the α-effect flips across the equator.
//!  * The dynamo acts through a vector potential, so the added poloidal field
//!    is div-free: divB stays at its (≈0) initial level.
//!  * calcScaleHeight reproduces a density-weighted RMS on a known profile.

const std = @import("std");
const config = @import("../config.zig");
const sim_mod = @import("../sim.zig");
const puffy = @import("../problems/puffy/puffy.zig");
const dynamo = @import("../sim/dynamo.zig");
const ct = @import("../sim/ct.zig");
const coco = @import("../metric/coco.zig");
const precompute = @import("../metric/precompute.zig");
const Geometry = @import("../geometry.zig").Geometry;
const p2u_mod = @import("../p2u.zig");
const invert = @import("../solve/invert.zig");
const invert_rad = @import("../solve/invert_rad.zig");
const radforce = @import("../physics/radforce.zig");

const SimP = sim_mod.Sim(config.puffy);
const LP = SimP.Layout;
const pi = std.math.pi;

/// injected toroidal field B³ = factor·B² (gives the dynamo a field to act on
/// and drives the ΔA_φ sign-flip); the C harness uses the same factor.
const inject_factor: f64 = 10.0;

fn dynamoOptions(nthreads: usize) SimP.Options {
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
        .nthreads = nthreads,
    };
}

/// inject B³ = inject_factor·B² into the domain (u kept consistent) so the
/// dynamo has a toroidal field to act on; shared by the sign-flip and
/// threading tests; deterministic, so both sims get identical states.
fn injectB3(s: *SimP) !void {
    const b2 = LP.index(.b2);
    const b3 = LP.index(.b3);
    var iy: i64 = 0;
    while (iy < s.nyi()) : (iy += 1) {
        var ix: i64 = 0;
        while (ix < s.nxi()) : (ix += 1) {
            var pp: [SimP.nv]f64 = undefined;
            s.p.load(ix, iy, 0, &pp);
            pp[b3] = inject_factor * pp[b2];
            const geom = s.cache.fillGeometry(ix, iy, 0);
            const uu = try p2u_mod.p2u(config.puffy, pp, &geom, s.opt.gam);
            s.p.store(ix, iy, 0, &pp);
            s.u.store(ix, iy, 0, &uu);
        }
    }
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

fn expectBits(x: f64, y: f64) !void {
    try std.testing.expectEqual(@as(u64, @bitCast(x)), @as(u64, @bitCast(y)));
}

/// Every numeric field of two Geometry blocks agrees bit-for-bit.
fn expectGeomBits(c: *const Geometry, f: *const Geometry) !void {
    try std.testing.expectEqual(c.coords, f.coords);
    for (0..4) |i| try expectBits(c.xxvec[i], f.xxvec[i]);
    for (0..4) |i| for (0..5) |j| {
        try expectBits(c.gg[i][j], f.gg[i][j]);
        try expectBits(c.GG[i][j], f.GG[i][j]);
    };
    try expectBits(c.gdet, f.gdet);
    try expectBits(c.alpha, f.alpha);
    try expectBits(c.gttpert, f.gttpert);
}

// Finding #1: the BL geometry, MYCOORDS→BL Jacobian, and gdet the dynamo /
// radviscosity passes now read from the MetricCache sidecar must be BYTE-FOR-
// BYTE equal to recomputing them per cell (geometryBLat / coco.dxdx /
// fillGeometry().gdet) — the guarantee that goldens are unaffected. Checked on
// every cell including the full ghost frame the passes touch.
test "M12 dynamo: BL geom / Jacobian / gdet caches are bit-identical to recompute (finding #1)" {
    const a = std.testing.allocator;
    var s = try SimP.init(a, puffy.makeGrid(48, 44), dynamoOptions(1));
    defer s.deinit();
    _ = try puffy.initAll(SimP, &s);

    const g = &s.grid;
    const mp = s.opt.mp;
    const ngx: i64 = @intCast(g.ngx);
    const ngy: i64 = @intCast(g.ngy);
    const ngz: i64 = @intCast(g.ngz);

    var nchecked: usize = 0;
    var iz: i64 = -ngz;
    while (iz < s.nzi() + ngz) : (iz += 1) {
        var iy: i64 = -ngy;
        while (iy < s.nyi() + ngy) : (iy += 1) {
            var ix: i64 = -ngx;
            while (ix < s.nxi() + ngx) : (ix += 1) {
                // (1) cached BL geometry == fresh geometryBLat, all fields
                const fresh = precompute.geometryBLat(g, .mks2, mp, ix, iy, iz);
                try expectGeomBits(s.cache.blGeom(ix, iy, iz), &fresh);

                // (2) cached MKS2→BL Jacobian == coco.dxdx at the cell center
                const xx = [4]f64{ 0, g.xc(ix), g.yc(iy), g.zc(iz) };
                const jf = coco.dxdx(xx, .mks2, .bl, mp);
                const jc = s.cache.jacMy2Bl(ix, iy, iz);
                for (0..4) |i| for (0..4) |k| try expectBits(jf[i][k], jc[i][k]);

                // (3) gdet accessor == fillGeometry().gdet
                try expectBits(s.cache.fillGeometry(ix, iy, iz).gdet, s.cache.gdet(ix, iy, iz));
                nchecked += 1;
            }
        }
    }
    std.debug.print("dynamo BL-cache bit-identity: {d} cells verified\n", .{nchecked});
    try std.testing.expect(nchecked > 2000);
}

test "M12 dynamo: ΔA_φ equatorial sign flip + |B³| non-increasing + divB" {
    const a = std.testing.allocator;
    var s = try SimP.init(a, puffy.makeGrid(48, 44), dynamoOptions(1));
    defer s.deinit();
    _ = try puffy.initAll(SimP, &s);

    const b3 = LP.index(.b3);
    const nx = s.nxi();
    const ny = s.nyi();

    // inject B³ = factor·B², keep u consistent
    var b3_before = try a.alloc(f64, @intCast(nx * ny));
    defer a.free(b3_before);
    try injectB3(&s);
    {
        var iy: i64 = 0;
        while (iy < ny) : (iy += 1) {
            var ix: i64 = 0;
            while (ix < nx) : (ix += 1) {
                b3_before[@intCast(iy * nx + ix)] = s.p.get(b3, ix, iy, 0);
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
                const aphi = s.dyn_a.get(2, ix, iy, 0);
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
    var s = try SimP.init(a, puffy.makeGrid(32, 40), dynamoOptions(1));
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

test "P1 dynamo threading: applyDynamo nthreads=4 is bit-identical to nthreads=1" {
    // The dynamo's parallel passes (per-ix scale-height columns, the ΔA_φ +
    // DAMPBETA ring loop, the superpose+p2u loop) all write only their own
    // cell/column, so any band split must reproduce the serial bits exactly —
    // same contract as the u2p/implicit threading gate, on a real MKS2 torus
    // (which also runs the threaded calc_u2p with polar-corrected cells).
    const a = std.testing.allocator;

    var s1 = try SimP.init(a, puffy.makeGrid(48, 44), dynamoOptions(1));
    defer s1.deinit();
    var s4 = try SimP.init(a, puffy.makeGrid(48, 44), dynamoOptions(4));
    defer s4.deinit();

    _ = try puffy.initAll(SimP, &s1);
    _ = try puffy.initAll(SimP, &s4);
    try injectB3(&s1);
    try injectB3(&s4);

    // two applications, as in one full RK2IMEX step
    for (0..2) |_| {
        try dynamo.applyDynamo(SimP, &s1, 0.0, 10.0);
        try dynamo.applyDynamo(SimP, &s4, 0.0, 10.0);
    }

    // full arrays (domain + ghosts: the ΔA_φ ring damps ghost B³ too),
    // the scale-height reduction, and the ΔA_φ scratch — all to the bit
    for (s1.p.data, s4.p.data) |v1, v4| try std.testing.expectEqual(v1, v4);
    for (s1.u.data, s4.u.data) |v1, v4| try std.testing.expectEqual(v1, v4);
    for (s1.dyn_a.data, s4.dyn_a.data) |v1, v4| try std.testing.expectEqual(v1, v4);
    for (s1.scaleth, s4.scaleth) |v1, v4| try std.testing.expectEqual(v1, v4);
}

// The dynamo *law* was extracted from the mimic_dynamo grid loop into the pure
// dynamoDeltaA kernel (sim/dynamo.zig). These gates drive it directly with
// hand-built cell states — no Sim, no metric cache — so they pin the physics a
// scientist edits (α sign flip, the height window, the horizon cutoff) without a
// 384×360 golden. Schwarzschild (a=0) BL radii: horizon 2, ISCO 6. The full-Sim
// golden still carries formula-exactness against C; these carry the invariants.
const rhor0: f64 = 2.0;
const risco0: f64 = 6.0;

/// A mid-disk cell that lands squarely in the dynamo-active window (r past the
/// ISCO, |θ−π/2| inside the scale height, field pitch below THETAANGLE).
fn activeCell(th: f64, bphi: f64) dynamo.DynamoCell {
    return .{
        .r = 10.0,
        .th = th,
        .bphi = bphi,
        .bsq = 1.0,
        .angle = 0.0, // < THETAANGLE ⇒ facangle = 1
        .prermhd = 1.0,
        .gg33 = 100.0,
        .scaleth = 0.0, // unused: these gates run with calchronthego = false
    };
}

test "M12 dynamo law: ΔA_φ flips sign across the equator (ALPHAFLIPSSIGN, pure kernel)" {
    // expectedhr path so hrdtheta is a fixed 0.3·π/2 — no scale-height gather.
    const dp = dynamo.Params{ .calchronthego = false, .dampbeta = false };
    const dth = 0.1; // both cells inside the scale height ⇒ faczh > 0

    const north = dynamo.dynamoDeltaA(dp, 0.0, 1.0, rhor0, risco0, activeCell(pi / 2.0 - dth, 1e-3));
    const south = dynamo.dynamoDeltaA(dp, 0.0, 1.0, rhor0, risco0, activeCell(pi / 2.0 + dth, 1e-3));

    try std.testing.expect(!north.skip and !south.skip);
    // effalpha = −(π/2−θ)·(…>0), everything else is even in (π/2−θ) and > 0, so
    // ΔA_φ is odd across the midplane: opposite signs, equal magnitude.
    try std.testing.expect(north.aphi < 0.0);
    try std.testing.expect(south.aphi > 0.0);
    try std.testing.expectApproxEqRel(north.aphi, -south.aphi, 1e-12);
}

test "M12 dynamo law: the height window switches ΔA_φ off beyond the scale height" {
    const dp = dynamo.Params{ .calchronthego = false, .dampbeta = true };
    const hrdtheta = dp.expectedhr * pi / 2.0;

    // just inside the scale height (|zh| < 1) the dynamo is live…
    const inside = dynamo.dynamoDeltaA(dp, 0.0, 1.0, rhor0, risco0, activeCell(pi / 2.0 - 0.5 * hrdtheta, 1e-3));
    try std.testing.expect(inside.aphi != 0.0);

    // …past it (|zh| > 1) faczh = max(0, 1−zh²) = 0, so both ΔA_φ and the
    // DAMPBETA increment vanish — the field is left untouched.
    const outside = dynamo.dynamoDeltaA(dp, 0.0, 1.0, rhor0, risco0, activeCell(pi / 2.0 - 1.5 * hrdtheta, 1e-3));
    try std.testing.expectEqual(@as(f64, 0.0), outside.aphi);
    try std.testing.expectEqual(@as(f64, 1e-3), outside.bphi); // B^φ unchanged
}

test "M12 dynamo law: cells inside 1.0001·r_horizon are skipped" {
    const dp = dynamo.Params{ .calchronthego = false };
    var c = activeCell(pi / 2.0 - 0.1, 7e-4);
    c.r = rhor0; // exactly on the horizon ⇒ inside 1.0001·rhor
    const out = dynamo.dynamoDeltaA(dp, 0.0, 1.0, rhor0, risco0, c);
    try std.testing.expect(out.skip);
    try std.testing.expectEqual(@as(f64, 0.0), out.aphi);
    try std.testing.expectEqual(@as(f64, 7e-4), out.bphi); // untouched
}
