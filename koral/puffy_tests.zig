//! M11 theory gates for the PUFFY problem code (problems/puffy/puffy.zig):
//!
//!   * the GK21 ln f quadrature against a test-local tanh-sinh integrator
//!   * lamBL / findRml bisection residuals at the 5ε target
//!   * the ℓ(λ) broken power law + the ω cross-formula (omega3d vs the
//!     independently transcribed expression inside lnf_integrand)
//!   * the LTE pressure-split quartic residual
//!   * torus surface: inner edge pinned at LT_RIN, polytropic ρ ∝ ε³
//!     scaling at the outer surface
//!   * reduced-grid full init: post-postinit max β ≡ MAXBETA, A-derived B
//!     divergence-free, and the bc.c contract (polar mirror maps, XBCLO
//!     copy, XBCHI r²-rescaling and no-inflow in BL)
//!
//! The C diff of the same pipeline is golden_puffy_test.zig (keystone).

const std = @import("std");
const build_options = @import("build_options");
const config = @import("config.zig");
const grid_mod = @import("grid.zig");
const sim_mod = @import("sim.zig");
const relele = @import("relele.zig");
const frames = @import("frames.zig");
const quad = @import("math/quad.zig");
const puffy = @import("problems/puffy/puffy.zig");
const thermo = @import("physics/thermo.zig");
const mhd = @import("physics/bfield.zig");
const radiation = @import("physics/radiation.zig");
const ct = @import("magn/ct.zig");

const Grid = grid_mod.Grid;
const pi_2: f64 = std.math.pi / 2.0;

// ---------------------------------------------------------------------------
// tanh-sinh cross-check integrator (test-local, independent of math/quad.zig)

fn tanhSinh(f: anytype, a: f64, b: f64) f64 {
    const c = 0.5 * (a + b);
    const w = 0.5 * (b - a);
    const tmax: f64 = 3.6;
    var h: f64 = 0.5;
    var prev: f64 = std.math.inf(f64);
    var value: f64 = 0.0;
    var lvl: usize = 0;
    while (lvl < 10) : (lvl += 1) {
        const n: i64 = @intFromFloat(@floor(tmax / h));
        var sum: f64 = 0.0;
        var k: i64 = -n;
        while (k <= n) : (k += 1) {
            const t = @as(f64, @floatFromInt(k)) * h;
            const s = std.math.sinh(t);
            const ch = std.math.cosh(pi_2 * s);
            const x = c + w * std.math.tanh(pi_2 * s);
            const wt = pi_2 * std.math.cosh(t) / (ch * ch);
            sum += f.eval(x) * wt;
        }
        value = w * h * sum;
        if (@abs(value - prev) <= 1e-13 * @abs(value)) return value;
        prev = value;
        h *= 0.5;
    }
    return value;
}

// sample torus points (r, θ) with r·sinθ ≥ LT_RIN and rml < LT_R2 so the
// integrand is analytic on [rin, rml] (tanh-sinh needs no interior kink)
const samples = [_][2]f64{
    .{ 50.0, pi_2 },
    .{ 40.0, 1.5 },
    .{ 100.0, 1.2 },
    .{ 200.0, 1.4 },
    .{ 300.0, pi_2 },
};

test "limotorus: GK21 lnf integral vs test-local tanh-sinh at 1e-10" {
    const tc = puffy.torusConsts(0.0);
    const integrand = puffy.LnfIntegrand{
        .a = 0.0,
        .lambreak1 = tc.lambreak1,
        .lambreak2 = tc.lambreak2,
        .xi = puffy.lt.xi,
    };
    for (samples) |s| {
        const r = s[0];
        const th = s[1];
        const gd = puffy.computeGd(r, th, 0.0);
        const lam = puffy.lamBL(r * @sin(th), gd, 0.0, tc.lambreak1, tc.lambreak2, puffy.lt.xi);
        const rml = puffy.findRml(lam, 0.0, tc.lambreak1, tc.lambreak2, puffy.lt.xi);
        try std.testing.expect(rml < puffy.lt.r2);

        const gk = quad.integrate(integrand, puffy.lt.rin, rml, 0.0, 1e-12);
        try std.testing.expect(gk.converged);
        const ts = tanhSinh(integrand, puffy.lt.rin, rml);
        try std.testing.expect(@abs(gk.value - ts) <= 1e-10 * @abs(ts) + 1e-14);
    }
}

test "limotorus: bisection residuals at the 5ε target" {
    const tc = puffy.torusConsts(0.0);
    const eps = std.math.floatEps(f64);
    for (samples) |s| {
        const r = s[0];
        const th = s[1];
        const R = r * @sin(th);
        const gd = puffy.computeGd(r, th, 0.0);
        const lam = puffy.lamBL(R, gd, 0.0, tc.lambreak1, tc.lambreak2, puffy.lt.xi);

        // lamBL: |f(λ)| consistent with a root within ~5ε (f ~ λ² scale)
        const lf = puffy.LamF{
            .gdtt = gd.gdtt,
            .gdtp = gd.gdtp,
            .gdpp = gd.gdpp,
            .a = 0.0,
            .lambreak1 = tc.lambreak1,
            .lambreak2 = tc.lambreak2,
            .xi = puffy.lt.xi,
        };
        try std.testing.expect(@abs(lf.eval(lam)) <= 100.0 * eps * lam * lam);

        // findRml: midplane λ(rml) reproduces λ² at the same accuracy
        const rml = puffy.findRml(lam, 0.0, tc.lambreak1, tc.lambreak2, puffy.lt.xi);
        const gdm = puffy.computeGd(rml, pi_2, 0.0);
        const lam_m = puffy.lamBL(rml, gdm, 0.0, tc.lambreak1, tc.lambreak2, puffy.lt.xi);
        try std.testing.expect(@abs(lam * lam - lam_m * lam_m) <= 200.0 * eps * lam * lam);
    }
}

test "limotorus: ℓ(λ) broken power law + ω cross-formula + Schwarzschild lK" {
    const tc = puffy.torusConsts(0.0);
    const xi = puffy.lt.xi;

    // a = 0 Keplerian ℓ at the ISCO: ℓ = √6/(1−1/3) = 3√6/2
    try std.testing.expect(@abs(puffy.lK(6.0, 0.0) - 1.5 * @sqrt(6.0)) <= 1e-13);

    // clamping: below the first break the profile is flat at ξ·lK(λb1),
    // above the second flat at ξ·lK(λb2), in between exactly ξ·lK(λ)
    const lmid = 0.5 * (tc.lambreak1 + tc.lambreak2);
    try std.testing.expectEqual(
        xi * puffy.lK(tc.lambreak1, 0.0),
        puffy.l3d(0.5 * tc.lambreak1, 0.0, tc.lambreak1, tc.lambreak2, xi),
    );
    try std.testing.expectEqual(
        xi * puffy.lK(tc.lambreak2, 0.0),
        puffy.l3d(2.0 * tc.lambreak2, 0.0, tc.lambreak1, tc.lambreak2, xi),
    );
    try std.testing.expectEqual(
        xi * puffy.lK(lmid, 0.0),
        puffy.l3d(lmid, 0.0, tc.lambreak1, tc.lambreak2, xi),
    );

    // λ(r) monotone on the midplane and the breaks sit at LT_R1/LT_R2
    var prev: f64 = 0.0;
    var r: f64 = 25.0;
    while (r <= 340.0) : (r += 15.0) {
        const gd = puffy.computeGd(r, pi_2, 0.0);
        const lam = puffy.lamBL(r, gd, 0.0, tc.lambreak1, tc.lambreak2, xi);
        try std.testing.expect(lam > prev);
        prev = lam;
    }

    // ω from omega3d vs the independently transcribed integrand expression:
    // for a = 0, ω = (r−2)ℓ/r³ on the midplane
    r = 25.0;
    while (r <= 340.0) : (r += 15.0) {
        const gd = puffy.computeGd(r, pi_2, 0.0);
        const lam = puffy.lamBL(r, gd, 0.0, tc.lambreak1, tc.lambreak2, xi);
        const l = puffy.l3d(lam, 0.0, tc.lambreak1, tc.lambreak2, xi);
        const om1 = puffy.omega3d(l, gd);
        const om2 = (r - 2.0) * l / (r * r * r);
        try std.testing.expect(@abs(om1 - om2) <= 5e-14 * @abs(om2));
    }
}

test "limotorus: LTE pressure-split quartic residual ≤ 1e-11·P" {
    const con = puffy.consts();
    const tc = puffy.torusConsts(0.0);
    const aaa = con.four_sigmarad / 3.0;

    // points inside the torus; PUFFY's split is radiation-dominated
    // (pgas/P ≈ 1.7%), and the Mathematica closed form carries a few-ulp
    // cancellation there — measured residual ≤ 4e-12·P, gate 1e-11·P
    // (C evaluates the identical expression)
    const inside = [_][2]f64{
        .{ 50.0, pi_2 },
        .{ 100.0, 1.45 },
        .{ 200.0, 1.4 },
        .{ 300.0, pi_2 },
        .{ 450.0, pi_2 },
    };
    for (inside) |s| {
        const dv = puffy.initDsandvels(s[0], s[1], 0.0, &tc);
        try std.testing.expect(dv.rho > 0.0);
        const P = (puffy.gam - 1.0) * dv.uu;
        const bbb = con.kb_over_mugas_mp * dv.rho;
        const t = puffy.tFromPtot(P, aaa, bbb);
        try std.testing.expect(t > 0.0);
        const resid = @abs(bbb * t + aaa * (t * t * t * t) - P);
        try std.testing.expect(resid <= 1e-11 * P);
    }

    // (no synthetic gas-dominated case: there naw1 = cbrt(9ab² − √(27a²b⁴+…))
    // cancels catastrophically — the formula is only conditioned in the
    // regime PUFFY uses it in)
}

test "limotorus: torus surface — inner edge at LT_RIN, outer ρ ∝ ε³ scaling" {
    const tc = puffy.torusConsts(0.0);

    // inside the cylinder R < 35 there is no torus
    try std.testing.expect(puffy.initDsandvels(34.99, pi_2, 0.0, &tc).rho < 0.0);
    // the equatorial inner edge sits exactly at rin (h → 1 there), so the
    // density just outside it is vanishingly small
    const near = puffy.initDsandvels(35.0 + 1.0e-6, pi_2, 0.0, &tc);
    try std.testing.expect(near.rho >= 0.0);
    try std.testing.expect(near.rho < 1.0e-10);

    // locate the outer equatorial surface by bisection on the rho<0 flag
    // (the equatorial torus reaches r ≈ 480, still inside RMAX = 500)
    var lo: f64 = 350.0; // inside
    var hi: f64 = 500.0; // outside
    try std.testing.expect(puffy.initDsandvels(lo, pi_2, 0.0, &tc).rho > 0.0);
    try std.testing.expect(puffy.initDsandvels(hi, pi_2, 0.0, &tc).rho < 0.0);
    var it: usize = 0;
    while (it < 60) : (it += 1) {
        const mid = 0.5 * (lo + hi);
        if (puffy.initDsandvels(mid, pi_2, 0.0, &tc).rho > 0.0) lo = mid else hi = mid;
    }
    const rsurf = lo;

    // Γ_LT = 4/3 polytrope: ρ ∝ ε^{1/(Γ−1)} = ε³, so halving the depth
    // divides ρ by 8
    const d1 = puffy.initDsandvels(rsurf - 2.0e-3, pi_2, 0.0, &tc).rho;
    const d2 = puffy.initDsandvels(rsurf - 1.0e-3, pi_2, 0.0, &tc).rho;
    try std.testing.expect(d1 > 0.0 and d2 > 0.0);
    const ratio = d1 / d2; // expect ≈ 8 (2³)
    try std.testing.expect(ratio > 6.0 and ratio < 10.0);
}

// ---------------------------------------------------------------------------
// reduced-grid full init: β ≡ MAXBETA, div B = 0, bc.c contract

const SimP = sim_mod.Sim(config.puffy);
const LP = SimP.Layout;

fn puffyOptions() SimP.Options {
    return .{
        .coords = .mks2,
        .mp = puffy.mp,
        .gam = puffy.gam,
        .bc_x = .specific,
        .bc_y = .specific,
        .specific_bc = &puffy.Bc(SimP).calc,
    };
}

test "puffy reduced-grid init: β ≡ MAXBETA, divB = 0, bc.c contract" {
    const a = std.testing.allocator;
    var s = try SimP.init(a, puffy.makeGrid(48, 44), puffyOptions());
    defer s.deinit();

    const fac = try puffy.initAll(SimP, &s);
    std.debug.print("puffy init 48x44: fac={e:.6}\n", .{fac});
    try std.testing.expect(fac > 0.0 and std.math.isFinite(fac));

    const nx = s.nxi();
    const ny = s.nyi();

    // -- max β over the domain is exactly MAXBETA (postinit.c contract) ----
    var maxb: f64 = 0.0;
    var bmax: f64 = 0.0;
    {
        var iy: i64 = 0;
        while (iy < ny) : (iy += 1) {
            var ix: i64 = 0;
            while (ix < nx) : (ix += 1) {
                var pp: [SimP.nv]f64 = undefined;
                s.p.load(ix, iy, 0, &pp);
                const geom = s.cache.fillGeometry(ix, iy, 0);
                const ug = try relele.uconUcovFromPrims(
                    .{ pp[LP.index(.vx)], pp[LP.index(.vy)], pp[LP.index(.vz)] },
                    &geom,
                );
                const bb = mhd.bconBcovBsqFrom4vel(
                    .{ pp[LP.index(.b1)], pp[LP.index(.b2)], pp[LP.index(.b3)] },
                    ug.con,
                    ug.cov,
                    &geom,
                );
                const pmag = bb.bsq / 2.0;
                var ptot = (puffy.gam - 1.0) * pp[LP.index(.uu)];
                const rt = try radiation.calcFfRtt(config.puffy, pp, &geom);
                ptot += -rt.rtt / 3.0;
                if (pmag / ptot > maxb) maxb = pmag / ptot;

                for ([_]usize{ LP.index(.b1), LP.index(.b2), LP.index(.b3) }) |ib| {
                    bmax = @max(bmax, @abs(pp[ib]) * geom.gdet);
                }
            }
        }
    }
    try std.testing.expect(@abs(maxb - puffy.maxbeta) <= 1e-12 * puffy.maxbeta);

    // -- A-derived B is divergence-free wherever the stencil is A-built ----
    // (cells at ix,iy ≥ 1 only touch domain corners of the vector potential)
    var worst_div: f64 = 0.0;
    {
        var iy: i64 = 1;
        while (iy < ny) : (iy += 1) {
            var ix: i64 = 1;
            while (ix < nx) : (ix += 1) {
                worst_div = @max(worst_div, @abs(ct.calcDivB(SimP, &s, ix, iy, 0)));
            }
        }
    }
    // normalize by the largest gdet·B over the smallest cell size
    const dxmin = @min(s.grid.cellSize(0, 0), s.grid.cellSize(0, 1));
    std.debug.print("puffy init: max|divB|·dx/max|gdetB| = {e:.3}\n", .{worst_div * dxmin / bmax});
    try std.testing.expect(worst_div * dxmin / bmax <= 1e-12);

    // -- bc.c contract -------------------------------------------------------
    // polar mirror maps: p(ix,−k) = flip(p(ix,k−1)), p(ix,ny−1+k) = flip(p(ix,ny−k))
    {
        var ix: i64 = 3;
        while (ix < nx) : (ix += 17) {
            var k: i64 = 1;
            while (k <= 3) : (k += 1) {
                var pg: [SimP.nv]f64 = undefined;
                var ps: [SimP.nv]f64 = undefined;
                s.p.load(ix, -k, 0, &pg);
                s.p.load(ix, k - 1, 0, &ps);
                for (0..SimP.nv) |iv| {
                    const expct = if (iv == LP.index(.vy) or iv == LP.index(.b2) or iv == LP.index(.fy)) -ps[iv] else ps[iv];
                    try std.testing.expectEqual(expct, pg[iv]);
                }
                s.p.load(ix, ny - 1 + k, 0, &pg);
                s.p.load(ix, ny - k, 0, &ps);
                for (0..SimP.nv) |iv| {
                    const expct = if (iv == LP.index(.vy) or iv == LP.index(.b2) or iv == LP.index(.fy)) -ps[iv] else ps[iv];
                    try std.testing.expectEqual(expct, pg[iv]);
                }
            }
        }
    }
    // XBCLO: plain copy of cell 0 (RMIN < r_horizon skips the inflow check)
    {
        var iy: i64 = 2;
        while (iy < ny) : (iy += 13) {
            var pg: [SimP.nv]f64 = undefined;
            var ps: [SimP.nv]f64 = undefined;
            s.p.load(-2, iy, 0, &pg);
            s.p.load(0, iy, 0, &ps);
            for (0..SimP.nv) |iv| try std.testing.expectEqual(ps[iv], pg[iv]);
        }
    }
    // XBCHI: ρ rescaled by (r_bound/r_ghost)² in BL, and no BL inflow in
    // either the gas or the radiation frame velocity. The BC zeroes u^r in
    // BL and transforms back, so re-deriving u^r_BL from the stored prims
    // carries roundtrip noise (measured −5e-19) — gate at −1e-15.
    {
        var iy: i64 = 2;
        while (iy < ny) : (iy += 13) {
            var k: i64 = 1;
            while (k <= 3) : (k += 1) {
                const ix = nx - 1 + k;
                var pg: [SimP.nv]f64 = undefined;
                var ps: [SimP.nv]f64 = undefined;
                s.p.load(ix, iy, 0, &pg);
                s.p.load(nx - 1, iy, 0, &ps);

                const gbl = puffy.fillGeometryBL(&s.grid, .mks2, ix, iy, 0);
                const bbl = puffy.fillGeometryBL(&s.grid, .mks2, nx - 1, iy, 0);
                const scale1 = bbl.xxvec[1] * bbl.xxvec[1] / gbl.xxvec[1] / gbl.xxvec[1];
                try std.testing.expect(
                    @abs(pg[LP.index(.rho)] - ps[LP.index(.rho)] * scale1) <=
                        1e-14 * ps[LP.index(.rho)] * scale1,
                );

                const geom = s.cache.fillGeometry(ix, iy, 0);
                var ucon = [4]f64{ 0, pg[LP.index(.vx)], pg[LP.index(.vy)], pg[LP.index(.vz)] };
                ucon = try relele.convert(ucon, .velr, .vel4, &geom, .recompute_ut);
                ucon = frames.trans2Coco(geom.xxvec, ucon, .mks2, .bl, puffy.mp);
                try std.testing.expect(ucon[1] >= -1e-15);

                var urf = [4]f64{ 0, pg[LP.index(.fx)], pg[LP.index(.fy)], pg[LP.index(.fz)] };
                urf = try relele.convert(urf, .velr, .vel4, &geom, .recompute_ut);
                urf = frames.trans2Coco(geom.xxvec, urf, .mks2, .bl, puffy.mp);
                try std.testing.expect(urf[1] >= -1e-15);
            }
        }
    }
}

// M14: the same init contract in 3D. This is the one piece the step golden
// cannot cover — it loads C's post-init B, so it never exercises the *init*
// calc_BfromA curl. Here the full limotorus + QUADLOOPS A_φ + 3D curl +
// β-normalization runs on a real 32×30×4 wedge (periodic φ), and we assert the
// A-derived field is divergence-free over the whole 3D domain and β-norm holds
// globally. Slow-gated (a limotorus solve per cell, ×nz). B³≡0 at t=0 (poloidal
// A_φ only), so this validates the 3D curl produces the same poloidal field at
// every φ-slice with zero ∇·B.
test "M14 puffy 3D reduced-grid init: β ≡ MAXBETA, divB = 0 (32×30×4)" {
    if (!build_options.slow_tests) return error.SkipZigTest;
    const a = std.testing.allocator;
    var s = try SimP.init(a, puffy.makeGridNz(32, 30, 4), puffyOptions());
    defer s.deinit();

    const fac = try puffy.initAll(SimP, &s);
    try std.testing.expect(fac > 0.0 and std.math.isFinite(fac));

    const nx = s.nxi();
    const ny = s.nyi();
    const nz = s.nzi();

    // -- max β over the full 3D domain is exactly MAXBETA (postinit.c) --------
    var maxb: f64 = 0.0;
    var bmax: f64 = 0.0;
    var b3max: f64 = 0.0;
    var b12max: f64 = 0.0;
    {
        var iz: i64 = 0;
        while (iz < nz) : (iz += 1) {
            var iy: i64 = 0;
            while (iy < ny) : (iy += 1) {
                var ix: i64 = 0;
                while (ix < nx) : (ix += 1) {
                    var pp: [SimP.nv]f64 = undefined;
                    s.p.load(ix, iy, iz, &pp);
                    const geom = s.cache.fillGeometry(ix, iy, iz);
                    const ug = try relele.uconUcovFromPrims(
                        .{ pp[LP.index(.vx)], pp[LP.index(.vy)], pp[LP.index(.vz)] },
                        &geom,
                    );
                    const bb = mhd.bconBcovBsqFrom4vel(
                        .{ pp[LP.index(.b1)], pp[LP.index(.b2)], pp[LP.index(.b3)] },
                        ug.con,
                        ug.cov,
                        &geom,
                    );
                    const pmag = bb.bsq / 2.0;
                    var ptot = (puffy.gam - 1.0) * pp[LP.index(.uu)];
                    const rt = try radiation.calcFfRtt(config.puffy, pp, &geom);
                    ptot += -rt.rtt / 3.0;
                    if (pmag / ptot > maxb) maxb = pmag / ptot;
                    b3max = @max(b3max, @abs(pp[LP.index(.b3)]));
                    b12max = @max(b12max, @max(@abs(pp[LP.index(.b1)]), @abs(pp[LP.index(.b2)])));
                    for ([_]usize{ LP.index(.b1), LP.index(.b2), LP.index(.b3) }) |ib| {
                        bmax = @max(bmax, @abs(pp[ib]) * geom.gdet);
                    }
                }
            }
        }
    }
    try std.testing.expect(@abs(maxb - puffy.maxbeta) <= 1e-12 * puffy.maxbeta);
    // toroidal field is zero to machine precision relative to the poloidal
    // field (3D curl of a purely poloidal A_φ — B³ is denormal-level noise,
    // ~1e-40 against a ~1e-11 poloidal scale, not the exact 0 of the 2D branch)
    try std.testing.expect(b3max <= 1e-18 * b12max);

    // -- A-derived B is divergence-free over the whole 3D interior -----------
    // (ix,iy ≥ 1 touch only domain corners of A; z wraps periodically, so the
    //  curl identity holds at every iz)
    var worst_div: f64 = 0.0;
    {
        var iz: i64 = 0;
        while (iz < nz) : (iz += 1) {
            var iy: i64 = 1;
            while (iy < ny) : (iy += 1) {
                var ix: i64 = 1;
                while (ix < nx) : (ix += 1) {
                    worst_div = @max(worst_div, @abs(ct.calcDivB(SimP, &s, ix, iy, iz)));
                }
            }
        }
    }
    const dxmin = @min(s.grid.cellSize(0, 0), @min(s.grid.cellSize(0, 1), s.grid.cellSize(0, 2)));
    std.debug.print("puffy 3D init 32×30×4: fac={e:.6}  max|divB|·dx/max|gdetB| = {e:.3}\n", .{ fac, worst_div * dxmin / bmax });
    try std.testing.expect(worst_div * dxmin / bmax <= 1e-12);
}

// ---------------------------------------------------------------------------
// hydro-only limotorus: stationarity under evolution
//
// With Γ = Γ_LT = 4/3 the polytrope p = (Γ−1)u = κρ^Γ_LT is exactly the
// relation the limotorus solution was built from, so the torus is a
// stationary solution of the pure-hydro equations. prepInitCell degrades to
// exactly that setup for a radiation-less, B-less config (no pressure
// split, no vector potential). This exercises correct_polaraxis on a real
// spherical grid — the M10 transcription ran only on a synthetic wedge.

const cfg_hydro = config.Config{
    .modules = &.{.hydro},
    .reconstruction = .ppm,
    .flux = .laxf,
    .timestepping = .rk2imex,
    .coords = .mks2,
};
const SimH = sim_mod.Sim(cfg_hydro);
const LH = SimH.Layout;

fn hydroTorusInit(s: *SimH) !void {
    const con = puffy.consts();
    const atm = puffy.atmConsts(&con);
    const tc = puffy.torusConsts(puffy.mp.a);
    var iy: i64 = 0;
    while (iy < s.nyi()) : (iy += 1) {
        var ix: i64 = 0;
        while (ix < s.nxi()) : (ix += 1) {
            const geom = s.cache.fillGeometry(ix, iy, 0);
            const geomBL = puffy.fillGeometryBL(&s.grid, .mks2, ix, iy, 0);
            var pp = try puffy.prepInitCell(cfg_hydro, &geom, &geomBL, &tc, &con, &atm);
            pp[LH.index(.entr)] = @import("physics/hydro.zig").sFromU(
                pp[LH.index(.rho)],
                pp[LH.index(.uu)],
                s.opt.gam,
            );
            const uu = try @import("p2u.zig").p2u(cfg_hydro, pp, &geom, s.opt.gam);
            s.p.store(ix, iy, 0, &pp);
            s.u.store(ix, iy, 0, &uu);
        }
    }
    try s.finishInit(); // ghost fill + the C initial-dt guess (problem.c:59)
}

/// L1 drift of ρ vs the initial state over the torus interior
/// (gdet-weighted; mask ρ₀ > 10⁻²¹ ≫ atmosphere ~10⁻²⁴).
fn torusDrift(s: *SimH, p0: []const f64) f64 {
    var num: f64 = 0.0;
    var den: f64 = 0.0;
    var iy: i64 = 0;
    while (iy < s.nyi()) : (iy += 1) {
        var ix: i64 = 0;
        while (ix < s.nxi()) : (ix += 1) {
            const idx: usize = @intCast((iy * s.nxi() + ix) * SimH.nv + @as(i64, @intCast(LH.index(.rho))));
            const rho0 = p0[idx];
            if (rho0 <= 1e-21) continue;
            var pp: [SimH.nv]f64 = undefined;
            s.p.load(ix, iy, 0, &pp);
            const geom = s.cache.fillGeometry(ix, iy, 0);
            num += @abs(pp[LH.index(.rho)] - rho0) * geom.gdet;
            den += rho0 * geom.gdet;
        }
    }
    return num / den;
}

fn runHydroTorus(a: std.mem.Allocator, n: usize, t_end: f64) !f64 {
    var s = try SimH.init(a, puffy.makeGrid(n, n), .{
        .coords = .mks2,
        .mp = puffy.mp,
        .gam = puffy.lt.gamma, // 4/3: the polytrope the torus was built with
        .bc_x = .specific,
        .bc_y = .specific,
        .specific_bc = &puffy.Bc(SimH).calc,
        .correct_polaraxis = true,
        .nccorrectpolar = 2,
    });
    defer s.deinit();
    try hydroTorusInit(&s);

    // snapshot the initial domain primitives
    const nc: usize = @intCast(s.nxi() * s.nyi());
    const p0 = try a.alloc(f64, nc * SimH.nv);
    defer a.free(p0);
    {
        var iy: i64 = 0;
        while (iy < s.nyi()) : (iy += 1) {
            var ix: i64 = 0;
            while (ix < s.nxi()) : (ix += 1) {
                var pp: [SimH.nv]f64 = undefined;
                s.p.load(ix, iy, 0, &pp);
                const idx: usize = @intCast((iy * s.nxi() + ix) * SimH.nv);
                @memcpy(p0[idx .. idx + SimH.nv], &pp);
            }
        }
    }

    var steps: usize = 0;
    while (s.t < t_end and steps < 20000) : (steps += 1) try s.step(null);
    const drift = torusDrift(&s, p0);
    std.debug.print("hydro torus {d}x{d}: t={d:.2} steps={d} L1 rho drift={e:.4}\n", .{ n, n, s.t, steps, drift });
    return drift;
}

test "puffy hydro limotorus: stationary, drift converges with resolution" {
    const a = std.testing.allocator;
    const d40 = try runHydroTorus(a, 40, 10.0);
    const d80 = try runHydroTorus(a, 80, 10.0);
    const order = std.math.log2(d40 / d80);
    std.debug.print("hydro torus drift order: {d:.2}\n", .{order});
    // measured: 8.1e-5 @ 40², 1.85e-5 @ 80² → order 2.13
    try std.testing.expect(d40 < 1e-3);
    try std.testing.expect(order > 1.5);
}

// ---------------------------------------------------------------------------
// init perturbation + seed MRI-quality report (campaign notes 2026-08-08)

test "perturbXi: pure, bounded, varies across cells" {
    const x = [4]f64{ 0, 1.234, 0.456, -0.789 };
    const a1 = puffy.perturbXi(x);
    const a2 = puffy.perturbXi(x);
    try std.testing.expectEqual(a1, a2); // pure
    try std.testing.expect(a1 >= -1.0 and a1 < 1.0);
    var y = x;
    y[2] = 0.4560001;
    try std.testing.expect(puffy.perturbXi(y) != a1); // decorrelated cells
}

test "puffy init perturbation: deterministic, bounded, off-by-default identical" {
    const a = std.testing.allocator;

    // baseline: the validated perturb = 0 path
    var s0 = try SimP.init(a, puffy.makeGrid(24, 20), puffyOptions());
    defer s0.deinit();
    _ = try puffy.initAll(SimP, &s0);

    puffy.perturb = 0.05;
    defer puffy.perturb = 0.0;
    var s1 = try SimP.init(a, puffy.makeGrid(24, 20), puffyOptions());
    defer s1.deinit();
    _ = try puffy.initAll(SimP, &s1);
    var s2 = try SimP.init(a, puffy.makeGrid(24, 20), puffyOptions());
    defer s2.deinit();
    _ = try puffy.initAll(SimP, &s2);

    var n_diff: usize = 0;
    var iy: i64 = 0;
    while (iy < s0.nyi()) : (iy += 1) {
        var ix: i64 = 0;
        while (ix < s0.nxi()) : (ix += 1) {
            var p0: [SimP.nv]f64 = undefined;
            var p1: [SimP.nv]f64 = undefined;
            var p2: [SimP.nv]f64 = undefined;
            s0.p.load(ix, iy, 0, &p0);
            s1.p.load(ix, iy, 0, &p1);
            s2.p.load(ix, iy, 0, &p2);
            // determinism: two perturbed inits are bit-identical
            try std.testing.expectEqual(p1[LP.index(.uu)], p2[LP.index(.uu)]);
            const uu_base = p0[LP.index(.uu)];
            const uu_pert = p1[LP.index(.uu)];
            if (uu_pert != uu_base) {
                n_diff += 1;
                // bounded near the 5% amplitude (the gas/rad pressure split
                // repartitions it, hence the slack)
                try std.testing.expect(@abs(uu_pert - uu_base) / uu_base < 0.10);
            }
        }
    }
    // the torus interior actually got noise
    try std.testing.expect(n_diff > 50);
}

test "puffy seed quality: positive on the initialized torus and linear in B" {
    const a = std.testing.allocator;
    var s = try SimP.init(a, puffy.makeGrid(24, 20), puffyOptions());
    defer s.deinit();
    _ = try puffy.initAll(SimP, &s);

    const q1 = try puffy.seedQuality(SimP, &s);
    try std.testing.expect(q1.mass > 0);
    const qth1 = q1.qth_m / q1.mass;
    const qr1 = q1.qr_m / q1.mass;
    try std.testing.expect(qth1 > 0 and std.math.isFinite(qth1));
    try std.testing.expect(qr1 > 0 and std.math.isFinite(qr1));

    // Q is linear in the field: scaling B ×2 doubles the mass-weighted mean
    // (the b² inertia correction is negligible at seed β)
    var iy: i64 = 0;
    while (iy < s.nyi()) : (iy += 1) {
        var ix: i64 = 0;
        while (ix < s.nxi()) : (ix += 1) {
            var pp: [SimP.nv]f64 = undefined;
            s.p.load(ix, iy, 0, &pp);
            pp[LP.index(.b1)] *= 2.0;
            pp[LP.index(.b2)] *= 2.0;
            pp[LP.index(.b3)] *= 2.0;
            try s.initCell(ix, iy, 0, pp);
        }
    }
    const q2 = try puffy.seedQuality(SimP, &s);
    try std.testing.expectEqual(q1.mass, q2.mass); // mask untouched by B
    try std.testing.expectApproxEqRel(2.0 * qth1, q2.qth_m / q2.mass, 1e-3);
}
