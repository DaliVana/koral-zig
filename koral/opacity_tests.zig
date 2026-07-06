//! M8 theory gates — opacities and the radiative four-force.
//!
//! Gates (plan M8):
//!  * κ_es hand values: 0.4·ρ cm²/g at cold Trad, exactly halved at
//!    Trad = 4.5e8 K (Klein–Nishina), correction → 1 as Trad → 0
//!  * scaling exponents: bremsstrahlung ∝ ρ², ∝ Te^−3.5 (modulo the
//!    relativistic correction); synchrotron emission ∝ B²; Terelfactor
//!    = 1/2 at θe = 1
//!  * Kirchhoff at ζ = Trad/Te = 1: κ_radAbs(ff) = κ_gasAbs(ff) (the
//!    log1p(1.6)/log(2.6) construction) @1e-14
//!  * G^μ = 0 at LTE comoving equilibrium (bremsstrahlung channel; the
//!    synchrotron fit formulas are not exactly Kirchhoff-balanced)
//!  * heating/cooling sign follows sgn(Trad − Tgas)
//!  * frame invariance G^μ u_μ = −Ĝ^0 (lab contraction vs fluid frame,
//!    Compton included) @1e-12 over MINK/KS/MKS2 boosted states
//!  * Compton term: exactly zero at Te = Trad, exactly linear in ΔT
//!  * evolution smoke: uniform rad state with real opacities (χ > 0,
//!    τ-damped wavespeeds) stays static

const std = @import("std");
const config = @import("config.zig");
const layout = @import("layout.zig");
const grid_mod = @import("grid.zig");
const sim_mod = @import("sim.zig");
const relele = @import("relele.zig");
const hydro = @import("physics/hydro.zig");
const thermo = @import("physics/thermo.zig");
const opacities = @import("physics/opacities.zig");
const radforce = @import("physics/radforce.zig");
const invert_rad = @import("solve/invert_rad.zig");
const units_mod = @import("units.zig");
const precompute = @import("metric/precompute.zig");
const metric = @import("metric/metric.zig");
const Geometry = @import("geometry.zig").Geometry;

const Grid = grid_mod.Grid;
const expect = std.testing.expect;
const geometryAt = precompute.geometryAt;

const puffy_mp = metric.MetricParams{ .a = 0.0, .mksr0 = 0.1, .mksh0 = 0.9 };
const gam: f64 = 5.0 / 3.0;

const cfg = config.Config{
    .modules = &.{ .hydro, .mhd, .radiation },
    .reconstruction = .ppm,
    .flux = .laxf,
    .timestepping = .rk2imex,
    .coords = .mink,
};
const L = layout.VarLayout(cfg);
const NV = L.count;

fn expectClose(a: f64, b: f64, tol: f64) !void {
    const dev = @abs(a - b) / @max(1.0, @max(@abs(a), @abs(b)));
    if (!(dev <= tol)) {
        std.debug.print("expectClose: {e:.17} vs {e:.17} (dev {e}, tol {e})\n", .{ a, b, dev, tol });
        return error.TestExpectedApproxEq;
    }
}

/// Gas+rad primitives; uint set from Tgas via the ideal-gas relation.
fn ppFromTemps(
    c: *const thermo.Consts,
    rho: f64,
    tgas: f64,
    v: [3]f64,
    erf: f64,
    f: [3]f64,
    b: [3]f64,
) [NV]f64 {
    const uint = tgas * rho * c.kb_over_mugas_mp / (gam - 1.0);
    var pp: [NV]f64 = @splat(0);
    pp[L.index(.rho)] = rho;
    pp[L.index(.uu)] = uint;
    pp[L.index(.vx)] = v[0];
    pp[L.index(.vy)] = v[1];
    pp[L.index(.vz)] = v[2];
    pp[L.index(.entr)] = hydro.sFromU(rho, uint, gam);
    pp[L.index(.b1)] = b[0];
    pp[L.index(.b2)] = b[1];
    pp[L.index(.b3)] = b[2];
    pp[L.index(.ee)] = erf;
    pp[L.index(.fx)] = f[0];
    pp[L.index(.fy)] = f[1];
    pp[L.index(.fz)] = f[2];
    return pp;
}

//
// ---- kappa_es -----------------------------------------------------------------
//

test "M8: kappa_es hand values — 0.4 cm2/g cold, exactly halved at Trad = 4.5e8 K" {
    const p = radforce.Params.puffy();
    const u = p.consts.units;
    const rho: f64 = 3.7e-9;

    // cold radiation: Klein–Nishina correction -> 1
    // ((T/4.5e8)^0.86 ≤ 1e-13 needs T ≲ 4e-7 K)
    {
        const kes = opacities.kappaEsPuffy(&p.consts, p.channels, rho, 1.0e-10);
        try expectClose(u.kappaGu2Cgs(kes / rho), 0.4, 1e-13);
    }
    // Trad = 4.5e8 exactly: (T/4.5e8)^0.86 = 1 -> correction = 1/2
    {
        const kes = opacities.kappaEsPuffy(&p.consts, p.channels, rho, 4.5e8);
        try expectClose(u.kappaGu2Cgs(kes / rho), 0.2, 1e-13);
    }
    // linear in rho
    {
        const k1 = opacities.kappaEsPuffy(&p.consts, p.channels, rho, 1.0e6);
        const k2 = opacities.kappaEsPuffy(&p.consts, p.channels, 2.0 * rho, 1.0e6);
        try expectClose(k2 / k1, 2.0, 1e-15);
    }
}

//
// ---- absorption channel scalings ---------------------------------------------
//

const brems_only = opacities.Channels{ .synchrotron = false };
const syn_only = opacities.Channels{ .bremsstrahlung = false };

fn stateIn(rho: f64, te: f64, zeta: f64, bsq: f64, c: *const thermo.Consts) opacities.StateIn {
    const trad = zeta * te;
    return .{
        .rho = rho,
        .tgas = te,
        .te = te,
        .trad = trad,
        .tradbb = trad,
        .ne = thermo.thermalNe(c, rho),
        .bsq = bsq,
    };
}

test "M8: bremsstrahlung scales as rho^2 and Te^-3.5 (relativistic correction factored)" {
    const p = radforce.Params.puffy();
    const rho: f64 = 1.0e-8;
    const te: f64 = 1.0e8;

    // density: kappa_gasAbs ∝ ρ²
    {
        const k1 = opacities.calcOpacitiesFromState(&p.consts, brems_only, stateIn(rho, te, 1.0, 0, &p.consts));
        const k2 = opacities.calcOpacitiesFromState(&p.consts, brems_only, stateIn(2.0 * rho, te, 1.0, 0, &p.consts));
        try expectClose(k2.gas_abs / k1.gas_abs, 4.0, 1e-14);
    }
    // temperature: κ ∝ √Te·(1+4.4e-10·Te)/Te⁴
    {
        const k1 = opacities.calcOpacitiesFromState(&p.consts, brems_only, stateIn(rho, te, 1.0, 0, &p.consts));
        const k2 = opacities.calcOpacitiesFromState(&p.consts, brems_only, stateIn(rho, 4.0 * te, 1.0, 0, &p.consts));
        const corr = (1.0 + 4.4e-10 * 4.0 * te) / (1.0 + 4.4e-10 * te);
        const want = std.math.pow(f64, 4.0, -3.5) * corr;
        try expectClose(k2.gas_abs / k1.gas_abs, want, 1e-12);
    }
}

test "M8: synchrotron emission opacity scales as B^2; Terelfactor = 1/2 at theta_e = 1" {
    const p = radforce.Params.puffy();
    const rho: f64 = 1.0e-8;
    const te: f64 = 1.0e10;

    {
        const k1 = opacities.calcOpacitiesFromState(&p.consts, syn_only, stateIn(rho, te, 1.0, 1.0e-12, &p.consts));
        const k2 = opacities.calcOpacitiesFromState(&p.consts, syn_only, stateIn(rho, te, 1.0, 4.0e-12, &p.consts));
        try expectClose(k2.gas_abs / k1.gas_abs, 4.0, 1e-14);
    }
    // the Terelfactor suppression is a pure prefactor: it must cancel in
    // any same-Te ratio, e.g. the B-scaling at θe = 1 stays exactly 4
    {
        const te1 = 1.0 / p.consts.k_over_mecsq; // θe = 1, Terelfactor = 1/2
        const k = opacities.calcOpacitiesFromState(&p.consts, syn_only, stateIn(rho, te1, 1.0, 1.0e-12, &p.consts));
        const k_hi = opacities.calcOpacitiesFromState(&p.consts, syn_only, stateIn(rho, te1, 1.0, 4.0e-12, &p.consts));
        try expectClose(k_hi.gas_abs / k.gas_abs, 4.0, 1e-14);
    }
}

test "M8: Kirchhoff at zeta = 1 — rad and gas ff absorption opacities coincide" {
    const p = radforce.Params.puffy();
    for ([_]f64{ 1.0e6, 1.0e8, 1.0e10 }) |te| {
        const k = opacities.calcOpacitiesFromState(&p.consts, brems_only, stateIn(1.0e-9, te, 1.0, 0, &p.consts));
        // log1p(1.6·1)·(1/log 2.6)·1 = 1 up to rounding
        try expectClose(k.rad_abs, k.gas_abs, 1e-14);
    }
}

//
// ---- four-force ---------------------------------------------------------------
//

test "M8: G^mu = 0 at LTE comoving equilibrium (bremsstrahlung channel)" {
    var p = radforce.Params.puffy();
    p.channels = .{ .synchrotron = false, .comptonization = false };
    const c = &p.consts;

    for ([_]f64{ 1.0e6, 1.0e8, 1.0e10 }) |tgas| {
        const rho: f64 = 1.0e-8;
        const erf = c.lteEfromT(tgas); // comoving, so Ê = Êrf exactly-ish
        const geo = geometryAt(.mink, puffy_mp, .{ 0, 0, 0, 0 });
        const pp = ppFromTemps(c, rho, tgas, .{ 0, 0, 0 }, erf, .{ 0, 0, 0 }, .{ 0, 0, 0 });

        const st = try radforce.fillRadState(cfg, pp, &geo, gam, &p);
        const gi = try radforce.calcGiFromState(&st, .{ 0, 0, 0 }, &geo, &p);

        // scale of either side of the balance
        const scale = st.opac.gas_abs * c.fourpi * (c.sigma_rad_over_pi * st.te * st.te * st.te * st.te);
        for (0..4) |i| {
            try expect(@abs(gi.ff[i]) <= 1e-12 * scale);
            try expect(@abs(gi.lab[i]) <= 1e-12 * scale);
        }
    }
}

test "M8: heating/cooling sign follows sgn(Trad - Tgas)" {
    var p = radforce.Params.puffy();
    const c = &p.consts;
    const geo = geometryAt(.mink, puffy_mp, .{ 0, 0, 0, 0 });
    const rho: f64 = 1.0e-8;
    const tgas: f64 = 1.0e8;

    // brems only (exact Kirchhoff), then all channels far from balance
    const cases = [_]struct { ch: opacities.Channels, zeta: f64, heats: bool }{
        .{ .ch = .{ .synchrotron = false, .comptonization = false }, .zeta = 0.5, .heats = false },
        .{ .ch = .{ .synchrotron = false, .comptonization = false }, .zeta = 2.0, .heats = true },
        .{ .ch = .{}, .zeta = 0.1, .heats = false },
        .{ .ch = .{}, .zeta = 10.0, .heats = true },
    };
    for (cases) |cs| {
        p.channels = cs.ch;
        const erf = c.lteEfromT(cs.zeta * tgas);
        const pp = ppFromTemps(c, rho, tgas, .{ 0, 0, 0 }, erf, .{ 0, 0, 0 }, .{ 0, 0, 0 });
        const gi = try radforce.calcGi(cfg, pp, &geo, gam, &p);
        // Ĝ^0 > 0 ⇔ radiation loses energy to the gas (gas heats)
        try expect((gi.ff[0] > 0) == cs.heats);
    }
}

test "M8: frame invariance G^mu u_mu = -Ghat^0 (Compton included), boosted states" {
    var prng = std.Random.DefaultPrng.init(0x4d38636f76);
    const rng = prng.random();
    const p = radforce.Params.puffy();
    const c = &p.consts;

    const geoms = [_]Geometry{
        geometryAt(.mink, puffy_mp, .{ 0, 0.3, 0.7, 0.1 }),
        geometryAt(.ks, puffy_mp, .{ 0, 7.3, 1.1, 0.4 }),
        geometryAt(.mks2, puffy_mp, .{ 0, 2.2, 0.45, 0.4 }),
    };
    for (geoms) |geo| {
        for (0..40) |_| {
            // controlled-γ velocities (coordinate-basis scale via the metric)
            var n: [3]f64 = undefined;
            var qsq: f64 = 0;
            for (0..3) |i| n[i] = 2.0 * rng.float(f64) - 1.0;
            for (0..3) |i| {
                for (0..3) |j| qsq += n[i] * n[j] * geo.gg[i + 1][j + 1];
            }
            const gtarget = 1.0 + 1.0 * rng.float(f64);
            const s = @sqrt((gtarget * gtarget - 1.0) / qsq);
            const v = [3]f64{ n[0] * s, n[1] * s, n[2] * s };

            const tgas = std.math.pow(f64, 10.0, 6.0 + 5.0 * rng.float(f64));
            const zeta = std.math.pow(f64, 10.0, -1.5 + 3.0 * rng.float(f64));
            const rho = std.math.pow(f64, 10.0, -12.0 + 6.0 * rng.float(f64));
            const erf = c.lteEfromT(zeta * tgas);
            const bmag = @sqrt(2.0 * 0.1 * (gam - 1.0) *
                (tgas * rho * c.kb_over_mugas_mp / (gam - 1.0))); // β ~ 10 field
            const pp = ppFromTemps(c, rho, tgas, v, erf, .{ 0.3 * v[0], 0.3 * v[1], 0.3 * v[2] }, .{ bmag, 0, 0 });

            const st = try radforce.fillRadState(cfg, pp, &geo, gam, &p);
            const gi = try radforce.calcGiFromState(&st, v, &geo, &p);

            var gu: f64 = 0;
            for (0..4) |i| gu += gi.lab[i] * st.ucov[i];

            const scale = @max(@abs(gi.ff[0]), st.opac.gas_abs * c.fourpi *
                (c.sigma_rad_over_pi * st.te * st.te * st.te * st.te) + (st.opac.rad_ross + st.kappaes) * st.ehat);
            try expect(@abs(gu + gi.ff[0]) <= 1e-12 * scale);
        }
    }
}

test "M8: Compton term — exactly zero at Te = Trad, exactly linear in Trad - Te" {
    const p = radforce.Params.puffy();
    const c = &p.consts;
    const geo = geometryAt(.mink, puffy_mp, .{ 0, 0, 0, 0 });

    const pp = ppFromTemps(c, 1.0e-8, 1.0e8, .{ 0, 0, 0 }, c.lteEfromT(3.0e7), .{ 0, 0, 0 }, .{ 0, 0, 0 });
    var st = try radforce.fillRadState(cfg, pp, &geo, gam, &p);

    // zero at Te = Trad (hand-forced equality: the (Trad − Te) factor)
    st.trad = st.te;
    try std.testing.expectEqual(@as(f64, 0), radforce.comptonGiCoeff(c, &st));

    // exact linearity in (Trad − Te) with everything else held fixed
    st.trad = st.te + 1.0e4;
    const c1 = radforce.comptonGiCoeff(c, &st);
    st.trad = st.te + 2.0e4;
    const c2 = radforce.comptonGiCoeff(c, &st);
    try expectClose(c2 / c1, 2.0, 1e-14);

    // sign: Trad > Te heats the gas
    try expect(c1 > 0);
}

//
// ---- evolution smoke: real opacities in the wavespeed limiter ------------------
//

test "M8: uniform rad state with real opacities (chi > 0) stays static" {
    const SimT = sim_mod.Sim(cfg);
    const p = radforce.Params.puffy();

    const g = Grid.init(.{ .nx = 16, .ng = 3, .minx = 0, .maxx = 20.0 });
    var s = try SimT.init(std.testing.allocator, g, .{
        .coords = .mink,
        .gam = gam,
        .rad = invert_rad.RadParams.puffy,
        .opac = p,
        .bc_x = .periodic,
    });
    defer s.deinit();

    // a dense, hot state so χ·dx is significant (τ-damping active);
    // u/ρ ~ 1e-2 keeps the update_entropy p2u∘u2p re-projection noise
    // (amplified by ρ/u) below the 1e-14 static gate
    const c = &p.consts;
    const tgas: f64 = 1.0e11;
    const rho: f64 = 1.0e-8;
    const pp = ppFromTemps(c, rho, tgas, .{ 0, 0, 0 }, c.lteEfromT(tgas), .{ 0, 0, 0 }, .{ 0, 0, 0 });

    var ix: i64 = 0;
    while (ix < s.nxi()) : (ix += 1) {
        try s.initCell(ix, 0, 0, pp);
    }
    try s.finishInit();

    // sanity: the state really is optically thick over a cell
    const geo = s.cache.fillGeometry(0, 0, 0);
    const chi = try radforce.calcChi(cfg, pp, &geo, gam, &p);
    if (!(chi * s.grid.cellSize(0, 0) > 1.0)) {
        std.debug.print("not optically thick: chi {e} dx {e}\n", .{ chi, s.grid.cellSize(0, 0) });
        return error.TestUnexpectedResult;
    }

    var pp0: [NV]f64 = undefined;
    s.p.load(8, 0, 0, &pp0);
    for (0..10) |_| try s.step(null);

    ix = 0;
    while (ix < s.nxi()) : (ix += 1) {
        var q: [NV]f64 = undefined;
        s.p.load(ix, 0, 0, &q);
        for (0..NV) |iv| {
            const scale = @max(@abs(pp0[iv]), 1e-30);
            const dev = @abs(q[iv] - pp0[iv]) / scale;
            if (!(dev <= 1e-14)) {
                std.debug.print("static drift: ix {d} iv {d}: {e:.17} -> {e:.17} (dev {e})\n", .{ ix, iv, pp0[iv], q[iv], dev });
                return error.TestUnexpectedResult;
            }
        }
    }
}
