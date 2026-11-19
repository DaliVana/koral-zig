//! Reproduction gate for Sadowski et al. (2013), section 4.6 and Figure 11:
//! a static, optically thin, polytropic atmosphere held up against gravity
//! by its pressure gradient and by the radiation force of a flux F_in
//! injected at its base. The momentum exchange between radiation and gas is
//! the only thing being tested: kappa_abs = 0, so there is no heating, and
//! the radial momentum equation in the Newtonian limit reads
//!
//!     (1 / rho) dp/dr = -(1 - f) / r^2,     f = kappa_es F_in r_in^2,
//!
//! f being the luminosity in Eddington units. With p = K rho^Gamma,
//!
//!     rho(r) = [ (Gamma - 1) / (Gamma K) (C + (1 - f) / r) ]^(1 / (Gamma - 1)),
//!     C = Gamma K rho_in^(Gamma - 1) / (Gamma - 1) - (1 - f) / r_in,
//!
//! and the flux is F = F_in r_in^2 / r^2. At f = 1 gravity is balanced by
//! radiation alone and the density is uniform.
//!
//! Setup as in the paper: M = 1 Msun, kappa_es = 0.4 cm^2/g, 40 cells
//! uniform in r between 1e6 and 1.4e6 r_g, rho_in = 1e-15 g/cm^3, T_in =
//! 1e6 K, E = F / 0.99 at the base, four models f = 1e-10, 0.1, 0.5, 1, run to
//! t = 2e9 M (two sound crossings) starting from the analytic solution.
//!
//! Two parameters the text does not give were recovered by fitting the
//! analytic profile to the paper's own density curves, digitised from the
//! EPS source of Figure 11 (test82.eps, 39 points per model): Gamma = 1.4
//! and mu = 1 reproduce all three non-trivial curves to an rms of 0.0025 of
//! rho_in, and no other pair on the grid Gamma in {4/3, 1.4, 1.5, 5/3, 2},
//! mu in {0.5, 0.6, 1, 1.2, 2} comes close (with Gamma = 5/3, mu = 1 the
//! f = 1e-10 atmosphere would end at r = 1.3e6, inside the domain).
//!
//! Differences from the paper's runs: PPM reconstruction instead of MP5,
//! Kerr-Schild coordinates (a = 0) instead of Schwarzschild.
//!
//! The paper's Figure 11 statements are: density within 1e-4 of the analytic
//! profile, F r^2 constant, residual radial velocity of order 1e-8 c.
//!
//! RESULT. Measured maxima over the cells (PPM, CFL 0.25, tight solver
//! tolerances, see below), four luminosities each:
//!
//!     time          density              F r^2      v^r / c
//!     1e8 M         0.9e-4 .. 1.5e-4     3.9e-3     0.4e-7 .. 2e-7
//!     2e9 M (bulk)  2.7e-3 .. 9.1e-3     3.9e-3     2e-7 .. 1e-6
//!
//! So the balance holds at the paper's level over a tenth of a sound
//! crossing, and drifts to the per-cent level over the paper's two
//! crossings: 30 to 90 times its quoted residual. "Bulk" is rho >= rho_in /
//! 10; in the outermost cell of the f = 1e-10 model, where the polytrope
//! nearly ends (rho = 0.005 rho_in), the error reaches 18 %. The flux sits
//! 0.2 % below the injected one on average with a +-0.2 % cell-to-cell
//! sawtooth. Both are properties of this scheme at 40 cells, not of the
//! paper's MP5 run, and are gated as measured.
//!
//! THREE THINGS THE RUN NEEDED:
//!  1. Tight inversion and Newton tolerances. This gas is cold, u / rho =
//!     2e-7, and C's U2PCONV = 1e-10 is relative to W = rho h gamma^2: the
//!     pressure is then only known to 1e-10 / 2e-7 = 5e-4, and the density
//!     shows 0.7 .. 1.6 % cell-to-cell noise at t = 1e8 (against 1e-4 with
//!     u2pconv = 1e-14).
//!  2. CFL 0.25. At 0.5 PPM + RK2 puts a 2.8 % jitter on the free-streaming
//!     flux (the same weak instability that costs the linear waves their
//!     order, paper2014_waves_tests.zig).
//!  3. The stationary M1 beam in the ghost cells below the base, not a
//!     fixed F / E (see `Beam`): halves the flux error and brings the
//!     density error at f >= 0.5 from 6e-4 down to 1e-4.

const std = @import("std");
const figdata = @import("../testing/figdata.zig");
const build_options = @import("build_options");
const config = @import("../config.zig");
const grid_mod = @import("../grid.zig");
const sim_mod = @import("../sim.zig");
const units_mod = @import("../units.zig");
const relele = @import("../relele.zig");
const hydro = @import("../physics/hydro.zig");
const thermo = @import("../physics/thermo.zig");
const radforce = @import("../physics/radforce.zig");
const invert = @import("../solve/invert.zig");
const invert_rad = @import("../solve/invert_rad.zig");
const implicit = @import("../solve/implicit.zig");
const Geometry = @import("../geometry.zig").Geometry;
const tubes = @import("../testing/tubes.zig");

const Grid = grid_mod.Grid;
const expect = std.testing.expect;

const cfg = config.Config{
    .modules = &.{ .hydro, .radiation },
    .reconstruction = .ppm,
    .flux = .laxf,
    .timestepping = .rk2imex,
    .coords = .ks,
};
const SimT = sim_mod.Sim(cfg);
const CoreT = SimT.CoreT;
const L = SimT.Layout;
const NV = SimT.nv;

const gam: f64 = 1.4;
const r_in: f64 = 1.0e6;
const r_out: f64 = 1.4e6;
const flux_factor: f64 = 0.99; // F / E of the injected beam

/// Stationary M1 radiation field streaming radially through a transparent
/// medium, as its flux factor phi = F / E. Energy conservation gives
/// r^2 F = const; the radial momentum equation of the M1 closure,
/// d(r^2 P) / dr = r (E - P) with P = chi(phi) E, then reads
///
///     d(chi / phi) / dr = (1 - chi) / (r phi),
///     chi = (3 + 4 phi^2) / (5 + 2 sqrt(4 - 3 phi^2)).
///
/// A beam injected with phi = 0.99 does not keep that value: it collimates
/// as it expands (phi = 0.9950 at the top of this atmosphere). Holding
/// phi = 0.99 in the ghost cells below the base, as a literal reading of
/// the paper's "E = F_in / 0.99" would, puts a kink in the radiation-frame
/// velocity at the boundary; PPM turns it into a cell-to-cell sawtooth that
/// the outgoing characteristics never damp, and the mean flux through the
/// atmosphere ends up 0.6 % below the injected one. The ghost cells and
/// the initial state therefore carry this solution, tabulated once.
const Beam = struct {
    const n = 4801;
    const r_lo: f64 = 0.96e6;
    const r_hi: f64 = 1.44e6;
    phi: [n]f64,

    fn chi(phi: f64) f64 {
        return (3.0 + 4.0 * phi * phi) / (5.0 + 2.0 * @sqrt(4.0 - 3.0 * phi * phi));
    }

    /// d phi / dr from the equation above, with d(chi / phi) / d phi taken
    /// by a centred difference.
    fn dphidr(r: f64, phi: f64) f64 {
        const h = 1.0e-7;
        const dg = (chi(phi + h) / (phi + h) - chi(phi - h) / (phi - h)) / (2.0 * h);
        return (1.0 - chi(phi)) / (r * phi) / dg;
    }

    fn rk4(r: f64, phi: f64, h: f64) f64 {
        const k1 = dphidr(r, phi);
        const k2 = dphidr(r + 0.5 * h, phi + 0.5 * h * k1);
        const k3 = dphidr(r + 0.5 * h, phi + 0.5 * h * k2);
        const k4 = dphidr(r + h, phi + h * k3);
        return phi + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
    }

    fn init() Beam {
        var b: Beam = undefined;
        const h = (r_hi - r_lo) / @as(f64, @floatFromInt(n - 1));
        const ibase: usize = @intFromFloat(@round((r_in - r_lo) / h));
        b.phi[ibase] = flux_factor;
        var i = ibase;
        while (i + 1 < n) : (i += 1) b.phi[i + 1] = rk4(r_lo + h * @as(f64, @floatFromInt(i)), b.phi[i], h);
        i = ibase;
        while (i > 0) : (i -= 1) b.phi[i - 1] = rk4(r_lo + h * @as(f64, @floatFromInt(i)), b.phi[i], -h);
        return b;
    }

    fn at(b: *const Beam, r: f64) f64 {
        const h = (r_hi - r_lo) / @as(f64, @floatFromInt(n - 1));
        const x = (r - r_lo) / h;
        const j: usize = @intFromFloat(@floor(x));
        const fr = x - @floor(x);
        return (1.0 - fr) * b.phi[j] + fr * b.phi[j + 1];
    }
};

/// Everything the initial state and both boundaries need.
const Atm = struct {
    beam: Beam,
    /// luminosity in Eddington units
    f: f64,
    rho_in: f64,
    k: f64, // p = K rho^Gamma
    f_in: f64, // flux at r_in (GU)

    fn init(f: f64, units: units_mod.Units, consts: *const thermo.Consts, kappa_es: f64) Atm {
        const rho_in = units.rhoCgs2Gu(1.0e-15);
        const u_in = thermo.uFromTrho(consts, 1.0e6, rho_in, gam);
        return .{
            .beam = Beam.init(),
            .f = f,
            .rho_in = rho_in,
            .k = (gam - 1.0) * u_in / std.math.pow(f64, rho_in, gam),
            .f_in = f / (kappa_es * r_in * r_in),
        };
    }

    fn rho(a: *const Atm, r: f64) f64 {
        const h = gam * a.k / (gam - 1.0);
        const c = h * std.math.pow(f64, a.rho_in, gam - 1.0) - (1.0 - a.f) / r_in;
        return std.math.pow(f64, (c + (1.0 - a.f) / r) / h, 1.0 / (gam - 1.0));
    }

    fn flux(a: *const Atm, r: f64) f64 {
        return a.f_in * (r_in * r_in) / (r * r);
    }

    /// Static gas on the analytic polytrope plus a radial beam with
    /// F / E = 0.99 in the frame of the gas.
    fn prims(a: *const Atm, core: *const CoreT, ix: i64) [NV]f64 {
        const r = core.grid.xc(ix);
        const geom = core.cache.fillGeometry(ix, 0, 0);
        const d = a.rho(r);
        const uint = a.k * std.math.pow(f64, d, gam) / (gam - 1.0);
        var pp: [NV]f64 = @splat(0);
        pp[L.index(.rho)] = d;
        pp[L.index(.uu)] = uint;
        pp[L.index(.entr)] = hydro.sFromU(d, uint, gam);
        setStatic(&pp, L.index(.vx), 0.0, r, &geom);

        // M1 with flux factor phi: the radiation rest frame moves at
        // beta = (2 - sqrt(4 - 3 phi^2)) / phi relative to the gas, and
        // E = Ebar (4 gamma^2 - 1) / 3.
        const phi = a.beam.at(r);
        const beta = (2.0 - @sqrt(4.0 - 3.0 * phi * phi)) / phi;
        const lor2 = 1.0 / (1.0 - beta * beta);
        const e_gas = a.flux(r) / phi;
        pp[L.index(.ee)] = 3.0 * e_gas / (4.0 * lor2 - 1.0);
        setStatic(&pp, L.index(.fx), @sqrt(lor2) * beta, r, &geom);
        return pp;
    }
};

/// Write the VELR primitive of a four-velocity that moves radially with
/// proper velocity `gbeta` = gamma beta relative to the static observer at
/// r. For a = 0 the radial coordinate component u^r = gamma beta
/// sqrt(1 - 2/r) is the same in Schwarzschild and Kerr-Schild coordinates.
fn setStatic(pp: *[NV]f64, slot: usize, gbeta: f64, r: f64, geom: *const Geometry) void {
    var ucon = [4]f64{ 0, gbeta * @sqrt(1.0 - 2.0 / r), 0, 0 };
    ucon[0] = relele.utFromSpatialUcon(ucon, geom) catch unreachable;
    const v = relele.convert(ucon, .vel4, .velr, geom, .recompute_ut) catch unreachable;
    pp[slot] = v[1];
    pp[slot + 1] = v[2];
    pp[slot + 2] = v[3];
}

fn bc(ctx: ?*const anyopaque, core: *const CoreT, ix: i64, iy: i64, iz: i64, t: f64, ifinit: bool, face: sim_mod.BcFace) relele.Error![NV]f64 {
    _ = iy;
    _ = iz;
    _ = t;
    _ = ifinit;
    const a: *const Atm = @ptrCast(@alignCast(ctx.?));
    // Gas: the analytic solution in every ghost cell, as in the paper.
    var pp = a.prims(core, ix);
    if (face == .xhi) {
        // Radiation leaves freely through the top: copy the last cell's
        // radiation-frame velocity and dilute its energy density as 1 / r^2.
        // (Continuing the tabulated beam here instead, scaled to the last
        // cell's luminosity, doubles the flux error: it feeds the cell-to-
        // cell sawtooth back in.)
        const il: i64 = @intCast(core.grid.nx - 1);
        const rl = core.grid.xc(il);
        const r = core.grid.xc(ix);
        pp[L.index(.ee)] = core.p.get(L.index(.ee), il, 0, 0) * (rl * rl) / (r * r);
        pp[L.index(.fx)] = core.p.get(L.index(.fx), il, 0, 0);
        pp[L.index(.fy)] = core.p.get(L.index(.fy), il, 0, 0);
        pp[L.index(.fz)] = core.p.get(L.index(.fz), il, 0, 0);
    }
    return pp;
}

const Result = struct {
    /// max |rho / rho_analytic - 1|
    drho: f64,
    /// max |F r^2 / (F_in r_in^2) - 1|, F the flux in the frame of the gas
    dflux: f64,
    /// max |v^r| / c
    vr: f64,
    nsteps: usize,
    nfail: u64,
    /// cell of the largest density error
    ix_worst: i64 = 0,
    /// the same over the cells with rho >= rho_in / 10
    drho_bulk: f64 = 0,
    ix_flux: i64 = 0,
    /// mean of F r^2 / (F_in r_in^2) - 1 over the cells
    flux_mean: f64 = 0,
};

/// Four-velocity (contravariant) of a VELR primitive triple.
fn ucon4(pp: *const [NV]f64, slot: usize, geom: *const Geometry) [4]f64 {
    return relele.convert(.{ 0, pp[slot], pp[slot + 1], pp[slot + 2] }, .velr, .vel4, geom, .recompute_ut) catch unreachable;
}

fn dot(a: [4]f64, b: [4]f64, geom: *const Geometry) f64 {
    var s: f64 = 0;
    for (0..4) |i| {
        for (0..4) |j| s += geom.gg[i][j] * a[i] * b[j];
    }
    return s;
}

fn run(alloc: std.mem.Allocator, f: f64, tend: f64, tight: bool, tsteplim: f64, fig: *figdata.Table) !Result {
    const units = units_mod.Units.init(1.0);
    const kappa_es = units.kappaCgs2Gu(0.4);
    const opac = radforce.Params.grey(units, tubes.comp, 0.0, kappa_es);
    const atm = Atm.init(f, units, &opac.consts, kappa_es);

    // A single equatorial cell in theta: the problem is 1D in r.
    const g = Grid.init(.{
        .nx = 40,
        .ng = 3,
        .minx = r_in,
        .maxx = r_out,
        .miny = 0.5 * std.math.pi - 0.05,
        .maxy = 0.5 * std.math.pi + 0.05,
    });
    // The atmosphere is as thin as E / rho ~ 1e-17 at f = 1e-10.
    var rad = invert_rad.RadParams.cdefault;
    rad.eerhoratiomin = 1.0e-50;
    rad.eeuuratiomin = 1.0e-50;
    var imp = implicit.ImplicitParams.cdefault;
    var floors = invert.FloorParams.cdefault;
    if (tight) {
        floors.u2pconv = 1.0e-14;
        imp.conv = 1.0e-15;
        imp.convrel = 1.0e-14;
        imp.convrelerr = 1.0e-15;
    }
    var s = try SimT.init(alloc, g, .{
        .phys = .{ .coords = .ks, .gam = gam, .floors = floors, .rad = rad, .opac = opac, .implicit = imp },
        .num = .{ .tsteplim = tsteplim },
        .bc = .{ .x = .{ .specific = .{ .f = &bc, .ctx = &atm } } },
    });
    defer s.deinit();

    var ix: i64 = 0;
    while (ix < s.nxi()) : (ix += 1) try s.initCell(ix, 0, 0, atm.prims(&s.core, ix));
    try s.finishInit();

    var nsteps: usize = 0;
    while (s.t < tend) {
        var dt: ?f64 = null;
        if (s.t + 1.0 / s.core.tstepdenmax > tend) dt = tend - s.t;
        try s.step(dt);
        nsteps += 1;
    }

    var res = Result{ .drho = 0, .dflux = 0, .vr = 0, .nsteps = nsteps, .nfail = s.n_radimp_failures };
    fig.block("L = {e} L_Edd, t = {e}", .{ f, tend });
    fig.meta("scales", "rho_in {e} F_in {e}", .{ atm.rho_in, atm.f_in });
    ix = 0;
    while (ix < s.nxi()) : (ix += 1) {
        const r = g.xc(ix);
        const geom = s.core.cache.fillGeometry(ix, 0, 0);
        var pp: [NV]f64 = undefined;
        s.core.p.load(ix, 0, 0, &pp);
        const dr = @abs(pp[L.index(.rho)] / atm.rho(r) - 1.0);
        if (atm.rho(r) >= 0.1 * atm.rho_in) res.drho_bulk = @max(res.drho_bulk, dr);
        if (dr > res.drho) {
            res.drho = dr;
            res.ix_worst = ix;
        }

        const ug = ucon4(&pp, L.index(.vx), &geom);
        const ur = ucon4(&pp, L.index(.fx), &geom);
        // radial velocity seen by the static observer
        res.vr = @max(res.vr, @abs(ug[1]) / @sqrt(1.0 - 2.0 / r));
        // flux in the gas frame: |F| = (4/3) Ebar G sqrt(G^2 - 1), G = -u_R.u
        const grel = -dot(ug, ur, &geom);
        const fhat = 4.0 / 3.0 * pp[L.index(.ee)] * grel * @sqrt(@max(grel * grel - 1.0, 0.0));
        if (@abs(fhat / atm.flux(r) - 1.0) > res.dflux) res.ix_flux = ix;
        res.dflux = @max(res.dflux, @abs(fhat / atm.flux(r) - 1.0));
        res.flux_mean += (fhat / atm.flux(r) - 1.0) / 40.0;
        fig.row(&.{ r, pp[L.index(.rho)] / atm.rho_in, atm.rho(r) / atm.rho_in, fhat * r * r / (r_in * r_in), atm.flux(r) * r * r / (r_in * r_in), ug[1] / @sqrt(1.0 - 2.0 / r) });
    }
    return res;
}

const luminosities = [_]f64{ 1.0e-10, 0.1, 0.5, 1.0 };

const Limits = struct { drho: f64, dflux: f64, vr: f64 };

fn check(models_f: []const f64, tend: f64, lim: Limits, fig: *figdata.Table) !void {
    fig.meta("source", "Sadowski et al. 2013, section 4.6, Figure 11: static atmosphere, 40 cells", .{});
    fig.meta("columns", "r rho/rho_in rho_exact/rho_in F(r/r_in)^2 F_exact(r/r_in)^2 v^r   (F in code units, gas frame)", .{});
    var ok = true;
    for (models_f) |f| {
        const r = try run(std.testing.allocator, f, tend, true, 0.25, fig);
        std.debug.print(
            "paper 2013 atmosphere L = {e:.1} L_Edd, t = {e:.1}: max |drho/rho| {e:.2} (bulk {e:.2}), max |d(F r^2)| {e:.2}, max |v^r| {e:.2} ({d} steps, {d} implicit failures, worst cells: density {d}, flux {d}; mean flux offset {e:.2})\n",
            .{ f, tend, r.drho, r.drho_bulk, r.dflux, r.vr, r.nsteps, r.nfail, r.ix_worst, r.ix_flux, r.flux_mean },
        );
        ok = ok and r.drho_bulk <= lim.drho and r.dflux <= lim.dflux and r.vr <= lim.vr and r.nfail == 0;
    }
    try expect(ok);
}

test "Sadowski 2013 section 4.6: static atmosphere stays in radiative-hydrostatic balance" {
    // The two models where radiation carries half and all of the weight,
    // over 1 % of a sound crossing (25 light crossings): a force imbalance
    // shows as drift. Measured at t = 2.5e7: density 2.0e-5 and 2.4e-5,
    // velocity 4e-8 and 7e-8, i.e. the paper's 1e-4 and 1e-8 levels.
    var fig = figdata.Table.init("k13_atmosphere_fast");
    defer fig.finish();
    try check(&.{ 0.5, 1.0 }, 1.0e7, .{ .drho = 1.0e-4, .dflux = 6.0e-3, .vr = 2.0e-7 }, &fig);
}

test "Sadowski 2013 section 4.6: all four atmospheres over a tenth of a sound crossing" {
    if (!build_options.slow_tests) return error.SkipZigTest;
    // measured: density 0.9e-4..1.5e-4, flux 3.9e-3, velocity 0.4e-7..2e-7
    var fig = figdata.Table.init("k13_atmosphere_short");
    defer fig.finish();
    try check(&luminosities, 1.0e8, .{ .drho = 3.0e-4, .dflux = 6.0e-3, .vr = 5.0e-7 }, &fig);
}

test "Sadowski 2013 section 4.6: static atmosphere at the paper's t = 2e9 (Figure 11)" {
    if (!build_options.slow_tests) return error.SkipZigTest;
    // About 12 minutes in a release build (800,000 steps per model).
    // Measured: bulk density 2.7e-3..9.1e-3, flux 3.9e-3, velocity up to
    // 1.3e-5 (f = 1e-10, at the edge of the polytrope).
    var fig = figdata.Table.init("k13_atmosphere");
    defer fig.finish();
    try check(&luminosities, 2.0e9, .{ .drho = 2.0e-2, .dflux = 6.0e-3, .vr = 3.0e-5 }, &fig);
}
