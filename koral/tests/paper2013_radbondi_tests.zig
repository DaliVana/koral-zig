//! Harness for Sadowski et al. (2013), section 4.8, Table 5 and Figures
//! 13-14: spherical accretion onto a non-rotating black hole with radiation,
//! in the setup of Fragile et al. (2012).
//!
//! THIS IS NOT YET A REPRODUCTION. The setup below is the paper's, it runs
//! with no implicit failures, and it is kept so the production-resolution
//! run for the paper can start from it; but at any resolution a test can
//! afford the answer is wrong, for a reason that is measured and gated here:
//!
//!   The gas is cold and falls at Mach ~ 800 (u / rho ~ 1e-6 against
//!   v^2 ~ 1e-3 at r = 1000). The total-energy update leaves a truncation
//!   error far larger than the internal energy, so the gas is heated
//!   spuriously, and bremsstrahlung, ~ rho^2 sqrt(T), follows. For model
//!   E10T6 at t = 2000:
//!       64 cells:  T / T_adiabat = 15 at r = 30,  L(1000) / L_Edd = 3.6e-3
//!      128 cells:  T / T_adiabat = 5.2,           L(1000) / L_Edd = 8.7e-4
//!   against 6.5e-6 in the paper's Table 5 (512 cells, MP5), and the
//!   luminosity is still rising at t = 12000. The error falls roughly as
//!   N^-2, which still leaves a factor of several at 512 cells with linear
//!   reconstruction; a 512-cell model costs hours, the innermost cell being
//!   0.0065 M wide.
//!
//! Setup. Gas falls freely, u^r = -sqrt(2 / r), with rho = Mdot / (4 pi r^2
//! |u^r|) and T = T_out (rho / rho_out)^(Gamma - 1); it cools by thermal
//! bremsstrahlung and scatters on electrons (kappa_es = 0.4 cm^2/g). The
//! radiation starts far below equilibrium, E = 3 f_p p_gas. The quantity of
//! interest is the luminosity L = 4 pi r^2 F at r = 1000 in Eddington units.
//! Grid: logarithmic in r - 2.2 between r_in = 2.5 and r_out = 2e4, which is
//! this code's MKS2 radial coordinate with R0 = 2.2. M = 3 Msun. Outer
//! boundary held at the initial state; inner boundary outflow with the
//! free-fall velocity and rho ~ r^-3/2, u ~ r^(-3 Gamma / 2).
//!
//! Open choices and differences from the paper:
//!  * Mdot_Edd is taken as L_Edd / c^2 (the paper does not define it);
//!  * the absorption opacity is this code's production bremsstrahlung
//!    channel (Gaunt factor 1.2, relativistic correction) instead of the
//!    Kramers form 1.7e-25 T^-7/2 rho^2 / m_p^2 the paper takes from
//!    Fragile et al.;
//!  * linear reconstruction instead of MP5.

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
    .reconstruction = .linear,
    .flux = .laxf,
    .timestepping = .rk2imex,
    .coords = .mks2,
};
const SimT = sim_mod.Sim(cfg);
const CoreT = SimT.CoreT;
const L = SimT.Layout;
const NV = SimT.nv;

const mks_r0: f64 = 2.2;
const r_in: f64 = 2.5;
const r_out: f64 = 2.0e4;

const Model = struct {
    name: []const u8,
    /// accretion rate in units of L_Edd / c^2
    mdot: f64,
    t_out: f64,
    fp: f64,
    /// L / L_Edd of the paper's Table 5
    lum_paper: f64,
};

const models = [_]Model{
    .{ .name = "E1T6", .mdot = 1.0, .t_out = 1.0e6, .fp = 1.2e-4, .lum_paper = 8.73e-8 },
    .{ .name = "E10T5", .mdot = 10.0, .t_out = 1.0e5, .fp = 1.2e-7, .lum_paper = 3.26e-6 },
    .{ .name = "E10T6", .mdot = 10.0, .t_out = 1.0e6, .fp = 1.2e-4, .lum_paper = 6.51e-6 },
    .{ .name = "E10T7", .mdot = 10.0, .t_out = 1.0e7, .fp = 1.2e-1, .lum_paper = 1.45e-5 },
    .{ .name = "E100T6", .mdot = 100.0, .t_out = 1.0e6, .fp = 1.2e-4, .lum_paper = 2.00e-4 },
};

const Setup = struct {
    consts: thermo.Consts,
    gam: f64,
    /// Mdot / (4 pi) in code units
    mdot4pi: f64,
    rho_out: f64,
    t_out: f64,
    fp: f64,

    fn rho(st: *const Setup, r: f64) f64 {
        return st.mdot4pi / (r * r * @sqrt(2.0 / r));
    }

    /// Initial / outer-boundary state at the cell (ix).
    fn prims(st: *const Setup, core: *const CoreT, ix: i64) [NV]f64 {
        const geom = core.cache.fillGeometry(ix, 0, 0);
        const r = radius(core, ix);
        const d = st.rho(r);
        const temp = st.t_out * std.math.pow(f64, d / st.rho_out, st.gam - 1.0);
        const uint = thermo.uFromTrho(&st.consts, temp, d, st.gam);
        return st.state(d, uint, 3.0 * st.fp * (st.gam - 1.0) * uint, r, &geom);
    }

    /// Free-falling gas with radiation at rest in the gas frame.
    fn state(st: *const Setup, d: f64, uint: f64, erad: f64, r: f64, geom: *const Geometry) [NV]f64 {
        var pp: [NV]f64 = @splat(0);
        pp[L.index(.rho)] = d;
        pp[L.index(.uu)] = uint;
        pp[L.index(.entr)] = hydro.sFromU(d, uint, st.gam);
        // u^r = -sqrt(2 / r) in Kerr-Schild; u^x1 = u^r / (dr / dx1)
        var ucon = [4]f64{ 0, -@sqrt(2.0 / r) / (r - mks_r0), 0, 0 };
        ucon[0] = relele.utFromSpatialUcon(ucon, geom) catch unreachable;
        const v = relele.convert(ucon, .vel4, .velr, geom, .recompute_ut) catch unreachable;
        for (0..3) |k| {
            pp[L.index(.vx) + k] = v[1 + k];
            pp[L.index(.fx) + k] = v[1 + k];
        }
        pp[L.index(.ee)] = erad;
        return pp;
    }
};

fn radius(core: *const CoreT, ix: i64) f64 {
    return mks_r0 + @exp(core.grid.xc(ix));
}

fn bc(ctx: ?*const anyopaque, core: *const CoreT, ix: i64, iy: i64, iz: i64, t: f64, ifinit: bool, face: sim_mod.BcFace) relele.Error![NV]f64 {
    _ = iy;
    _ = iz;
    _ = t;
    _ = ifinit;
    const st: *const Setup = @ptrCast(@alignCast(ctx.?));
    if (face == .xhi) return st.prims(core, ix);
    // inner edge: free-fall velocity, rho ~ r^-3/2, u ~ r^(-3 Gamma / 2),
    // radiation energy density copied
    const geom = core.cache.fillGeometry(ix, 0, 0);
    const r = radius(core, ix);
    const r0 = radius(core, 0);
    const d = core.p.get(L.index(.rho), 0, 0, 0) * std.math.pow(f64, r / r0, -1.5);
    const uint = core.p.get(L.index(.uu), 0, 0, 0) * std.math.pow(f64, r / r0, -1.5 * st.gam);
    var pp = st.state(d, uint, core.p.get(L.index(.ee), 0, 0, 0), r, &geom);
    // radiation leaves with the velocity it has in the first cell
    for (0..3) |k| pp[L.index(.fx) + k] = core.p.get(L.index(.fx) + k, 0, 0, 0);
    return pp;
}

const Result = struct {
    /// L / L_Edd = r^2 F kappa_es at r = 1000 and at r = 300
    lum_1000: f64,
    lum_300: f64,
    /// max |rho / rho_freefall - 1| over 10 < r < 5000
    drho: f64,
    nsteps: usize,
    nfail: u64,
    seconds: f64,
    /// T_gas over the adiabat T_out (rho / rho_out)^(Gamma - 1) at r = 30, 300, 3000
    t_ratio: [3]f64 = .{ 0, 0, 0 },
};

/// Luminosity through the sphere of cell ix in Eddington units:
/// L = -4 pi r^2 R^r_t and L_Edd = 4 pi / kappa_es (G = c = M = 1).
fn luminosity(s: *const SimT, ix: i64, kappa_es: f64) f64 {
    const geom = s.core.cache.fillGeometry(ix, 0, 0);
    const r = radius(&s.core, ix);
    var pp: [NV]f64 = undefined;
    s.core.p.load(ix, 0, 0, &pp);
    const ur = relele.convert(.{ 0, pp[L.index(.fx)], pp[L.index(.fy)], pp[L.index(.fz)] }, .velr, .vel4, &geom, .recompute_ut) catch unreachable;
    var ur_t: f64 = 0;
    for (0..4) |j| ur_t += geom.gg[0][j] * ur[j];
    // R^x1_t = (4/3) Ebar u_R^x1 u_R,t;  R^r_t = (dr / dx1) R^x1_t
    const rrt = 4.0 / 3.0 * pp[L.index(.ee)] * ur[1] * ur_t * (r - mks_r0);
    return -r * r * rrt * kappa_es;
}

fn cellAt(s: *const SimT, r: f64) i64 {
    return @intFromFloat(@floor((@log(r - mks_r0) - s.core.grid.minx) / s.core.grid.dx));
}

fn run(alloc: std.mem.Allocator, m: *const Model, nx: usize, tend: f64, fig: *figdata.Table) !Result {
    const units = units_mod.Units.init(3.0);
    const kappa_es = units.kappaCgs2Gu(0.4);
    var opac = radforce.Params.init(units, tubes.comp);
    opac.channels = .{ .bremsstrahlung = true, .synchrotron = false, .kleinnishina = false, .comptonization = false };
    opac.kappaes = .{ .grey = kappa_es };

    const gam = 1.0 + (1.0 / 3.0) * (2.0 + 2.0 * m.fp) / (1.0 + 2.0 * m.fp);
    // Mdot_Edd = L_Edd / c^2 = 4 pi / kappa_es
    const mdot4pi = m.mdot / kappa_es;
    var st = Setup{
        .consts = opac.consts,
        .gam = gam,
        .mdot4pi = mdot4pi,
        .rho_out = mdot4pi / (r_out * r_out * @sqrt(2.0 / r_out)),
        .t_out = m.t_out,
        .fp = m.fp,
    };

    const g = Grid.init(.{
        .nx = nx,
        .ng = 2,
        .minx = @log(r_in - mks_r0),
        .maxx = @log(r_out - mks_r0),
        // one equatorial cell; x2 = 1/2 is theta = pi/2 in MKS2
        .miny = 0.49,
        .maxy = 0.51,
    });
    var floors = invert.FloorParams.cdefault;
    floors.u2pconv = 1.0e-14; // cold gas: see paper2013_atmosphere_tests.zig
    floors.uurhoratiomin = 1.0e-14;
    var s = try SimT.init(alloc, g, .{
        .phys = .{
            .coords = .mks2,
            .mp = .{ .a = 0, .mksr0 = mks_r0, .mksh0 = 0.7 },
            .gam = gam,
            .floors = floors,
            .rad = invert_rad.RadParams.cdefault,
            .opac = opac,
            .implicit = implicit.ImplicitParams.puffy,
        },
        .num = .{ .minmod_theta = 1.5, .fixups = .{ .radimp = true } },
        .bc = .{ .x = .{ .specific = .{ .f = &bc, .ctx = &st } } },
    });
    defer s.deinit();

    var ix: i64 = 0;
    while (ix < s.nxi()) : (ix += 1) try s.initCell(ix, 0, 0, st.prims(&s.core, ix));
    try s.finishInit();

    const start = sim_mod.nowNs();
    var nsteps: usize = 0;
    while (s.t < tend) {
        var dt: ?f64 = null;
        if (s.t + 1.0 / s.core.tstepdenmax > tend) dt = tend - s.t;
        try s.step(dt);
        nsteps += 1;
    }
    const secs = @as(f64, @floatFromInt(sim_mod.nowNs() - start)) * 1.0e-9;

    var res = Result{
        .lum_1000 = luminosity(&s, cellAt(&s, 1000.0), kappa_es),
        .lum_300 = luminosity(&s, cellAt(&s, 300.0), kappa_es),
        .drho = 0,
        .nsteps = nsteps,
        .nfail = s.n_radimp_failures,
        .seconds = secs,
    };
    for ([_]f64{ 30.0, 300.0, 3000.0 }, 0..) |rq, k| {
        const jx = cellAt(&s, rq);
        const r = radius(&s.core, jx);
        const d = s.core.p.get(L.index(.rho), jx, 0, 0);
        const t_gas = thermo.tFromUrho(&st.consts, s.core.p.get(L.index(.uu), jx, 0, 0), d, gam);
        res.t_ratio[k] = t_gas / (st.t_out * std.math.pow(f64, st.rho(r) / st.rho_out, gam - 1.0));
    }
    fig.block("model {s} cells {d} t {d}", .{ m.name, nx, tend });
    ix = 0;
    while (ix < s.nxi()) : (ix += 1) {
        const r = radius(&s.core, ix);
        const d = s.core.p.get(L.index(.rho), ix, 0, 0);
        const t_gas = thermo.tFromUrho(&st.consts, s.core.p.get(L.index(.uu), ix, 0, 0), d, gam);
        fig.row(&.{
            r,                                                                 d / st.rho(r),
            t_gas,                                                             st.t_out * std.math.pow(f64, st.rho(r) / st.rho_out, gam - 1.0),
            s.core.p.get(L.index(.ee), ix, 0, 0),                              luminosity(&s, ix, kappa_es),
        });
    }
    ix = 0;
    while (ix < s.nxi()) : (ix += 1) {
        const r = radius(&s.core, ix);
        if (r < 10.0 or r > 5000.0) continue;
        res.drho = @max(res.drho, @abs(s.core.p.get(L.index(.rho), ix, 0, 0) / st.rho(r) - 1.0));
    }
    return res;
}

test "Sadowski 2013 section 4.8: radiative Bondi runs, and its spurious heating falls with resolution" {
    if (!build_options.slow_tests) return error.SkipZigTest;
    const a = std.testing.allocator;
    const m = &models[2]; // E10T6
    var res: [2]Result = undefined;
    var fig = figdata.Table.init("k13_radbondi");
    defer fig.finish();
    fig.meta("source", "Sadowski et al. 2013, section 4.8, Figure 13 setup. NOT a reproduction: coarse grids, see the test header", .{});
    fig.meta("columns", "r rho/rho_freefall T_gas T_adiabat Ebar L(r)/L_Edd", .{});
    fig.meta("paper_luminosity", "{e}", .{m.lum_paper});
    for ([_]usize{ 32, 64 }, 0..) |nx, k| {
        res[k] = try run(a, m, nx, 2000.0, &fig);
        const r = res[k];
        std.debug.print(
            "paper 2013 radiative Bondi {s}, {d} cells, t = 2000: L(1000)/L_Edd {e:.3} (paper {e:.3}), L(300)/L_Edd {e:.3}, max |drho| {e:.2}, T/T_adiabat at r=30,300,3000: {d:.3} {d:.3} {d:.3}, {d} steps, {d} implicit failures, {d:.1} s\n",
            .{ m.name, nx, r.lum_1000, m.lum_paper, r.lum_300, r.drho, r.t_ratio[0], r.t_ratio[1], r.t_ratio[2], r.nsteps, r.nfail, r.seconds },
        );
        try expect(r.nfail == 0);
        // the density stays on the free-fall profile
        try expect(r.drho <= 0.1);
    }
    // Not a reproduction (see the header): the gas is hotter than the
    // adiabat and the luminosity too high, both less so on the finer grid.
    try expect(res[1].t_ratio[0] < res[0].t_ratio[0]);
    try expect(res[1].lum_1000 < res[0].lum_1000);
    try expect(res[1].lum_1000 > m.lum_paper);
}
