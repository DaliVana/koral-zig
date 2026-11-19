//! Reproduction gate for Sadowski et al. (2013), section 4.7, Table 4 and
//! Figure 12: a beam of light emitted azimuthally near a Schwarzschild black
//! hole, with gas and radiation decoupled (kappa = chi = 0). It tests the
//! radiative transport terms in curved spacetime and nothing else.
//!
//! Three models, each on 30 cells uniform in r between r_in and r_out and 60
//! cells uniform in phi between 0 and pi/2, in the equatorial plane:
//!
//!     model   r_beam        r_in    r_out
//!       1     3.0 +- 0.1    2.5     3.5     (the photon orbit)
//!       2     6.0 +- 0.2    5.3     7.5
//!       3     16.0 +- 0.5   14.0    20.5
//!
//! The beam enters through the phi = 0 boundary over the r_beam range with
//! T_beam = 1e10 K = 1000 T0 and F^phi = 0.99999 E for the static observer;
//! every other boundary is outflow. Linear reconstruction, minmod theta = 1,
//! as in the paper.
//!
//! The paper overlays photon geodesics computed with GYOTO on a colour map.
//! Here the same comparison is a number: a photon launched tangentially at
//! r0 follows, with u = 1 / r,
//!
//!     d^2 u / d phi^2 + u = 3 u^2,     u(0) = 1 / r0,  u'(0) = 0,
//!
//! integrated here with RK4. At phi = pi/16, pi/8 and 3 pi/16, where all
//! three beams are still inside r_out, the energy-weighted mean radius of
//! the beam is compared with the geodesic of the beam centre, at the paper's
//! resolution and at twice that.
//!
//! Differences from the paper's runs: Kerr-Schild coordinates with a = 0
//! (the radial coordinate and the static observer are the same as in
//! Schwarzschild coordinates), RK2 in time.

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
const invert_rad = @import("../solve/invert_rad.zig");
const Geometry = @import("../geometry.zig").Geometry;
const tubes = @import("../testing/tubes.zig");

const Grid = grid_mod.Grid;
const expect = std.testing.expect;

const cfg = config.Config{
    .modules = &.{ .hydro, .radiation },
    .reconstruction = .linear,
    .flux = .laxf,
    .timestepping = .rk2imex,
    .coords = .ks,
};
const SimT = sim_mod.Sim(cfg);
const CoreT = SimT.CoreT;
const L = SimT.Layout;
const NV = SimT.nv;

const gam: f64 = 4.0 / 3.0;
const t_amb: f64 = 1.0e7;
const t_beam: f64 = 1.0e10;
const flux_factor: f64 = 0.99999;
const rho_amb: f64 = 1.0e-30; // "negligibly small values for all primitives"

const Model = struct {
    name: []const u8,
    r_beam: f64,
    half_width: f64,
    r_in: f64,
    r_out: f64,
};

const models = [_]Model{
    .{ .name = "1", .r_beam = 3.0, .half_width = 0.1, .r_in = 2.5, .r_out = 3.5 },
    .{ .name = "2", .r_beam = 6.0, .half_width = 0.2, .r_in = 5.3, .r_out = 7.5 },
    .{ .name = "3", .r_beam = 16.0, .half_width = 0.5, .r_in = 14.0, .r_out = 20.5 },
};

const Setup = struct {
    consts: thermo.Consts,
    model: *const Model,

    /// Static gas with radiation in LTE at the ambient temperature.
    fn ambient(st: *const Setup, core: *const CoreT, ix: i64) [NV]f64 {
        const r = core.grid.xc(ix);
        const geom = core.cache.fillGeometry(ix, 0, 0);
        const uint = thermo.uFromTrho(&st.consts, t_amb, rho_amb, gam);
        var pp: [NV]f64 = @splat(0);
        pp[L.index(.rho)] = rho_amb;
        pp[L.index(.uu)] = uint;
        pp[L.index(.entr)] = hydro.sFromU(rho_amb, uint, gam);
        setVelocity(&pp, L.index(.vx), 0.0, r, &geom);
        pp[L.index(.ee)] = st.consts.lteEfromT(t_amb);
        setVelocity(&pp, L.index(.fx), 0.0, r, &geom);
        return pp;
    }

    /// The beam: E = 4 sigma T_beam^4 and F^phi = 0.99999 E in the frame of
    /// the static observer at r.
    fn beam(st: *const Setup, core: *const CoreT, ix: i64) [NV]f64 {
        const r = core.grid.xc(ix);
        const geom = core.cache.fillGeometry(ix, 0, 0);
        var pp = st.ambient(core, ix);
        const beta = (2.0 - @sqrt(4.0 - 3.0 * flux_factor * flux_factor)) / flux_factor;
        const lor2 = 1.0 / (1.0 - beta * beta);
        pp[L.index(.ee)] = 3.0 * st.consts.lteEfromT(t_beam) / (4.0 * lor2 - 1.0);
        setVelocity(&pp, L.index(.fx), @sqrt(lor2) * beta, r, &geom);
        return pp;
    }
};

/// VELR primitive of a four-velocity moving azimuthally with proper velocity
/// `gbeta` = gamma beta relative to the static observer at radius r in the
/// equatorial plane: u^phi = gamma beta / r, u^r = 0.
fn setVelocity(pp: *[NV]f64, slot: usize, gbeta: f64, r: f64, geom: *const Geometry) void {
    var ucon = [4]f64{ 0, 0, 0, gbeta / r };
    ucon[0] = relele.utFromSpatialUcon(ucon, geom) catch unreachable;
    const v = relele.convert(ucon, .vel4, .velr, geom, .recompute_ut) catch unreachable;
    pp[slot] = v[1];
    pp[slot + 1] = v[2];
    pp[slot + 2] = v[3];
}

fn bc(ctx: ?*const anyopaque, core: *const CoreT, ix: i64, iy: i64, iz: i64, t: f64, ifinit: bool, face: sim_mod.BcFace) relele.Error![NV]f64 {
    _ = iy;
    _ = t;
    _ = ifinit;
    const st: *const Setup = @ptrCast(@alignCast(ctx.?));
    const g = &core.grid;
    const nx: i64 = @intCast(g.nx);
    const nz: i64 = @intCast(g.nz);
    const jx = std.math.clamp(ix, 0, nx - 1);
    const jz = std.math.clamp(iz, 0, nz - 1);
    if (face == .zlo) {
        const r = g.xc(jx);
        if (@abs(r - st.model.r_beam) <= st.model.half_width) return st.beam(core, jx);
    }
    // outflow everywhere else: the nearest domain cell
    var pp: [NV]f64 = undefined;
    core.p.load(jx, 0, jz, &pp);
    return pp;
}

/// Radius of a photon launched tangentially at r0 after sweeping phi.
fn geodesicRadius(r0: f64, phi: f64) f64 {
    const n: usize = 4000;
    const h = phi / @as(f64, @floatFromInt(n));
    var u = 1.0 / r0;
    var w: f64 = 0.0; // du / dphi
    for (0..n) |_| {
        const k1u = w;
        const k1w = 3.0 * u * u - u;
        const k2u = w + 0.5 * h * k1w;
        const k2w = 3.0 * (u + 0.5 * h * k1u) * (u + 0.5 * h * k1u) - (u + 0.5 * h * k1u);
        const k3u = w + 0.5 * h * k2w;
        const k3w = 3.0 * (u + 0.5 * h * k2u) * (u + 0.5 * h * k2u) - (u + 0.5 * h * k2u);
        const k4u = w + h * k3w;
        const k4w = 3.0 * (u + h * k3u) * (u + h * k3u) - (u + h * k3u);
        u += h / 6.0 * (k1u + 2.0 * k2u + 2.0 * k3u + k4u);
        w += h / 6.0 * (k1w + 2.0 * k2w + 2.0 * k3w + k4w);
    }
    return 1.0 / u;
}

test "geodesic integrator: the photon orbit is circular and a distant ray is straight" {
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), geodesicRadius(3.0, 0.5 * std.math.pi), 1e-9);
    // flat-space limit: r = r0 / cos(phi), bent inward by ~ M
    const r = geodesicRadius(1.0e4, 0.25 * std.math.pi);
    try expect(r < 1.0e4 * @sqrt(2.0) and r > 1.0e4 * @sqrt(2.0) - 3.0);
}

/// Azimuths, in units of pi, where the beam is measured. All three models
/// still have the whole beam inside r_out at 3 pi / 16.
const fractions = [_]f64{ 0.0625, 0.125, 0.1875 };

const Slice = struct {
    /// energy-weighted mean radius and rms half-width of the beam
    centroid: f64,
    width: f64,
    /// peak energy density relative to the injected one
    peak: f64,
};

fn run(alloc: std.mem.Allocator, m: *const Model, refine: usize, tend: f64, out: *[3]Slice, fig: *figdata.Table) !void {
    const units = units_mod.Units.init(10.0);
    var st = Setup{ .consts = thermo.Consts.init(units, tubes.comp), .model = m };

    const g = Grid.init(.{
        .nx = 30 * refine,
        .nz = 60 * refine,
        .ng = 2,
        .minx = m.r_in,
        .maxx = m.r_out,
        .miny = 0.5 * std.math.pi - 0.05,
        .maxy = 0.5 * std.math.pi + 0.05,
        .minz = 0.0,
        .maxz = 0.5 * std.math.pi,
    });
    var rad = invert_rad.RadParams.cdefault;
    rad.eerhoratiomax = 1.0e100; // radiation in vacuum
    rad.eeuuratiomax = 1.0e100;
    rad.gammamaxrad = 1000.0; // the beam has a radiation Lorentz factor of 158
    var s = try SimT.init(alloc, g, .{
        // opac = null: no coupling at all (C: SKIPRADSOURCE)
        .phys = .{ .coords = .ks, .gam = gam, .rad = rad },
        .num = .{ .minmod_theta = 1.0 },
        .bc = .{
            .x = .{ .specific = .{ .f = &bc, .ctx = &st } },
            .z = .{ .specific = .{ .f = &bc, .ctx = &st } },
        },
    });
    defer s.deinit();

    var iz: i64 = 0;
    while (iz < s.nzi()) : (iz += 1) {
        var ix: i64 = 0;
        while (ix < s.nxi()) : (ix += 1) try s.initCell(ix, 0, iz, st.ambient(&s.core, ix));
    }
    try s.finishInit();

    while (s.t < tend) {
        var dt: ?f64 = null;
        if (s.t + 1.0 / s.core.tstepdenmax > tend) dt = tend - s.t;
        try s.step(dt);
    }

    // injected coordinate-frame energy density at the beam centre, for scale
    const e_ref = blk: {
        var worst: f64 = 0;
        var ix: i64 = 0;
        while (ix < s.nxi()) : (ix += 1) worst = @max(worst, labEnergy(&s, ix, 0));
        break :blk worst;
    };
    // figure data: the whole r-phi map, scaled to the injected energy density
    fig.block("model {s} grid {d}x{d} r_beam {d} half_width {d}", .{ m.name, 30 * refine, 60 * refine, m.r_beam, m.half_width });
    {
        var jz: i64 = 0;
        while (jz < s.nzi()) : (jz += 1) {
            var jx: i64 = 0;
            while (jx < s.nxi()) : (jx += 1) fig.row(&.{ g.xc(jx), g.zc(jz), labEnergy(&s, jx, jz) / e_ref });
        }
    }
    for (fractions, 0..) |frac, k| {
        const zi: i64 = @intFromFloat(@floor(frac * std.math.pi / g.dz));
        var m0: f64 = 0;
        var m1: f64 = 0;
        var peak: f64 = 0;
        var ix: i64 = 0;
        while (ix < s.nxi()) : (ix += 1) {
            const e = labEnergy(&s, ix, zi);
            m0 += e;
            m1 += e * g.xc(ix);
            peak = @max(peak, e);
        }
        const c = m1 / m0;
        var m2: f64 = 0;
        ix = 0;
        while (ix < s.nxi()) : (ix += 1) m2 += labEnergy(&s, ix, zi) * (g.xc(ix) - c) * (g.xc(ix) - c);
        out[k] = .{ .centroid = c, .width = @sqrt(m2 / m0), .peak = peak / e_ref };
    }
}

/// Coordinate-frame radiation energy density -R^t_t (the conserved variable
/// over sqrt(-g)).
fn labEnergy(s: *const SimT, ix: i64, iz: i64) f64 {
    const geom = s.core.cache.fillGeometry(ix, 0, iz);
    return -s.core.u.get(L.index(.ee), ix, 0, iz) / geom.gdet;
}

test "Sadowski 2013 section 4.7: light beams near a black hole follow photon geodesics (Figure 12)" {
    if (!build_options.slow_tests) return error.SkipZigTest;
    const a = std.testing.allocator;
    var ok = true;
    var fig = figdata.Table.init("k13_beam_maps");
    defer fig.finish();
    fig.meta("source", "Sadowski et al. 2013, section 4.7, Figure 12: light beams near a Schwarzschild black hole", .{});
    fig.meta("columns", "r phi E/E_injected   (coordinate-frame energy density -R^t_t)", .{});
    var geo = figdata.Table.init("k13_beam_geodesics");
    defer geo.finish();
    geo.meta("source", "photon geodesics launched tangentially at phi = 0 from r_beam and r_beam +- half_width", .{});
    geo.meta("columns", "phi r_centre r_inner r_outer", .{});
    for (&models) |*m| {
        geo.block("model {s}", .{m.name});
        for (0..91) |j| {
            const phi = @as(f64, @floatFromInt(j)) * (0.5 * std.math.pi / 90.0);
            geo.row(&.{ phi, geodesicRadius(m.r_beam, phi), geodesicRadius(m.r_beam - m.half_width, phi), geodesicRadius(m.r_beam + m.half_width, phi) });
        }
        const dr = (m.r_out - m.r_in) / 30.0;
        var off: [2][3]f64 = undefined;
        for ([_]usize{ 1, 2 }, 0..) |refine, ir| {
            var sl: [3]Slice = undefined;
            // long enough for the beam to cross the quarter circle twice
            try run(a, m, refine, 2.0 * m.r_out * std.math.pi, &sl, &fig);
            for (fractions, 0..) |frac, k| {
                const phi = frac * std.math.pi;
                const exact = geodesicRadius(m.r_beam, phi);
                off[ir][k] = (sl[k].centroid - exact) / dr;
                std.debug.print(
                    "paper 2013 beam model {s}, {d}x{d}, phi = {d:.4} pi: centroid r = {d:.4}, geodesic {d:.4} ({d:.2} base cells off), rms half-width {d:.3}, peak {d:.3}\n",
                    .{ m.name, 30 * refine, 60 * refine, frac, sl[k].centroid, exact, off[ir][k], sl[k].width, sl[k].peak },
                );
                // the beam is still a beam: most of its peak survives
                ok = ok and sl[k].peak >= 0.75;
            }
        }
        if (m.r_beam == 3.0) {
            // On the photon orbit the beam follows a grid line: it must stay
            // at r = 3 all the way round. Measured -0.16 (-0.12 refined).
            for (0..3) |k| ok = ok and @abs(off[0][k]) <= 0.3 and @abs(off[1][k]) <= 0.3;
        } else {
            // Off the photon orbit the beam cuts across the grid. Its
            // centroid runs OUTSIDE the geodesic by 0.8-1.9 cells at the
            // paper's resolution (the paper notes the same broadening and
            // "numerical diffusion ... whenever the beam is tilted with
            // respect to the grid"), and the offset shrinks when the grid is
            // refined: model 2 1.39 -> 0.97, model 3 1.85 -> 1.01 at pi/8.
            for (0..3) |k| ok = ok and @abs(off[0][k]) <= 2.2;
            ok = ok and @abs(off[1][0]) < @abs(off[0][0]) and @abs(off[1][1]) < @abs(off[0][1]);
        }
    }
    try expect(ok);
}
