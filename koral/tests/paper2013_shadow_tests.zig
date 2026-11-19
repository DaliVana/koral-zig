//! Reproduction gate for Sadowski et al. (2013), section 4.5 and Figure 10:
//! the shadow cast by an optically thick blob in a beam of light.
//!
//! A cold dense blob sits in a hot thin ambient medium at constant pressure,
//!
//!     rho = rho0 + (rho1 - rho0) exp(-sqrt(x^2 + y^2) / w^2),
//!     rho0 = 1e-4, rho1 = 1e3, w = 0.22,     T = T0 rho0 / rho,
//!
//! with pure absorption, alpha = kappa rho = rho per unit length. (The paper
//! prints T = T0 rho / rho0; constant pressure, which it states, needs the
//! inverse.) Radiation starts in LTE and at rest. Domain [-1, 3] x [-1, 1]
//! (read off the axes of Figure 10; the text gives only the 100 x 50
//! cells), periodic in y, outflow on the right, and on the left a beam
//! E_L = 4 sigma T_L^4, F^x = 0.99999 E_L, T_L = 100 T0. Run to t = 10, two
//! and a half light crossings.
//!
//! The paper shows the result as a picture. There is an exact answer to
//! compare with: the blob is cold (its own emission is 1e-28 of the beam)
//! and the beam is collimated, so behind the blob
//!
//!     E(x, y) = E_L exp(-tau(y)),     tau(y) = integral of rho dx,
//!
//! along each horizontal ray. The M1 closure with F ~ E must reproduce that
//! profile, up to the numerical diffusion of the shadow edge. Gates: the
//! L1 distance from exp(-tau) across the shadow at x = 2, the position of
//! the half-light edge, the depth of the umbra, the unobscured beam (E = E_L,
//! F / E = 1) and stationarity. Measured at 100 x 50: L1 5.1e-4, edge at
//! y = 0.2971 against 0.2968 exact (cell size 0.04), umbra 6e-9 E_L.
//!
//! Choices the paper leaves open, none of which enters the dimensionless
//! result because radiation is dynamically and thermally negligible over
//! the run: unit mass 1e-4 Msun and T0 = 1e11 K (E_L = 1.4e-9, and the blob
//! core at 1e4 K, clear of the 100 K temperature floor), Gamma = 5/3. The
//! Eddington-closure run of Figure 10 (top) is not reproduced; this code
//! has M1 only.
//!
//! WHAT IT TOOK TO RUN, each found by a failure:
//!  * Linear reconstruction (minmod theta = 1.5). With production PPM the
//!    front where E falls by 28 orders of magnitude in two cells makes
//!    |F| > E in the blob core; the radiation inversion then caps cells at
//!    gamma_max, and from t = 3.3 the energy there runs away (1e136 E_L by
//!    t = 10).
//!  * gamma_max,rad = 1000: the beam itself has a radiation Lorentz factor
//!    of 158, above C's default ceiling of 100 (which inflates the injected
//!    energy by 8 %).
//!  * The implicit and radiative fixups on (off in the validated PUFFY
//!    build, on in koral_lite_puffy). Without them the first failed solve,
//!    at t = 0.93 when the front reaches cells of optical depth 20, spreads
//!    over the grid within 30 steps and ends in NanInFlux.
//!
//! NOT REPRODUCED: the two-beam variant (Figure 11). Two beams crossing at
//! (0.93, -0.37) and its mirror image, with |F| / E = 0.99999, are unstable
//! here: the radiation energy on the axis where the beam edges first meet
//! grows by a factor 10-40 per step from t = 0.25, with a mirror boundary
//! and equally on the full symmetric domain without one, for gamma_max,rad
//! 100 or 1000 and flux factors down to 0.999. At flux factor 0.99 the run
//! is stable but no longer the paper's problem: the merged field has
//! F / E = 0.7 and finite M1 pressure, and drains back out through the
//! unlit part of the left boundary instead of casting a shadow. The 2013
//! code ran this case; why this one does not is open.

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
const invert_rad = @import("../solve/invert_rad.zig");
const tubes = @import("../testing/tubes.zig");

const Grid = grid_mod.Grid;
const expect = std.testing.expect;

const cfg = config.Config{
    .modules = &.{ .hydro, .radiation },
    .reconstruction = .linear,
    .flux = .laxf,
    .timestepping = .rk2imex,
    .coords = .mink,
};
const SimT = sim_mod.Sim(cfg);
const CoreT = SimT.CoreT;
const L = SimT.Layout;
const NV = SimT.nv;

const gam: f64 = 5.0 / 3.0;
const rho_amb: f64 = 1.0e-4;
const rho_blob: f64 = 1.0e3;
const blob_w: f64 = 0.22;
const t_amb: f64 = 1.0e11;
/// Unit mass in Msun: sets the radiation constant in code units.
const unit_mass: f64 = 1.0e-4;
const t_beam: f64 = 100.0 * t_amb;
const flux_factor: f64 = 0.99999;

fn rhoAt(x: f64, y: f64) f64 {
    return rho_amb + (rho_blob - rho_amb) * @exp(-@sqrt(x * x + y * y) / (blob_w * blob_w));
}

/// Optical depth of the horizontal ray at height y from x0 to x1
/// (alpha = rho), by Simpson's rule on a grid far finer than the blob.
fn tau(y: f64, x0: f64, x1: f64) f64 {
    const n: usize = 8000;
    const h = (x1 - x0) / @as(f64, @floatFromInt(n));
    var sum: f64 = rhoAt(x0, y) + rhoAt(x1, y);
    for (1..n) |i| {
        const w: f64 = if (i % 2 == 1) 4.0 else 2.0;
        sum += w * rhoAt(x0 + h * @as(f64, @floatFromInt(i)), y);
    }
    return sum * h / 3.0;
}

const Setup = struct {
    consts: thermo.Consts,
    e_beam: f64,

    /// Ambient-pressure gas at rest in LTE at (x, y).
    fn gas(st: *const Setup, x: f64, y: f64) [NV]f64 {
        const rho = rhoAt(x, y);
        const temp = t_amb * rho_amb / rho;
        const uint = thermo.uFromTrho(&st.consts, temp, rho, gam);
        var pp: [NV]f64 = @splat(0);
        pp[L.index(.rho)] = rho;
        pp[L.index(.uu)] = uint;
        pp[L.index(.entr)] = hydro.sFromU(rho, uint, gam);
        pp[L.index(.ee)] = st.consts.lteEfromT(temp);
        return pp;
    }

    /// Ambient gas carrying the beam: lab-frame E = e_beam, F = 0.99999 E
    /// along +x. For M1 the radiation rest frame then moves with
    /// beta = (2 - sqrt(4 - 3 phi^2)) / phi and E = Ebar (4 gamma^2 - 1) / 3.
    fn beam(st: *const Setup, x: f64, y: f64) [NV]f64 {
        var pp = st.gas(x, y);
        const beta = (2.0 - @sqrt(4.0 - 3.0 * flux_factor * flux_factor)) / flux_factor;
        const lor2 = 1.0 / (1.0 - beta * beta);
        const gbeta = @sqrt(lor2) * beta;
        pp[L.index(.ee)] = 3.0 * st.e_beam / (4.0 * lor2 - 1.0);
        pp[L.index(.fx)] = gbeta;
        return pp;
    }
};

fn load(core: *const CoreT, ix: i64, iy: i64) [NV]f64 {
    var pp: [NV]f64 = undefined;
    core.p.load(ix, iy, 0, &pp);
    return pp;
}

fn bc(ctx: ?*const anyopaque, core: *const CoreT, ix: i64, iy: i64, iz: i64, t: f64, ifinit: bool, face: sim_mod.BcFace) relele.Error![NV]f64 {
    _ = iz;
    _ = t;
    _ = ifinit;
    const st: *const Setup = @ptrCast(@alignCast(ctx.?));
    const g = &core.grid;
    const x = g.xc(ix);
    const y = g.yc(iy);
    const nx: i64 = @intCast(g.nx);
    const ny: i64 = @intCast(g.ny);
    return switch (face) {
        .xlo => st.beam(x, y), // the light source
        .xhi => load(core, nx - 1, std.math.clamp(iy, 0, ny - 1)), // outflow
        else => unreachable, // y is periodic
    };
}

/// Lab-frame radiation moments of a cell from its conserved variables
/// (Minkowski, sqrt(-g) = 1): E = -R^t_t, F^i = R^t_i.
const Moments = struct { e: f64, fx: f64, fy: f64 };

fn moments(s: *const SimT, ix: i64, iy: i64) Moments {
    return .{
        .e = -s.core.u.get(L.index(.ee), ix, iy, 0),
        .fx = s.core.u.get(L.index(.fx), ix, iy, 0),
        .fy = s.core.u.get(L.index(.fy), ix, iy, 0),
    };
}

const Run = struct {
    s: SimT,
    st: *Setup,
    a: std.mem.Allocator,

    fn deinit(r: *Run) void {
        r.s.deinit();
        r.a.destroy(r.st);
    }
};

fn start(a: std.mem.Allocator, nx: usize, ny: usize) !Run {
    const units = units_mod.Units.init(unit_mass);
    const opac = radforce.Params.grey(units, tubes.comp, 1.0, 0.0);

    const st = try a.create(Setup);
    errdefer a.destroy(st);
    st.* = .{ .consts = opac.consts, .e_beam = opac.consts.lteEfromT(t_beam) };

    const g = Grid.init(.{
        .nx = nx,
        .ny = ny,
        .ng = 3,
        .minx = -1.0,
        .maxx = 3.0,
        .miny = -1.0,
        .maxy = 1.0,
    });
    var rad = invert_rad.RadParams.cdefault;
    // the blob core is at T0 / 1e7: E / rho ~ 1e-48 there
    rad.eerhoratiomin = 1.0e-60;
    rad.eeuuratiomin = 1.0e-60;
    // F / E = 0.99999 is a radiation Lorentz factor of 158
    rad.gammamaxrad = 1000.0;

    var s = try SimT.init(a, g, .{
        .phys = .{ .coords = .mink, .gam = gam, .rad = rad, .opac = opac },
        // see the header for each of these
        .num = .{ .fixups = .{ .radimp = true, .u2prad = true }, .minmod_theta = 1.5 },
        .bc = .{
            .x = .{ .specific = .{ .f = &bc, .ctx = st } },
            .y = .periodic,
        },
    });
    errdefer s.deinit();

    var iy: i64 = 0;
    while (iy < s.nyi()) : (iy += 1) {
        var ix: i64 = 0;
        while (ix < s.nxi()) : (ix += 1) try s.initCell(ix, iy, 0, st.gas(g.xc(ix), g.yc(iy)));
    }
    try s.finishInit();
    return .{ .s = s, .st = st, .a = a };
}

fn advance(s: *SimT, tend: f64) !void {
    while (s.t < tend) {
        var dt: ?f64 = null;
        if (s.t + 1.0 / s.core.tstepdenmax > tend) dt = tend - s.t;
        try s.step(dt);
    }
}

fn columnAt(s: *const SimT, x: f64) i64 {
    return @intFromFloat(@floor((x - s.core.grid.minx) / s.core.grid.dx));
}

// ---- single beam -------------------------------------------------------------

const SingleResult = struct {
    /// mean over the column x = 2 of |E / E_L - exp(-tau)|
    l1: f64,
    /// |y| where E = E_L / 2 at x = 2, numerical and exact
    edge: f64,
    edge_exact: f64,
    /// E / E_L on the axis at x = 2 (deepest shadow)
    umbra: f64,
    /// worst |E / E_L - 1| and |F^x / E - 1| at |y| > 0.7
    beam_e: f64,
    beam_f: f64,
    /// largest change of E / E_L anywhere between t = 9 and t = 10
    drift: f64,
};

fn halfLight(vals: []const f64, ys: []const f64) f64 {
    // upper half, scanning outward from the axis
    const mid = vals.len / 2;
    var j = mid;
    while (j + 1 < vals.len) : (j += 1) {
        if (vals[j] < 0.5 and vals[j + 1] >= 0.5) {
            const fr = (0.5 - vals[j]) / (vals[j + 1] - vals[j]);
            return ys[j] + fr * (ys[j + 1] - ys[j]);
        }
    }
    return std.math.nan(f64);
}

fn runSingle(nx: usize, ny: usize) !SingleResult {
    const a = std.testing.allocator;
    var r = try start(a, nx, ny);
    defer r.deinit();
    const s = &r.s;
    const el = r.st.e_beam;
    const g = &s.core.grid;

    try advance(s, 9.0);
    const before = try a.alloc(f64, nx * ny);
    defer a.free(before);
    var iy: i64 = 0;
    while (iy < s.nyi()) : (iy += 1) {
        var ix: i64 = 0;
        while (ix < s.nxi()) : (ix += 1) before[@intCast(iy * s.nxi() + ix)] = moments(s, ix, iy).e / el;
    }
    try advance(s, 10.0);

    var res: SingleResult = .{ .l1 = 0, .edge = 0, .edge_exact = 0, .umbra = 0, .beam_e = 0, .beam_f = 0, .drift = 0 };
    iy = 0;
    while (iy < s.nyi()) : (iy += 1) {
        var ix: i64 = 0;
        while (ix < s.nxi()) : (ix += 1) {
            res.drift = @max(res.drift, @abs(moments(s, ix, iy).e / el - before[@intCast(iy * s.nxi() + ix)]));
        }
    }

    const col = columnAt(s, 2.0);
    const xcol = g.xc(col);
    const num = try a.alloc(f64, ny);
    defer a.free(num);
    const exact = try a.alloc(f64, ny);
    defer a.free(exact);
    const ys = try a.alloc(f64, ny);
    defer a.free(ys);
    iy = 0;
    while (iy < s.nyi()) : (iy += 1) {
        const j: usize = @intCast(iy);
        ys[j] = g.yc(iy);
        num[j] = moments(s, col, iy).e / el;
        exact[j] = @exp(-tau(ys[j], g.minx, xcol));
        res.l1 += @abs(num[j] - exact[j]) / @as(f64, @floatFromInt(ny));
        if (@abs(ys[j]) > 0.7) {
            // every column: the beam passes above and below the blob untouched
            var ix: i64 = 0;
            while (ix < s.nxi()) : (ix += 1) {
                const m = moments(s, ix, iy);
                res.beam_e = @max(res.beam_e, @abs(m.e / el - 1.0));
                res.beam_f = @max(res.beam_f, @abs(m.fx / m.e - 1.0));
            }
        }
    }
    // figure data: the map, and the column x = 2 against exp(-tau)
    {
        var map = figdata.Table.init("k13_shadow_map");
        defer map.finish();
        map.meta("source", "Sadowski et al. 2013, section 4.5, Figure 10 (M1 panel): single-beam shadow at t = 10", .{});
        map.meta("columns", "x y E/E_L Fx/E Fy/E rho   (lab-frame moments)", .{});
        map.meta("shape", "{d} {d}", .{ ny, nx });
        var jy: i64 = 0;
        while (jy < s.nyi()) : (jy += 1) {
            var jx: i64 = 0;
            while (jx < s.nxi()) : (jx += 1) {
                const m = moments(s, jx, jy);
                map.row(&.{ g.xc(jx), g.yc(jy), m.e / el, m.fx / m.e, m.fy / m.e, s.core.p.get(L.index(.rho), jx, jy, 0) });
            }
        }
        var cut = figdata.Table.init("k13_shadow_cut");
        defer cut.finish();
        cut.meta("source", "single-beam shadow, column x = {d}: transmitted light against the exact exp(-tau)", .{xcol});
        cut.meta("columns", "y E/E_L exp(-tau)", .{});
        for (0..ny) |j| cut.row(&.{ ys[j], num[j], exact[j] });
    }
    res.edge = halfLight(num, ys);
    res.edge_exact = halfLight(exact, ys);
    res.umbra = @max(num[ny / 2], num[ny / 2 - 1]);
    std.debug.print(
        "paper 2013 shadow {d}x{d}: L1(E/E_L - exp(-tau)) at x=2: {e:.3}; half-light edge y={d:.4} (exact {d:.4}, dy {d:.3}); umbra E/E_L {e:.2}; beam |dE| {e:.2} |dF/E| {e:.2}; drift t=9..10 {e:.2}\n",
        .{ nx, ny, res.l1, res.edge, res.edge_exact, g.dy, res.umbra, res.beam_e, res.beam_f, res.drift },
    );
    return res;
}

test "Sadowski 2013 section 4.5: single-beam shadow matches exp(-tau) (Figure 10)" {
    if (!build_options.slow_tests) return error.SkipZigTest;
    const r = try runSingle(100, 50);
    try expect(r.l1 <= 3.0e-2);
    try expect(@abs(r.edge - r.edge_exact) <= 0.04); // one cell
    try expect(r.umbra <= 1.0e-3);
    try expect(r.beam_e <= 2.0e-2 and r.beam_f <= 1.0e-3);
    try expect(r.drift <= 1.0e-2);
}
