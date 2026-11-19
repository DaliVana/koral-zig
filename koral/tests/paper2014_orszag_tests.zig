//! Reproduction gate for Sadowski et al. (2014, MNRAS 439, 503), section 4.3
//! and Figure 6: the Orszag-Tang vortex, in the non-relativistic regime of a
//! relativistic code (velocity factor C = 100).
//!
//! Setup, following the paper (which follows Stone et al. 2008): uniform
//! rho = 25 / (36 pi) and p = 5 / (12 pi C^2), velocity
//! u = (-sin 2 pi y, sin 2 pi x, 0) / C, magnetic field from the potential
//! A^z = (B0 / 4 pi) cos 4 pi x + (B0 / 2 pi) cos 2 pi y, B0 = 1 / sqrt(4 pi)
//! / C, on the periodic unit square, run to t = 50 (t = 0.5 unscaled).
//! Linear reconstruction, minmod theta = 1.5, Lax-Friedrichs, RK2.
//!
//! The paper runs 640 x 640 cells and shows the density map and a cut along
//! y = 0.75. This test runs 256 x 256 (a minute on four threads; 640^2 costs
//! 16 times that) and dumps the same two things for tools/paperfig.py.
//!
//! The paper compares by eye with Stone et al. and Gammie et al. (2003).
//! What can be gated without their data are the exact properties of the
//! problem:
//!   * the initial data are invariant under the half turn
//!     (x, y) -> (1 - x, 1 - y), and the equations preserve that: rho must
//!     keep the symmetry to roundoff-amplified level,
//!   * mass, and the total x and y momentum (zero), are conserved on the
//!     periodic box,
//!   * div B stays at machine zero in the interior (flux-CT),
//!   * shocks have formed by t = 0.5: the density contrast is that of the
//!     published solution (max / min about 5).

const std = @import("std");
const figdata = @import("../testing/figdata.zig");
const build_options = @import("build_options");
const config = @import("../config.zig");
const grid_mod = @import("../grid.zig");
const sim_mod = @import("../sim.zig");
const hydro = @import("../physics/hydro.zig");

const Grid = grid_mod.Grid;
const expect = std.testing.expect;

const cfg = config.Config{
    .modules = &.{ .hydro, .mhd },
    .reconstruction = .linear,
    .flux = .laxf,
    .timestepping = .rk2imex,
    .coords = .mink,
};
const SimT = sim_mod.Sim(cfg);
const L = SimT.Layout;
const NV = SimT.nv;

const gam: f64 = 5.0 / 3.0;
const cscale: f64 = 100.0;
const tp: f64 = 2.0 * std.math.pi;

test "Sadowski 2014 section 4.3: Orszag-Tang vortex keeps its symmetry, mass and div B (Figure 6)" {
    if (!build_options.slow_tests) return error.SkipZigTest;
    const a = std.testing.allocator;
    const n: usize = 256;
    const g = Grid.init(.{ .nx = n, .ny = n, .ng = 2, .minx = 0, .maxx = 1, .miny = 0, .maxy = 1 });
    var s = try SimT.init(a, g, .{
        .phys = .{ .coords = .mink, .gam = gam },
        .num = .{ .minmod_theta = 1.5 },
        .bc = .{ .x = .periodic, .y = .periodic },
        // bit-identical to the serial path by construction
        .parallel = .{ .nthreads = 4 },
    });
    defer s.deinit();

    const rho0 = 25.0 / (36.0 * std.math.pi);
    const p0 = 5.0 / (12.0 * std.math.pi) / (cscale * cscale);
    const b0 = 1.0 / @sqrt(4.0 * std.math.pi) / cscale;
    var iy: i64 = 0;
    while (iy < s.nyi()) : (iy += 1) {
        var ix: i64 = 0;
        while (ix < s.nxi()) : (ix += 1) {
            const x = g.xc(ix);
            const y = g.yc(iy);
            const vx = -@sin(tp * y) / cscale;
            const vy = @sin(tp * x) / cscale;
            const lor = 1.0 / @sqrt(1.0 - vx * vx - vy * vy);
            var pp: [NV]f64 = @splat(0);
            pp[L.index(.rho)] = rho0;
            pp[L.index(.uu)] = p0 / (gam - 1.0);
            pp[L.index(.entr)] = hydro.sFromU(rho0, p0 / (gam - 1.0), gam);
            pp[L.index(.vx)] = lor * vx;
            pp[L.index(.vy)] = lor * vy;
            // the vector potential rides in the B slots until calcBfromA
            pp[L.index(.b3)] = b0 / (2.0 * tp) * @cos(2.0 * tp * x) + b0 / tp * @cos(tp * y);
            try s.initCell(ix, iy, 0, pp);
        }
    }
    try s.setBc(0.0, true);
    try s.calcBfromA(true);
    try s.setBc(0.0, true);
    s.initTimestepGuess();

    const Totals = struct { mass: f64, px: f64, py: f64, pscale: f64 };
    const totals = struct {
        fn of(sim: *const SimT) Totals {
            var t = Totals{ .mass = 0, .px = 0, .py = 0, .pscale = 0 };
            var jy: i64 = 0;
            while (jy < sim.nyi()) : (jy += 1) {
                var jx: i64 = 0;
                while (jx < sim.nxi()) : (jx += 1) {
                    t.mass += sim.core.u.get(L.index(.rho), jx, jy, 0);
                    t.px += sim.core.u.get(L.index(.vx), jx, jy, 0);
                    t.py += sim.core.u.get(L.index(.vy), jx, jy, 0);
                    t.pscale += @abs(sim.core.u.get(L.index(.vx), jx, jy, 0));
                }
            }
            return t;
        }
    };
    const before = totals.of(&s);

    const tend = 0.5 * cscale;
    while (s.t < tend) {
        var dt: ?f64 = null;
        if (s.t + 1.0 / s.core.tstepdenmax > tend) dt = tend - s.t;
        try s.step(dt);
    }
    const after = totals.of(&s);

    // half-turn symmetry, density extrema, interior div B
    const ni: i64 = @intCast(n);
    var asym: f64 = 0;
    var rmin: f64 = 1e30;
    var rmax: f64 = 0;
    var divb: f64 = 0;
    var bmax: f64 = 0;
    iy = 0;
    while (iy < ni) : (iy += 1) {
        var ix: i64 = 0;
        while (ix < ni) : (ix += 1) {
            const r = s.core.p.get(L.index(.rho), ix, iy, 0);
            const r_mirror = s.core.p.get(L.index(.rho), ni - 1 - ix, ni - 1 - iy, 0);
            asym = @max(asym, @abs(r - r_mirror) / rho0);
            rmin = @min(rmin, r);
            rmax = @max(rmax, r);
            for (0..3) |k| bmax = @max(bmax, @abs(s.core.p.get(L.index(.b1) + k, ix, iy, 0)));
            if (ix >= 2 and ix < ni - 2 and iy >= 2 and iy < ni - 2) divb = @max(divb, @abs(s.calcDivB(ix, iy, 0)));
        }
    }
    const dmass = @abs(after.mass / before.mass - 1.0);
    const dmom = @max(@abs(after.px), @abs(after.py)) / after.pscale;
    std.debug.print(
        "paper 2014 Orszag-Tang {d}^2, t = 0.5: half-turn asymmetry {e:.2}, mass drift {e:.2}, net momentum / sum|p| {e:.2}, interior divB dx/B {e:.2}, rho max/min {d:.3}\n",
        .{ n, asym, dmass, dmom, divb * g.dx / bmax, rmax / rmin },
    );

    // figure data: the density map and the paper's cut at y = 0.75, in the
    // unscaled units of Stone et al. / Gammie et al.
    {
        var map = figdata.Table.init("k14_orszag_map");
        defer map.finish();
        map.meta("source", "Sadowski et al. 2014, section 4.3, Figure 6 (top): Orszag-Tang at t = 0.5, {d}^2 cells (paper: 640^2)", .{n});
        map.meta("columns", "x y rho magnetic_pressure(B^2/2, unscaled)", .{});
        map.meta("shape", "{d} {d}", .{ n, n });
        iy = 0;
        while (iy < ni) : (iy += 1) {
            var ix: i64 = 0;
            while (ix < ni) : (ix += 1) {
                var bsq: f64 = 0;
                for (0..3) |k| {
                    const b = s.core.p.get(L.index(.b1) + k, ix, iy, 0) * cscale;
                    bsq += b * b;
                }
                map.row(&.{ g.xc(ix), g.yc(iy), s.core.p.get(L.index(.rho), ix, iy, 0), 0.5 * bsq });
            }
        }
        var cut = figdata.Table.init("k14_orszag_cut");
        defer cut.finish();
        const jcut: i64 = @intFromFloat(@floor(0.75 / g.dy));
        cut.meta("source", "Sadowski et al. 2014, Figure 6 (bottom): density along y = {d}", .{g.yc(jcut)});
        cut.meta("columns", "x 4pi*rho", .{});
        var ix: i64 = 0;
        while (ix < ni) : (ix += 1) cut.row(&.{ g.xc(ix), 4.0 * std.math.pi * s.core.p.get(L.index(.rho), ix, jcut, 0) });
    }

    try expect(asym <= 1.0e-6);
    try expect(dmass <= 1.0e-12);
    try expect(dmom <= 1.0e-9);
    try expect(divb * g.dx / bmax <= 1.0e-11);
    try expect(rmax / rmin >= 3.0 and rmax / rmin <= 8.0);
}
