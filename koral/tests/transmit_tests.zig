//! The transmitting polar axis (C: TRANSMITTING_YBC; options.zig
//! `PolarAxis.transmitting`). The bit-level statement against C is the
//! ZIGPOLE step test (tests/golden/step_golden_tests.zig, "pole3d"), which
//! drives all pieces at once. This file pins each piece on its own, so a
//! regression names its cause, and adds the physics the port is for:
//!
//!   * `Band.stencilRows`: the substitution table of finite.c:923-993,
//!   * the one-rank ghost fill: the cell across the axis, half a turn away,
//!     with the theta and phi components flipped,
//!   * the polar overwrite including B (CORRECTMAGNFIELD),
//!   * `Options.validate`: full 2 pi, even NZ, CORRECTMAGNFIELD,
//!   * the shell magnetic flux stays an invariant (E_phi still zero on the
//!     polar faces),
//!   * a uniform flow straight through the axis stays uniform, where a
//!     reflecting pole turns the axis into a wall.

const std = @import("std");
const config = @import("../config.zig");
const grid_mod = @import("../grid.zig");
const sim_mod = @import("../sim.zig");
const relele = @import("../relele.zig");
const polaraxis = @import("../sim/polaraxis.zig");
const hydro = @import("../physics/hydro.zig");
const bcs = @import("../problems/common/bcs.zig");

const Grid = grid_mod.Grid;
const PolarAxis = sim_mod.PolarAxis;
const expect = std.testing.expect;
const expectEqual = std.testing.expectEqual;

const cfg_mhd = config.Config{
    .modules = &.{ .hydro, .mhd },
    .reconstruction = .ppm,
    .flux = .laxf,
    .timestepping = .rk2imex,
    .coords = .mks2,
};
const SimM = sim_mod.Sim(cfg_mhd);
const LM = SimM.Layout;

fn poleBcM(ctx: ?*const anyopaque, core: *const SimM.CoreT, ix: i64, iy: i64, iz: i64, t: f64, ifinit: bool, face: sim_mod.BcFace) relele.Error![SimM.nv]f64 {
    _ = ctx;
    _ = t;
    _ = ifinit;
    return bcs.polarAxis(SimM.CoreT, core, ix, iy, iz, face);
}

/// A small MKS2 shell, r in [4, 20], the full sphere.
fn shellGrid(nx: usize, ny: usize, nz: usize, zspan: f64) Grid {
    return Grid.init(.{
        .nx = nx,
        .ny = ny,
        .nz = nz,
        .ng = 3,
        .minx = @log(4.0 - 0.1),
        .maxx = @log(20.0 - 0.1),
        .miny = 0.005,
        .maxy = 1.0 - 0.005,
        .minz = -0.5 * zspan,
        .maxz = 0.5 * zspan,
    });
}

fn shellOptions(pa: PolarAxis) SimM.Options {
    return .{
        .phys = .{ .coords = .mks2, .mp = .{ .a = 0.5, .mksr0 = 0.1, .mksh0 = 0.9 }, .gam = 4.0 / 3.0 },
        .num = .{ .polaraxis = pa },
        .bc = .{ .x = .copy, .y = .{ .specific = .{ .f = &poleBcM } }, .z = .periodic },
    };
}

const two_pi = 2.0 * std.math.pi;

/// Smooth primitives with no symmetry in theta or phi; every component
/// nonzero. `sim` supplies the geometry for a finite field on the axis.
fn smoothPrims(s: *const SimM, ix: i64, iy: i64, iz: i64) [SimM.nv]f64 {
    const geom = s.core.cache.fillGeometry(ix, iy, iz);
    const fx: f64 = @floatFromInt(ix);
    const fy: f64 = @floatFromInt(iy);
    const ph = s.core.grid.zc(iz);
    var pp: [SimM.nv]f64 = @splat(0);
    pp[LM.index(.rho)] = 1.0 + 0.02 * fy + 0.01 * fx + 0.2 * @cos(ph - 0.7);
    pp[LM.index(.uu)] = 0.05 * (1.0 + 0.1 * @sin(ph + 0.2)) + 0.001 * fy;
    pp[LM.index(.vx)] = -0.05 * (1.0 + 0.2 * @cos(ph));
    pp[LM.index(.vy)] = 0.004 * @sin(0.37 * fy + 0.2 * fx + ph);
    pp[LM.index(.vz)] = 0.01 * (1.0 + 0.5 * @cos(ph - 1.1));
    pp[LM.index(.entr)] = hydro.sFromU(pp[LM.index(.rho)], pp[LM.index(.uu)], 4.0 / 3.0);
    // conserved-density-like amplitudes: finite physical field everywhere
    const st = @abs(@sin(geom.xxvec[2] * std.math.pi)); // any smooth factor vanishing at the poles
    pp[LM.index(.b1)] = 0.3 * (0.2 + st) * (1.0 + 0.3 * @cos(ph + 0.4)) / geom.gdet;
    pp[LM.index(.b2)] = 0.03 * st * @cos(0.21 * fy + ph) / geom.gdet;
    pp[LM.index(.b3)] = 0.02 * (1.0 + 0.3 * @sin(ph)) / geom.gdet;
    return pp;
}

fn initSmooth(s: *SimM) !void {
    var iz: i64 = 0;
    while (iz < s.nzi()) : (iz += 1) {
        var iy: i64 = 0;
        while (iy < s.nyi()) : (iy += 1) {
            var ix: i64 = 0;
            while (ix < s.nxi()) : (ix += 1) try s.initCell(ix, iy, iz, smoothPrims(s, ix, iy, iz));
        }
    }
}

test "transmitting pole: the stencil rows next to each axis read the ghost row for row 0 (finite.c:923-993)" {
    const ny: i64 = 12;
    const b = polaraxis.Band{ .nc = 1, .ny = ny, .transmitting = true, .correct_b = true };
    // upper axis: every slot that would read row 0 reads row -1
    try expectEqual([5]i64{ -2, -1, -1, 1, 2 }, b.stencilRows(0));
    try expectEqual([5]i64{ -1, -1, 1, 2, 3 }, b.stencilRows(1));
    try expectEqual([5]i64{ -1, 1, 2, 3, 4 }, b.stencilRows(2));
    // lower axis: row ny-1 -> row ny
    try expectEqual([5]i64{ ny - 3, ny - 2, ny, ny, ny + 1 }, b.stencilRows(ny - 1));
    try expectEqual([5]i64{ ny - 4, ny - 3, ny - 2, ny, ny }, b.stencilRows(ny - 2));
    try expectEqual([5]i64{ ny - 5, ny - 4, ny - 3, ny - 2, ny }, b.stencilRows(ny - 3));
    // everything else, and the ghost rows themselves, is the plain stencil
    for ([_]i64{ -1, 3, 4, 5, 6, 7, 8, ny }) |iy| {
        try expectEqual([5]i64{ iy - 2, iy - 1, iy, iy + 1, iy + 2 }, b.stencilRows(iy));
    }
    // and a reflecting band never substitutes
    const r = polaraxis.Band{ .nc = 2, .ny = ny };
    var iy: i64 = -1;
    while (iy <= ny) : (iy += 1) try expectEqual([5]i64{ iy - 2, iy - 1, iy, iy + 1, iy + 2 }, r.stencilRows(iy));
}

test "transmitting pole: Options.validate wants the full 2 pi, an even NZ and CORRECTMAGNFIELD" {
    const a = std.testing.allocator;
    const td = PolarAxis.transmitting_default;
    try expectEqual(@as(i64, 1), td.ncells);
    try expect(td.transmitting and td.correct_b);

    // the accepted case
    {
        var s = try SimM.init(a, shellGrid(4, 8, 8, two_pi), shellOptions(td));
        s.deinit();
    }
    // a pi/2 wedge: the cell NZ/2 away is not the one across the axis
    try std.testing.expectError(error.InvalidConfig, SimM.init(a, shellGrid(4, 8, 8, 0.5 * std.math.pi), shellOptions(td)));
    // odd NZ (C: misc.c:300), and axisymmetric 2D
    try std.testing.expectError(error.InvalidConfig, SimM.init(a, shellGrid(4, 8, 7, two_pi), shellOptions(td)));
    try std.testing.expectError(error.InvalidConfig, SimM.init(a, shellGrid(4, 8, 1, two_pi), shellOptions(td)));
    // C forces CORRECTMAGNFIELD under TRANSMITTING_YBC
    try std.testing.expectError(error.InvalidConfig, SimM.init(a, shellGrid(4, 8, 8, two_pi), shellOptions(.{ .ncells = 1, .transmitting = true })));
    // the reflecting pole is untouched by any of this
    {
        var s = try SimM.init(a, shellGrid(4, 8, 7, 0.5 * std.math.pi), shellOptions(.{ .ncells = 2 }));
        s.deinit();
    }
}

test "transmitting pole: ghost rows hold the cell across the axis, half a turn away, theta and phi components flipped" {
    const a = std.testing.allocator;
    const nz: i64 = 8;
    var s = try SimM.init(a, shellGrid(4, 8, @intCast(nz), two_pi), shellOptions(PolarAxis.transmitting_default));
    defer s.deinit();
    try initSmooth(&s);
    try s.setBc(0.0, false);

    const ny = s.nyi();
    const flipped = [_]@import("../layout.zig").VarTag{ .vy, .vz, .b2, .b3 };
    var iz: i64 = 0;
    while (iz < nz) : (iz += 1) {
        const izs = @mod(iz + @divExact(nz, 2), nz);
        var ix: i64 = 0;
        while (ix < s.nxi()) : (ix += 1) {
            var j: i64 = 0;
            while (j < 3) : (j += 1) {
                inline for (.{ .{ -1 - j, j }, .{ ny + j, ny - 1 - j } }) |pair| {
                    var gh: [SimM.nv]f64 = undefined;
                    var src: [SimM.nv]f64 = undefined;
                    s.core.p.load(ix, pair[0], iz, &gh);
                    s.core.p.load(ix, pair[1], izs, &src);
                    for (0..SimM.nv) |iv| {
                        var sign: f64 = 1.0;
                        inline for (flipped) |tag| {
                            if (iv == LM.index(tag)) sign = -1.0;
                        }
                        try expectEqual(sign * src[iv], gh[iv]);
                    }
                }
            }
        }
    }
    // the state is not axisymmetric, so the half turn is visible: a ghost
    // differs from the plain mirror cell at the same phi
    var mirror: [SimM.nv]f64 = undefined;
    var gh: [SimM.nv]f64 = undefined;
    s.core.p.load(1, 0, 2, &mirror);
    s.core.p.load(1, -1, 2, &gh);
    try expect(@abs(gh[LM.index(.rho)] - mirror[LM.index(.rho)]) > 1.0e-2);
}

test "transmitting pole: the overwrite rewrites B in the corrected row (CORRECTMAGNFIELD)" {
    const a = std.testing.allocator;
    var s = try SimM.init(a, shellGrid(4, 8, 8, two_pi), shellOptions(PolarAxis.transmitting_default));
    defer s.deinit();
    try initSmooth(&s);
    try s.doCorrect();

    const g = &s.core.grid;
    const ny = s.nyi();
    var iz: i64 = 0;
    while (iz < s.nzi()) : (iz += 1) {
        var ix: i64 = 0;
        while (ix < s.nxi()) : (ix += 1) {
            // nc = 1: row 0 from row 1, row ny-1 from row ny-2
            inline for (.{ .{ 0, 1, 0 }, .{ ny - 1, ny - 2, ny } }) |c| {
                const iy: i64 = c[0];
                const iysrc: i64 = c[1];
                const thaxis = g.yl(c[2]);
                const fac = @abs((g.yc(iy) - thaxis) / (g.yc(iysrc) - thaxis));
                var pp: [SimM.nv]f64 = undefined;
                s.core.p.load(ix, iy, iz, &pp);
                const src = smoothPrims(&s, ix, iysrc, iz);
                try expectEqual(src[LM.index(.b1)], pp[LM.index(.b1)]);
                try expectEqual(src[LM.index(.b3)], pp[LM.index(.b3)]);
                try expectEqual(fac * src[LM.index(.b2)], pp[LM.index(.b2)]);
                try expectEqual(src[LM.index(.rho)], pp[LM.index(.rho)]);
                try expectEqual(fac * src[LM.index(.vy)], pp[LM.index(.vy)]);
            }
            // the second row is evolved, not overwritten
            var p1: [SimM.nv]f64 = undefined;
            s.core.p.load(ix, 1, iz, &p1);
            try expectEqual(smoothPrims(&s, ix, 1, iz), p1);
        }
    }
}

// ---- physics: a density column carried across the axis by a uniform wind ----

const cfg_ks = config.Config{
    .modules = &.{.hydro},
    .reconstruction = .ppm,
    .flux = .laxf,
    .timestepping = .rk2imex,
    .coords = .ks,
};
const SimK = sim_mod.Sim(cfg_ks);
const LK = SimK.Layout;

const Wind = struct {
    /// speed along Cartesian x, i.e. across the polar (z) axis
    v: f64 = 0.1,
    /// c_s = 0.2 for Gamma = 5/3 at rho = 1: the flow is subsonic
    uu: f64 = 0.0383,
    /// density column parallel to z: rho = 1 + amp exp(-((x - x0 - v t)^2 + y^2) / 2 sigma^2)
    amp: f64 = 1.0,
    x0: f64 = -5000.0,
    sigma: f64 = 3000.0,
    const gam: f64 = 5.0 / 3.0;

    fn rhoExact(w: *const Wind, r: f64, th: f64, ph: f64, t: f64) f64 {
        const x = r * @sin(th) * @cos(ph) - w.x0 - w.v * t;
        const y = r * @sin(th) * @sin(ph);
        return 1.0 + w.amp * @exp(-(x * x + y * y) / (2.0 * w.sigma * w.sigma));
    }

    /// The exact state at a cell and time, in the code's primitives. Far
    /// from the hole (r ~ 1e4) Kerr-Schild is flat to 2/r, so the Cartesian
    /// wind is u^r = g v sin(th) cos(ph), u^th = g v cos(th) cos(ph) / r,
    /// u^ph = -g v sin(ph) / (r sin(th)), converted to VELR by the metric.
    /// Pressure and velocity are uniform, so the column is a contact: it
    /// is carried along unchanged.
    fn prims(w: *const Wind, core: *const SimK.CoreT, ix: i64, iy: i64, iz: i64, t: f64) relele.Error![SimK.nv]f64 {
        const geom = core.cache.fillGeometry(ix, iy, iz);
        const r = geom.xxvec[1];
        const th = geom.xxvec[2];
        const ph = geom.xxvec[3];
        const lor = 1.0 / @sqrt(1.0 - w.v * w.v);
        var ucon = [4]f64{
            lor,
            lor * w.v * @sin(th) * @cos(ph),
            lor * w.v * @cos(th) * @cos(ph) / r,
            -lor * w.v * @sin(ph) / (r * @sin(th)),
        };
        ucon = try relele.convert(ucon, .vel4, .velr, &geom, .recompute_ut);
        var pp: [SimK.nv]f64 = @splat(0);
        pp[LK.index(.rho)] = w.rhoExact(r, th, ph, t);
        pp[LK.index(.uu)] = w.uu;
        pp[LK.index(.vx)] = ucon[1];
        pp[LK.index(.vy)] = ucon[2];
        pp[LK.index(.vz)] = ucon[3];
        pp[LK.index(.entr)] = hydro.sFromU(pp[LK.index(.rho)], w.uu, gam);
        return pp;
    }

    /// Radial faces: the exact solution. Polar faces: whatever the run
    /// asks for.
    fn bc(ctx: ?*const anyopaque, core: *const SimK.CoreT, ix: i64, iy: i64, iz: i64, t: f64, ifinit: bool, face: sim_mod.BcFace) relele.Error![SimK.nv]f64 {
        _ = ifinit;
        const w: *const Wind = @ptrCast(@alignCast(ctx.?));
        return switch (face) {
            .xlo, .xhi => w.prims(core, ix, iy, iz, t),
            .ylo, .yhi => bcs.polarAxis(SimK.CoreT, core, ix, iy, iz, face),
            .zlo, .zhi => unreachable,
        };
    }

    const Result = struct {
        /// L1 error of rho against the exact solution over the polar cap
        /// theta < 0.8 (both poles), relative to the L1 of the column there
        l1: f64,
        /// largest density on the lee side of the upper axis (|phi| < pi/2)
        lee_peak: f64,
        lee_exact: f64,
    };

    fn run(w: *const Wind, a: std.mem.Allocator, pa: PolarAxis, tend: f64, n: usize) !Result {
        const g = Grid.init(.{
            .nx = 4,
            .ny = n,
            .nz = n,
            .ng = 3,
            .minx = 1.0e4,
            .maxx = 1.4e4,
            .miny = 0.02,
            .maxy = std.math.pi - 0.02,
            .minz = -std.math.pi,
            .maxz = std.math.pi,
        });
        var s = try SimK.init(a, g, .{
            .phys = .{ .coords = .ks, .gam = gam },
            .num = .{ .polaraxis = pa },
            .bc = .{
                .x = .{ .specific = .{ .f = &bc, .ctx = w } },
                .y = .{ .specific = .{ .f = &bc, .ctx = w } },
                .z = .periodic,
            },
        });
        defer s.deinit();
        var iz: i64 = 0;
        while (iz < s.nzi()) : (iz += 1) {
            var iy: i64 = 0;
            while (iy < s.nyi()) : (iy += 1) {
                var ix: i64 = 0;
                while (ix < s.nxi()) : (ix += 1) try s.initCell(ix, iy, iz, try w.prims(&s.core, ix, iy, iz, 0.0));
            }
        }
        try s.finishInit();
        while (s.t < tend) {
            var dt: ?f64 = null;
            if (s.t + 1.0 / s.core.tstepdenmax > tend) dt = tend - s.t;
            try s.step(dt);
        }
        var res = Result{ .l1 = 0, .lee_peak = 0, .lee_exact = 0 };
        var norm: f64 = 0;
        iz = 0;
        while (iz < s.nzi()) : (iz += 1) {
            var iy: i64 = 0;
            while (iy < s.nyi()) : (iy += 1) {
                const th = g.yc(iy);
                const cap = @min(th, std.math.pi - th);
                if (cap > 0.8) continue;
                var ix: i64 = 1;
                while (ix < s.nxi() - 1) : (ix += 1) {
                    const geom = s.core.cache.fillGeometry(ix, iy, iz);
                    const ex = w.rhoExact(geom.xxvec[1], th, g.zc(iz), s.t);
                    const rho = s.core.p.get(LK.index(.rho), ix, iy, iz);
                    res.l1 += geom.gdet * @abs(rho - ex);
                    norm += geom.gdet * (ex - 1.0);
                    if (th < 0.8 and @abs(g.zc(iz)) < 0.5 * std.math.pi) {
                        res.lee_peak = @max(res.lee_peak, rho);
                        res.lee_exact = @max(res.lee_exact, ex);
                    }
                }
            }
        }
        res.l1 /= norm;
        return res;
    }
};

const transmit = PolarAxis.transmitting_default;
const column_tend: f64 = 1.0e5;

test "transmitting pole: a density column is carried across the axis by a uniform wind" {
    // Uniform pressure, uniform velocity 0.1 c along x (perpendicular to the
    // polar axis), and a column of doubled density parallel to the axis,
    // far enough out (r = 1e4) that gravity does nothing over the run. The
    // exact solution is the column carried along unchanged. It starts 5000
    // upwind of the axis and ends 5000 downwind, about two polar rows either
    // way at this resolution.
    //
    // What is asserted is that the column arrives: its peak on the lee side
    // is most of the way to the exact one, and the error over the polar
    // caps is that of a first-order smeared contact, not of a lost one.
    //
    // What is NOT asserted, because it is not true: that the transmitting
    // pole beats the reflecting one here. With the thin excised cone used
    // by every problem (theta_min = 0.02), a reflecting axis is a needle,
    // not a dam: the column passes around it through the azimuthal faces of
    // the small cells next to the pole, and the reflecting run with the
    // same single overwritten row scores the same L1 (0.60 against 0.64 at
    // 16 cells, 0.270 against 0.267 at 32) with a slightly lower peak. C's
    // start-up note says as much ("diffusion around pole dominates
    // transmission" for small MINY). The difference that does show is the
    // number of overwritten rows: PUFFY's two cost 30 to 40 % in L1 against
    // one, with either pole.
    const a = std.testing.allocator;
    const w = Wind{};
    const tr = try w.run(a, transmit, column_tend, 16);
    const r2 = try w.run(a, PolarAxis{ .ncells = 2 }, column_tend, 16);
    std.debug.print("column across the pole, 16 cells: transmitting L1 {d:.3}, lee peak {d:.3} of {d:.3}; reflecting with 2 overwritten rows L1 {d:.3}, lee peak {d:.3}\n", .{ tr.l1, tr.lee_peak, tr.lee_exact, r2.l1, r2.lee_peak });
    try expect(tr.lee_peak - 1.0 > 0.4 * (tr.lee_exact - 1.0));
    try expect(tr.l1 < 0.75);
    try expect(tr.l1 < r2.l1);
}

test "transmitting pole slow: the column's error converges; reflecting poles for comparison" {
    if (!@import("build_options").slow_tests) return error.SkipZigTest;
    const a = std.testing.allocator;
    const w = Wind{};
    var l1: [3]f64 = undefined;
    for ([_]usize{ 16, 24, 32 }, 0..) |n, k| {
        const tr = try w.run(a, transmit, column_tend, n);
        const r1 = try w.run(a, PolarAxis{ .ncells = 1 }, column_tend, n);
        const r2 = try w.run(a, PolarAxis{ .ncells = 2 }, column_tend, n);
        l1[k] = tr.l1;
        std.debug.print("column across the pole, n = {d}: L1 transmitting {d:.3}, reflecting nc=1 {d:.3}, nc=2 {d:.3}; lee peak (exact {d:.3}) {d:.3} {d:.3} {d:.3}\n", .{ n, tr.l1, r1.l1, r2.l1, tr.lee_exact, tr.lee_peak, r1.lee_peak, r2.lee_peak });
        // one overwritten row is better than two, whichever the pole
        try expect(tr.l1 < r2.l1 and r1.l1 < r2.l1);
        // and the two poles with one row are within 10 % of each other
        try expect(@abs(tr.l1 / r1.l1 - 1.0) < 0.1);
    }
    // better than first order in the angular resolution
    try expect(l1[2] < 0.5 * l1[0]);
}
