//! M10 theory gates for the polar-axis special treatment
//! (CORRECT_POLARAXIS, NCCORRECTPOLAR = 2; PUFFY define.h:107-109):
//!
//!   correct_polaraxis (finite.c:5525): most-polar rows overwritten from
//!                                        row nc, θ-velocities scaled, B
//!                                        untouched (no CORRECTMAGNFIELD)
//!   calc_u2p (u2p.c:57/80/92): polar cells get u2p_solver_Bonly
//!                                        and skip both floor checks
//!   op_implicit (finite.c:1427): polar cells skipped entirely
//!   cell_fixup (finite.c:5042): polar cells never fixup targets
//!   adjust_fluxcttoth_emfs (magn.c:388): corner EMFs zeroed on the polar
//!                                        faces; shell magnetic flux conserved
//!
//! Pure-Zig pins of the transcription; the C-diff of the whole pipeline on
//! a spherical grid arrives with the PUFFY t=0 + step goldens (M11/M13), and
//! the physics anchor (2D Bondi/Michel driven through the band: stationary,
//! θ-uniform to roundoff) is ks_evolution_tests.zig.

const std = @import("std");
const config = @import("../config.zig");
const layout = @import("../layout.zig");
const grid_mod = @import("../grid.zig");
const sim_mod = @import("../sim.zig");
const polaraxis = @import("../sim/polaraxis.zig");
const p2u_mod = @import("../p2u.zig");
const hydro = @import("../physics/hydro.zig");
const radforce = @import("../physics/radforce.zig");
const invert_rad = @import("../solve/invert_rad.zig");
const precompute = @import("../metric/precompute.zig");
const relele = @import("../relele.zig");
const mhd = @import("../physics/bfield.zig");
const implicit = @import("../solve/implicit.zig");

const cfg = config.puffy;
const SimT = sim_mod.Sim(cfg);
const L = SimT.Layout;
const NV = SimT.nv;
const Grid = grid_mod.Grid;

const gam: f64 = 5.0 / 3.0;

/// small MKS2 wedge away from the horizon: r ∈ [4, 20], full θ
fn testGrid() Grid {
    return Grid.init(.{
        .nx = 4,
        .ny = 8,
        .ng = 3,
        .minx = @log(4.0 - 0.1),
        .maxx = @log(20.0 - 0.1),
        .miny = 0.001,
        .maxy = 1.0 - 0.001,
    });
}

fn testOptions() SimT.Options {
    return .{
        .phys = .{ .coords = .mks2, .gam = gam, .rad = invert_rad.RadParams.puffy, .opac = radforce.Params.puffy(), .implicit = implicit.ImplicitParams.puffy },
        .num = .{ .polaraxis = .{ .ncells = 2 } },
        .bc = .{ .y = .copy },
    };
}

/// smooth iy-dependent primitives (physical, mildly magnetized, radiative)
fn ppAt(ix: i64, iy: i64) [NV]f64 {
    const fy: f64 = @floatFromInt(iy);
    const fx: f64 = @floatFromInt(ix);
    var pp: [NV]f64 = @splat(0);
    pp[L.index(.rho)] = 1.0 + 0.1 * fy + 0.01 * fx;
    pp[L.index(.uu)] = 0.1 + 0.01 * fy;
    pp[L.index(.vx)] = 0.01;
    pp[L.index(.vy)] = 0.002 * (fy + 1.0);
    pp[L.index(.vz)] = 0.003;
    pp[L.index(.entr)] = hydro.sFromU(pp[L.index(.rho)], pp[L.index(.uu)], gam);
    pp[L.index(.b1)] = 1.0e-3 + 1.0e-4 * fy;
    pp[L.index(.b2)] = 2.0e-4;
    pp[L.index(.b3)] = 3.0e-4;
    pp[L.index(.ee)] = 0.05 + 0.001 * fy;
    pp[L.index(.fx)] = 0.001;
    pp[L.index(.fy)] = 0.004 * (fy + 1.0);
    pp[L.index(.fz)] = 0.002;
    return pp;
}

fn initSmooth(s: *SimT) !void {
    var iy: i64 = 0;
    while (iy < s.nyi()) : (iy += 1) {
        var ix: i64 = 0;
        while (ix < s.nxi()) : (ix += 1) {
            try s.initCell(ix, iy, 0, ppAt(ix, iy));
        }
    }
}

test "correct_polaraxis: rows copied from nc, θ-velocities scaled, B untouched" {
    const a = std.testing.allocator;
    var s = try SimT.init(a, testGrid(), testOptions());
    defer s.deinit();
    try initSmooth(&s);

    try s.doCorrect();

    const g = &s.core.grid;
    const ny = s.nyi();
    var ix: i64 = 0;
    while (ix < s.nxi()) : (ix += 1) {
        for ([_][3]i64{
            // {iy, iysrc, axis-face index}
            .{ 0, 2, 0 },
            .{ 1, 2, 0 },
            .{ ny - 1, ny - 3, ny },
            .{ ny - 2, ny - 3, ny },
        }) |c| {
            const iy = c[0];
            const iysrc = c[1];
            const thaxis = g.yl(c[2]);
            const fac = @abs((g.yc(iy) - thaxis) / (g.yc(iysrc) - thaxis));

            var pp: [NV]f64 = undefined;
            s.core.p.load(ix, iy, 0, &pp);
            const pps = ppAt(ix, iysrc);
            const pp0 = ppAt(ix, iy); // pre-correction target values

            // copied scalars & in-row velocities
            inline for ([_]layout.VarTag{ .rho, .uu, .entr, .vx, .vz, .ee, .fx, .fz }) |tag| {
                try std.testing.expectEqual(pps[L.index(tag)], pp[L.index(tag)]);
            }
            // θ-components scaled from the source row
            try std.testing.expectEqual(fac * pps[L.index(.vy)], pp[L.index(.vy)]);
            try std.testing.expectEqual(fac * pps[L.index(.fy)], pp[L.index(.fy)]);
            // B untouched (PUFFY has no CORRECTMAGNFIELD)
            inline for ([_]layout.VarTag{ .b1, .b2, .b3 }) |tag| {
                try std.testing.expectEqual(pp0[L.index(tag)], pp[L.index(tag)]);
            }

            // conserveds rewritten via p2u at the target geometry
            var uu: [NV]f64 = undefined;
            s.core.u.load(ix, iy, 0, &uu);
            const geom = s.core.cache.fillGeometry(ix, iy, 0);
            const uu_expect = try p2u_mod.p2u(cfg, pp, &geom, gam);
            for (0..NV) |iv| try std.testing.expectEqual(uu_expect[iv], uu[iv]);
        }
        // source row itself untouched
        var pp2: [NV]f64 = undefined;
        s.core.p.load(ix, 2, 0, &pp2);
        const exp2 = ppAt(ix, 2);
        for (0..NV) |iv| try std.testing.expectEqual(exp2[iv], pp2[iv]);
    }
}

test "calc_u2p on polar rows: B-only inversion, prims and floors untouched" {
    const a = std.testing.allocator;
    var s = try SimT.init(a, testGrid(), testOptions());
    defer s.deinit();
    try initSmooth(&s);

    // scale the conserveds of a polar cell and an interior cell; u2p must
    // ignore the polar cell (except B = uu[B]/gdet) and invert the interior
    const ix: i64 = 1;
    for ([_]i64{ 0, 3 }) |iy| {
        var uu: [NV]f64 = undefined;
        s.core.u.load(ix, iy, 0, &uu);
        for (0..NV) |iv| uu[iv] *= 1.5;
        s.core.u.store(ix, iy, 0, &uu);
    }

    try s.calcU2p(0.0);

    // polar row: prims still the initialized values, but B rows follow uu
    {
        var pp: [NV]f64 = undefined;
        s.core.p.load(ix, 0, 0, &pp);
        const exp = ppAt(ix, 0);
        const geom = s.core.cache.fillGeometry(ix, 0, 0);
        var uu: [NV]f64 = undefined;
        s.core.u.load(ix, 0, 0, &uu);
        for (0..NV) |iv| {
            const is_b = iv >= L.index(.b1) and iv <= L.index(.b3);
            if (is_b) {
                // C shape: uu * (1./gdetu), not uu/gdetu
                try std.testing.expectEqual(uu[iv] * (1.0 / geom.gdet), pp[iv]);
            } else {
                try std.testing.expectEqual(exp[iv], pp[iv]);
            }
        }
        try std.testing.expectEqual(@as(i32, 0), s.getFlag(.hd_fixup, ix, 0, 0));
        try std.testing.expectEqual(@as(i32, 0), s.getFlag(.rad_fixup, ix, 0, 0));
    }
    // interior row: actually inverted (rho changed from the initialized value)
    {
        var pp: [NV]f64 = undefined;
        s.core.p.load(ix, 3, 0, &pp);
        const exp = ppAt(ix, 3);
        try std.testing.expect(@abs(pp[L.index(.rho)] - exp[L.index(.rho)]) >
            0.1 * exp[L.index(.rho)]);
    }
}

test "op_implicit skips polar rows; cell_fixup never targets them" {
    const a = std.testing.allocator;
    var s = try SimT.init(a, testGrid(), testOptions());
    defer s.deinit();
    try initSmooth(&s);

    const u_before = try a.alloc(f64, s.core.u.data.len);
    defer a.free(u_before);
    @memcpy(u_before, s.core.u.data);
    const p_before = try a.alloc(f64, s.core.p.data.len);
    defer a.free(p_before);
    @memcpy(p_before, s.core.p.data);

    try s.opImplicit(1.0e3);

    const ny = s.nyi();
    var changed_interior = false;
    var iy: i64 = 0;
    while (iy < ny) : (iy += 1) {
        var ix: i64 = 0;
        while (ix < s.nxi()) : (ix += 1) {
            var pp: [NV]f64 = undefined;
            var uu: [NV]f64 = undefined;
            s.core.p.load(ix, iy, 0, &pp);
            s.core.u.load(ix, iy, 0, &uu);
            const off = s.core.p.cellOffset(ix, iy, 0);
            var same = true;
            for (0..NV) |iv| {
                if (pp[iv] != p_before[off + iv] or uu[iv] != u_before[off + iv]) same = false;
            }
            if (iy < 2 or iy > ny - 3) {
                // polar rows bitwise untouched
                try std.testing.expect(same);
            } else if (!same) {
                changed_interior = true;
            }
        }
    }
    try std.testing.expect(changed_interior);

    // a stale nonzero flag on a polar cell must not attract fixups
    s.core.num.fixups.radimp = true;
    var pp_polar: [NV]f64 = undefined;
    s.core.p.load(1, 0, 0, &pp_polar);
    s.core.setFlag(.radimp_fixup, 1, 0, 0, -1);
    try s.cellFixup(.radimp_fixup);
    var pp_after: [NV]f64 = undefined;
    s.core.p.load(1, 0, 0, &pp_after);
    for (0..NV) |iv| try std.testing.expectEqual(pp_polar[iv], pp_after[iv]);
}

test "polaraxis.Band: the claimed rows are exactly the overwritten rows" {
    // The whole point of routing both halves through one `Band`: the rows the
    // predicate claims (so u2p inverts B only, cell_fixup and op_implicit skip
    // them) must be exactly the rows doCorrect then rewrites. A drift between
    // the two leaves rows that no pass supplies — silently, since the B-only
    // u2p branch zeroes the fixup flags. Sim.init preconditions (5) and (7)
    // make the two configs where that could happen unconstructible; this pins
    // the agreement itself on a config that IS constructible.
    const a = std.testing.allocator;
    var s = try SimT.init(a, testGrid(), testOptions());
    defer s.deinit();
    try initSmooth(&s);

    const b = polaraxis.band(SimT.CoreT, &s.core) orelse return error.TestUnexpectedResult;

    var before = std.ArrayList([NV]f64).empty;
    defer before.deinit(a);
    var iy: i64 = 0;
    while (iy < s.nyi()) : (iy += 1) {
        var pp: [NV]f64 = undefined;
        s.core.p.load(1, iy, 0, &pp);
        try before.append(a, pp);
    }

    try s.doCorrect();

    iy = 0;
    while (iy < s.nyi()) : (iy += 1) {
        var pp: [NV]f64 = undefined;
        s.core.p.load(1, iy, 0, &pp);
        const untouched = std.mem.eql(f64, &before.items[@intCast(iy)], &pp);
        // initSmooth varies with iy, so every claimed row genuinely moves
        try std.testing.expectEqual(b.owns(iy), !untouched);
    }
}

// ---- adjust_fluxcttoth_emfs (magn.c:388) -----------------------------------

const cfg_mhd = config.Config{
    .modules = &.{ .hydro, .mhd },
    .reconstruction = .ppm,
    .flux = .laxf,
    .timestepping = .rk2imex,
    .coords = .mks2,
};

/// Radial magnetic flux through the shell of column `ix`: the theta sum of
/// the conserved B^r (= sqrt(-g) B^r; the cells are uniform in the internal
/// theta coordinate).
fn shellFlux(comptime S: type, s: *const S, ix: i64) struct { net: f64, abs: f64 } {
    var net: f64 = 0;
    var abs: f64 = 0;
    var iy: i64 = 0;
    while (iy < s.nyi()) : (iy += 1) {
        const v = s.core.u.get(S.Layout.index(.b1), ix, iy, 0);
        net += v;
        abs += @abs(v);
    }
    return .{ .net = net, .abs = abs };
}

test "flux-CT with the polar band: no magnetic flux crosses the axis, shell flux conserved to roundoff" {
    // The x-face flux of B^r is identically zero in flux-CT and the y-face
    // flux is the average of two corner E_phi, so d/dt sum_theta(gdet B^r)
    // telescopes to the E_phi on the two polar faces. adjust_fluxcttoth_emfs
    // zeroes those: the sum is then an invariant of the scheme, whatever the
    // flow does. Without the zeroing it is not even approximately one on an
    // MKS2 grid: the reflected ghost row sits at a different sin(theta) than
    // its mirror cell, the two x-face B^theta fluxes that should cancel in
    // the corner average do not, and the residual E_phi pumps flux through
    // the axis (the seed of the funnel runaway, see polaraxis.zeroAxisEmfs).
    //
    // The state is deliberately not reflection-symmetric about either pole
    // and carries B^theta and v^theta right up to the axis.
    const puffy = @import("../problems/puffy/puffy.zig");
    const S = sim_mod.Sim(cfg_mhd);
    const LM = S.Layout;
    const a = std.testing.allocator;
    const g = Grid.init(.{
        .nx = 12,
        .ny = 24,
        .ng = 3,
        .minx = @log(4.0 - 0.1),
        .maxx = @log(20.0 - 0.1),
        .miny = 0.001,
        .maxy = 1.0 - 0.001,
    });
    var s = try S.init(a, g, .{
        // the MetricParams default mksh0 = 1 is the degenerate (all-equator) map
        .phys = .{ .coords = .mks2, .mp = .{ .a = 0.5, .mksr0 = 0.1, .mksh0 = 0.9 }, .gam = 4.0 / 3.0 },
        .num = .{ .polaraxis = .{ .ncells = 2 } },
        .bc = .{
            .x = .{ .specific = .{ .f = &puffy.Bc(S).calc } },
            .y = .{ .specific = .{ .f = &puffy.Bc(S).calc } },
        },
    });
    defer s.deinit();

    var iy: i64 = 0;
    while (iy < s.nyi()) : (iy += 1) {
        var ix: i64 = 0;
        while (ix < s.nxi()) : (ix += 1) {
            const geom = s.core.cache.fillGeometry(ix, iy, 0);
            const fy: f64 = @floatFromInt(iy);
            const fx: f64 = @floatFromInt(ix);
            var pp: [S.nv]f64 = @splat(0);
            pp[LM.index(.rho)] = 1.0 + 0.02 * fy + 0.01 * fx;
            pp[LM.index(.uu)] = 0.05 + 0.001 * fy;
            pp[LM.index(.vx)] = -0.05;
            pp[LM.index(.vy)] = 0.004 * @sin(0.37 * fy + 0.2 * fx);
            pp[LM.index(.vz)] = 0.01;
            pp[LM.index(.entr)] = hydro.sFromU(pp[LM.index(.rho)], pp[LM.index(.uu)], 4.0 / 3.0);
            // monopole-like B^r with a theta gradient, plus a B^theta
            // that vanishes on the axis like sin(theta) and a weak
            // toroidal part; a weak field (plasma beta > 100 throughout)
            const gb = precompute.geometryBLat(&g, .mks2, s.core.phys.mp, ix, iy, 0);
            const st = @sin(gb.xxvec[2]);
            pp[LM.index(.b1)] = 0.3 * st * (1.0 + 0.3 * @cos(gb.xxvec[2])) / geom.gdet;
            pp[LM.index(.b2)] = 0.03 * st * st * @cos(0.21 * fy) / geom.gdet;
            pp[LM.index(.b3)] = 1.0e-4 / gb.xxvec[1];
            try s.initCell(ix, iy, 0, pp);
        }
    }
    try s.finishInit();

    var before: [12]f64 = undefined;
    for (&before, 0..) |*b, ix| b.* = shellFlux(S, &s, @intCast(ix)).net;

    const b0 = s.core.u.get(LM.index(.b1), 5, 0, 0);
    for (0..20) |_| try s.step(null);
    const moved = @abs(s.core.u.get(LM.index(.b1), 5, 0, 0) / b0 - 1.0);

    var worst: f64 = 0;
    for (before, 0..) |b, ix| {
        const f = shellFlux(S, &s, @intCast(ix));
        worst = @max(worst, @abs(f.net - b) / f.abs);
    }
    var beta_min: f64 = 1e300;
    iy = 0;
    while (iy < s.nyi()) : (iy += 1) {
        var ix: i64 = 0;
        while (ix < s.nxi()) : (ix += 1) {
            const geom = s.core.cache.fillGeometry(ix, iy, 0);
            var pp: [S.nv]f64 = undefined;
            s.core.p.load(ix, iy, 0, &pp);
            const u = try relele.uconUcovFromPrims(.{ pp[LM.index(.vx)], pp[LM.index(.vy)], pp[LM.index(.vz)] }, &geom);
            const bf = mhd.bconBcovBsqFrom4vel(.{ pp[LM.index(.b1)], pp[LM.index(.b2)], pp[LM.index(.b3)] }, u.con, u.cov, &geom);
            beta_min = @min(beta_min, 2.0 * (4.0 / 3.0 - 1.0) * pp[LM.index(.uu)] / bf.bsq);
        }
    }
    std.debug.print("polar-axis EMF zeroing: min plasma beta {d:.2}; shell flux drift {e:.2} after 20 steps (first-row B^r moved by {e:.2})\n", .{ beta_min, worst, moved });
    // the field in the first row did evolve; only the shell sum is pinned
    try std.testing.expect(moved > 1.0e-6);
    try std.testing.expect(worst < 1.0e-13);
}
