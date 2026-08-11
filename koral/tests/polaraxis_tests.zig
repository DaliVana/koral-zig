//! M10 theory gates for the polar-axis special treatment
//! (CORRECT_POLARAXIS, NCCORRECTPOLAR = 2 — PUFFY define.h:107-109):
//!
//!   correct_polaraxis (finite.c:5525)  — most-polar rows overwritten from
//!                                        row nc, θ-velocities scaled, B
//!                                        untouched (no CORRECTMAGNFIELD)
//!   calc_u2p (u2p.c:57/80/92)          — polar cells get u2p_solver_Bonly
//!                                        and skip both floor checks
//!   op_implicit (finite.c:1427)        — polar cells skipped entirely
//!   cell_fixup (finite.c:5042)         — polar cells never fixup targets
//!
//! Pure-Zig pins of the transcription; the C-diff of the whole pipeline on
//! a spherical grid arrives with the PUFFY t=0 + step goldens (M11/M13).

const std = @import("std");
const config = @import("../config.zig");
const layout = @import("../layout.zig");
const grid_mod = @import("../grid.zig");
const sim_mod = @import("../sim.zig");
const p2u_mod = @import("../p2u.zig");
const hydro = @import("../physics/hydro.zig");
const radforce = @import("../physics/radforce.zig");
const invert_rad = @import("../solve/invert_rad.zig");
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
        .coords = .mks2,
        .gam = gam,
        .rad = invert_rad.RadParams.puffy,
        .opac = radforce.Params.puffy(),
        .implicit = implicit.ImplicitParams.puffy,
        .correct_polaraxis = true,
        .nccorrectpolar = 2,
        .bc_y = .copy,
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

    const g = &s.grid;
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
            s.p.load(ix, iy, 0, &pp);
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
            s.u.load(ix, iy, 0, &uu);
            const geom = s.cache.fillGeometry(ix, iy, 0);
            const uu_expect = try p2u_mod.p2u(cfg, pp, &geom, gam);
            for (0..NV) |iv| try std.testing.expectEqual(uu_expect[iv], uu[iv]);
        }
        // source row itself untouched
        var pp2: [NV]f64 = undefined;
        s.p.load(ix, 2, 0, &pp2);
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
        s.u.load(ix, iy, 0, &uu);
        for (0..NV) |iv| uu[iv] *= 1.5;
        s.u.store(ix, iy, 0, &uu);
    }

    try s.calcU2p(0.0);

    // polar row: prims still the initialized values, but B rows follow uu
    {
        var pp: [NV]f64 = undefined;
        s.p.load(ix, 0, 0, &pp);
        const exp = ppAt(ix, 0);
        const geom = s.cache.fillGeometry(ix, 0, 0);
        var uu: [NV]f64 = undefined;
        s.u.load(ix, 0, 0, &uu);
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
        s.p.load(ix, 3, 0, &pp);
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

    const u_before = try a.alloc(f64, s.u.data.len);
    defer a.free(u_before);
    @memcpy(u_before, s.u.data);
    const p_before = try a.alloc(f64, s.p.data.len);
    defer a.free(p_before);
    @memcpy(p_before, s.p.data);

    try s.opImplicit(0.0, 1.0e3);

    const ny = s.nyi();
    var changed_interior = false;
    var iy: i64 = 0;
    while (iy < ny) : (iy += 1) {
        var ix: i64 = 0;
        while (ix < s.nxi()) : (ix += 1) {
            var pp: [NV]f64 = undefined;
            var uu: [NV]f64 = undefined;
            s.p.load(ix, iy, 0, &pp);
            s.u.load(ix, iy, 0, &uu);
            const off = s.p.cellOffset(ix, iy, 0);
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
    s.opt.do_radimp_fixups = true;
    var pp_polar: [NV]f64 = undefined;
    s.p.load(1, 0, 0, &pp_polar);
    s.setFlag(.radimp_fixup, 1, 0, 0, -1);
    try s.cellFixup(.radimp_fixup);
    var pp_after: [NV]f64 = undefined;
    s.p.load(1, 0, 0, &pp_after);
    for (0..NV) |iv| try std.testing.expectEqual(pp_polar[iv], pp_after[iv]);
}
