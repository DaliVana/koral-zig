//! Threading determinism gates (M13 + P1).
//!
//! Every per-step pass is dispatched band-parallel on the persistent team
//! when opt.nthreads > 1 (P1 of the parallelization plan). Because every
//! band writes only its own cells/columns/faces, reads only region-frozen
//! state (geometry reads are `*const`), and the only cross-band reductions
//! are order-insensitive (max/min, integer sums), the parallel result must
//! be BIT-IDENTICAL to the serial one; anything else is a data race. Two
//! gates certify this from the same deterministic initial states:
//!
//!  * a 2D Minkowski radiation-MHD box (the original M13 gate; implicit,
//!    u2p, sweep/fluxes/fluxct, wavespeed dt-reduce, stage arithmetic, copy
//!    BCs, fixups) after several steps at nthreads 1 vs 4;
//!  * a full PUFFY 48×44 torus step ×2; adds radviscosity, the dynamo
//!    (scale-height columns + ΔA_φ + superpose), polar-axis correction and
//!    the specific MKS2 boundary conditions.
//!
//! (The golden step tests all run at nthreads = 1, so these are what
//! certify the threaded path.)

const std = @import("std");
const config = @import("../config.zig");
const grid_mod = @import("../grid.zig");
const sim_mod = @import("../sim.zig");
const hydro = @import("../physics/hydro.zig");
const radiation = @import("../physics/radiation.zig");
const radforce = @import("../physics/radforce.zig");
const implicit = @import("../solve/implicit.zig");
const invert = @import("../solve/invert.zig");
const invert_rad = @import("../solve/invert_rad.zig");
const p2u_mod = @import("../p2u.zig");
const puffy = @import("../problems/puffy/puffy.zig");

const Grid = grid_mod.Grid;

const cfg = config.Config{
    .modules = &.{ .hydro, .mhd, .radiation },
    .reconstruction = .ppm,
    .flux = .laxf,
    .timestepping = .rk2imex,
    .coords = .mink,
};
const SimT = sim_mod.Sim(cfg);
const L = SimT.Layout;

fn opts(nthreads: usize) SimT.Options {
    return .{
        .coords = .mink,
        .gam = 5.0 / 3.0,
        .floors = invert.FloorParams.puffy,
        .rad = invert_rad.RadParams.puffy,
        .opac = radforce.Params.puffy(),
        .implicit = implicit.ImplicitParams.puffy,
        .bc_x = .copy,
        .bc_y = .copy,
        .nthreads = nthreads,
    };
}

/// A mildly perturbed 2D radiation-MHD box so that rows genuinely differ
/// (fixup/branch choices vary), exercising the parallel dispatch non-trivially.
fn initBox(s: *SimT) !void {
    const nx = s.nxi();
    const ny = s.nyi();
    var iy: i64 = 0;
    while (iy < ny) : (iy += 1) {
        var ix: i64 = 0;
        while (ix < nx) : (ix += 1) {
            const fx = @as(f64, @floatFromInt(ix)) / @as(f64, @floatFromInt(nx));
            const fy = @as(f64, @floatFromInt(iy)) / @as(f64, @floatFromInt(ny));
            const bump = 1.0 + 0.3 * @sin(6.28 * fx) * @cos(6.28 * fy);
            var pp: [SimT.nv]f64 = @splat(0);
            pp[L.index(.rho)] = 1.0 * bump;
            pp[L.index(.uu)] = 0.1 * bump;
            pp[L.index(.vx)] = 0.05 * @sin(6.28 * fy);
            pp[L.index(.vy)] = 0.02 * @cos(6.28 * fx);
            pp[L.index(.entr)] = hydro.sFromU(pp[L.index(.rho)], pp[L.index(.uu)], 5.0 / 3.0);
            pp[L.index(.b1)] = 0.01;
            pp[L.index(.b2)] = 0.02 * bump;
            pp[L.index(.ee)] = 0.2 * bump;
            pp[L.index(.fx)] = 0.0;
            pp[L.index(.fy)] = 0.0;
            try s.initCell(ix, iy, 0, pp);
        }
    }
    try s.setBc(0.0, true);
    s.initTimestepGuess();
}

test "M13 threading: nthreads=4 is bit-identical to nthreads=1 (2D radiation)" {
    const a = std.testing.allocator;
    const g = Grid.init(.{ .nx = 24, .ny = 20, .ng = 3, .minx = 0, .maxx = 1, .miny = 0, .maxy = 1 });

    var s1 = try SimT.init(a, g, opts(1));
    defer s1.deinit();
    var s4 = try SimT.init(a, g, opts(4));
    defer s4.deinit();

    try initBox(&s1);
    try initBox(&s4);

    // force the same dt on both (serial's CFL choice), several steps
    for (0..4) |_| {
        const dt = 1.0 / s1.tstepdenmax;
        try s1.step(dt);
        try s4.step(dt);
    }

    // every conserved and primitive must match to the last bit
    var iz: i64 = 0;
    while (iz < s1.nzi()) : (iz += 1) {
        var iy: i64 = 0;
        while (iy < s1.nyi()) : (iy += 1) {
            var ix: i64 = 0;
            while (ix < s1.nxi()) : (ix += 1) {
                for (0..SimT.nv) |iv| {
                    try std.testing.expectEqual(s1.u.get(iv, ix, iy, iz), s4.u.get(iv, ix, iy, iz));
                    try std.testing.expectEqual(s1.p.get(iv, ix, iy, iz), s4.p.get(iv, ix, iy, iz));
                }
            }
        }
    }
    // counters (summed across threads) must also agree exactly
    try std.testing.expectEqual(s1.n_radimp_failures, s4.n_radimp_failures);
    try std.testing.expectEqual(s1.n_radimp_iters, s4.n_radimp_iters);
    // and the CFL dt reduction (per-worker max/min partials, merged)
    try std.testing.expectEqual(s1.tstepdenmax, s4.tstepdenmax);
    try std.testing.expectEqual(s1.tstepdenmin, s4.tstepdenmin);
}

// ---- the P1 whole-step gate on the real problem ---------------------------

const SimP = sim_mod.Sim(config.puffy);

/// The production PUFFY option set (PROBLEMS/puffy/main.zig) at test scale;
/// every threaded pass is active: radviscosity, dynamo, polar correction,
/// specific MKS2 BCs.
fn puffyOpts(nthreads: usize) SimP.Options {
    return .{
        .coords = .mks2,
        .mp = puffy.mp,
        .gam = puffy.gam,
        .floors = invert.FloorParams.puffy,
        .rad = invert_rad.RadParams.puffy,
        .opac = radforce.Params.puffy(),
        .implicit = implicit.ImplicitParams.puffy,
        .correct_polaraxis = true,
        .nccorrectpolar = 2,
        .radviscosity = true,
        .dynamo = true,
        .bc_x = .specific,
        .bc_y = .specific,
        .specific_bc = &puffy.Bc(SimP).calc,
        .nthreads = nthreads,
    };
}

test "P1 threading: full PUFFY step nthreads=4 is bit-identical to nthreads=1" {
    const a = std.testing.allocator;

    var s1 = try SimP.init(a, puffy.makeGrid(48, 44), puffyOpts(1));
    defer s1.deinit();
    var s4 = try SimP.init(a, puffy.makeGrid(48, 44), puffyOpts(4));
    defer s4.deinit();

    _ = try puffy.initAll(SimP, &s1);
    _ = try puffy.initAll(SimP, &s4);
    s1.initTimestepGuess();
    s4.initTimestepGuess();

    // the production sequence: the tiny guess-dt startup step, then a
    // CFL-sized step (forced identically on both sims)
    for (0..2) |_| {
        const dt = 1.0 / s1.tstepdenmax;
        try s1.step(dt);
        try s4.step(dt);
    }

    // full arrays incl. ghosts, every flag, the reductions — to the bit
    for (s1.p.data, s4.p.data) |v1, v4| try std.testing.expectEqual(v1, v4);
    for (s1.u.data, s4.u.data) |v1, v4| try std.testing.expectEqual(v1, v4);
    for (s1.visc.?.rijvisc.data, s4.visc.?.rijvisc.data) |v1, v4| try std.testing.expectEqual(v1, v4);
    for (s1.dynamo.?.scaleth, s4.dynamo.?.scaleth) |v1, v4| try std.testing.expectEqual(v1, v4);
    for (s1.flags, s4.flags) |f1, f4| try std.testing.expectEqual(f1, f4);
    try std.testing.expectEqual(s1.n_radimp_failures, s4.n_radimp_failures);
    try std.testing.expectEqual(s1.n_radimp_iters, s4.n_radimp_iters);
    try std.testing.expectEqual(s1.tstepdenmax, s4.tstepdenmax);
    try std.testing.expectEqual(s1.tstepdenmin, s4.tstepdenmin);
    try std.testing.expectEqual(s1.t, s4.t);
}
