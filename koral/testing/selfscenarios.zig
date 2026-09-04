//! The scenarios pinned by the self-goldens, defined ONCE so the generator
//! (`tools/gen_self_golden.zig`) and the checker (`tests/selfgolden_tests.zig`)
//! cannot drift apart: both call `run` and the checker simply compares its
//! fresh records against the committed ones. A baseline you can no longer
//! reproduce by construction is worse than no baseline at all.
//!
//! The option sets are written out inline rather than taken from
//! `puffy.simOptions(p)` on purpose. They reference this repository's physics
//! constants (`FloorParams.puffy`, `RadParams.puffy`, …), so a change to the
//! *code* moves the numbers and the self-golden catches it, which is the whole
//! point: while the `.toml`-driven knobs stay pinned at fixed values, so
//! editing a params default or adding a new one does not silently redefine the
//! baseline.
//!
//! The step loop mirrors the production driver (problems/puffy/main.zig): init
//! → `initTimestepGuess` → repeat { `dt = cflDt()`; `step(dt)` }. The dt comes
//! from the sim's own CFL logic rather than being forced, so the wavespeed and
//! timestep machinery is pinned along with the state.

const std = @import("std");
const config = @import("../config.zig");
const sim_mod = @import("../sim.zig");
const storage = @import("../sim/storage.zig");
const selfgolden = @import("selfgolden.zig");
const puffy = @import("../problems/puffy/puffy.zig");
const invert = @import("../solve/invert.zig");
const invert_rad = @import("../solve/invert_rad.zig");
const implicit = @import("../solve/implicit.zig");
const radforce = @import("../physics/radforce.zig");

const SimP = sim_mod.Sim(config.puffy);

pub const Scenario = struct {
    /// human label, embedded in the file
    name: []const u8,
    /// path under tests/selfgolden/
    file: []const u8,
    nx: usize,
    ny: usize,
    nsteps: usize,
    /// gated behind -Dslow-tests
    slow: bool,
};

/// Small enough for the default battery, large enough to carry the physics
/// that matters: torus + atmosphere, both polar axes, both radial BCs.
pub const puffy_fast = Scenario{
    .name = "puffy 32x30 mks2, 3 CFL steps",
    .file = "puffy_fast.kslf",
    .nx = 32,
    .ny = 30,
    .nsteps = 3,
    .slow = false,
};

/// The production-shaped keystone: same grid as the threading determinism
/// gate, long enough for the dynamo and radiative viscosity to have acted.
pub const puffy_full = Scenario{
    .name = "puffy 48x44 mks2, 8 CFL steps",
    .file = "puffy_full.kslf",
    .nx = 48,
    .ny = 44,
    .nsteps = 8,
    .slow = true,
};

pub const all = [_]Scenario{ puffy_fast, puffy_full };

/// The full PUFFY physics stack, matching problems/puffy/main.zig's production
/// option set at its validated defaults (the `.toml` knobs pinned).
fn puffyOptions() SimP.Options {
    return .{
        .phys = .{ .coords = .mks2, .mp = puffy.mp, .gam = puffy.gam, .floors = invert.FloorParams.puffy, .rad = invert_rad.RadParams.puffy, .opac = radforce.Params.puffy(), .implicit = implicit.ImplicitParams.puffy },
        .num = .{ .polaraxis = .{ .ncells = 2 } },
        .bc = .{ .x = .{ .specific = .{ .f = &puffy.Bc(SimP).calc } }, .y = .{ .specific = .{ .f = &puffy.Bc(SimP).calc } } },
        .radvisc = .{},
        .dynamo = .{},
    };
}

/// Run `sc` and return the records: index 0 is the post-init state, index k>0
/// the state after step k. Caller owns the returned Writer.
pub fn run(allocator: std.mem.Allocator, sc: Scenario) !selfgolden.Writer {
    var s = try SimP.init(allocator, puffy.makeGrid(sc.nx, sc.ny), puffyOptions());
    defer s.deinit();

    _ = try puffy.initAll(SimP, &s);
    s.initTimestepGuess();

    var w = selfgolden.Writer.init(allocator, .{
        .nx = @intCast(sc.nx),
        .ny = @intCast(sc.ny),
        .nz = 1,
        .ng = @intCast(s.core.grid.ng),
        .nv = SimP.nv,
        .ncell = s.core.grid.cellCount(),
        .nflags = storage.n_flags,
    }, sc.name);
    errdefer w.deinit();

    // Scalars every step; full fields only at the endpoints (see the format
    // note in selfgolden.zig — a perturbation at step k is still there at the
    // end, and each snapshot costs ~350 KiB of committed history).
    try addScalars(&w, &s);
    try w.addSnapshot(0, s.core.u.data, s.core.p.data, s.core.flags);
    for (0..sc.nsteps) |istep| {
        try s.step(s.cflDt());
        try addScalars(&w, &s);
        if (istep + 1 == sc.nsteps) try w.addSnapshot(istep + 1, s.core.u.data, s.core.p.data, s.core.flags);
    }
    return w;
}

fn addScalars(w: *selfgolden.Writer, s: *const SimP) !void {
    try w.addScalars(.{
        .t = s.t,
        .dt = s.dt,
        .own_dt = s.own_dt,
        .tstepdenmax = s.core.tstepdenmax,
        .tstepdenmin = s.core.tstepdenmin,
        .n_radimp_failures = @floatFromInt(s.n_radimp_failures),
        .n_radimp_iters = @floatFromInt(s.n_radimp_iters),
    });
}
