//! Theory gates for the CFL step (sim/timestep.zig, Core.cflDt).
//!
//!  * Courant–Friedrichs–Lewy on a uniform Minkowski gas at rest:
//!    Δt = TSTEPLIM / Σ_d (c_s/Δx_d) with c_s² = Γ(Γ−1)u / (ρ + Γu). The
//!    per-dimension speeds are SUMMED (C: finite.c calc_wavespeeds, the
//!    conservative multi-D bound), not maxed; 1D and 2D with Δx ≠ Δy @1e-12.
//!  * boosted gas (v along x): the fastest x characteristic is the
//!    relativistic velocity addition (v + c_s)/(1 + v c_s); the transverse
//!    (y) speed is c_s √(1−v²)/√(1−v²c_s²) — the 3+1 SR hydro eigenvalues
//!    with v_y = 0 (Font, Ibáñez, Martí & Marquina 1994; Martí & Müller
//!    2003 eq. 21) @1e-12.
//!  * TSTEPLIM scales Δt linearly.
//!
//! Each Δt is read the way the driver reads it: after one (forced, tiny)
//! Sim.step, which resets tstepdenmax and recomputes the wavespeeds on the
//! current state; the uniform state is invariant to 1e-14, so the CFL bound
//! is that of the analytic state.

const std = @import("std");
const config = @import("../config.zig");
const grid_mod = @import("../grid.zig");
const sim_mod = @import("../sim.zig");
const hydro = @import("../physics/hydro.zig");

const Grid = grid_mod.Grid;

const cfg = config.Config{
    .modules = &.{.hydro},
    .reconstruction = .ppm,
    .flux = .laxf,
    .timestepping = .rk2imex,
    .coords = .mink,
};
const SimT = sim_mod.Sim(cfg);
const L = SimT.Layout;
const NV = SimT.nv;

const gam: f64 = 5.0 / 3.0;
const rho: f64 = 1.0;
const uint: f64 = 0.05;

fn soundSpeed() f64 {
    return @sqrt(gam * (gam - 1.0) * uint / (rho + gam * uint));
}

/// Δt the driver would take next on a uniform gas moving at 3-velocity v
/// along x, on an nx×ny grid with cell sizes dx, dy.
fn cflDt(a: std.mem.Allocator, nx: usize, ny: usize, dx: f64, dy: f64, v: f64, tsteplim: f64) !f64 {
    const g = Grid.init(.{
        .nx = nx,
        .ny = ny,
        .ng = 3,
        .minx = 0,
        .maxx = dx * @as(f64, @floatFromInt(nx)),
        .miny = 0,
        .maxy = dy * @as(f64, @floatFromInt(ny)),
    });
    var s = try SimT.init(a, g, .{
        .phys = .{ .coords = .mink, .gam = gam },
        .num = .{ .tsteplim = tsteplim },
        .bc = .{ .x = .periodic, .y = .periodic },
    });
    defer s.deinit();

    var pp: [NV]f64 = @splat(0);
    pp[L.index(.rho)] = rho;
    pp[L.index(.uu)] = uint;
    pp[L.index(.vx)] = v / @sqrt(1.0 - v * v); // VELR
    pp[L.index(.entr)] = hydro.sFromU(rho, uint, gam);
    var iy: i64 = 0;
    while (iy < s.nyi()) : (iy += 1) {
        var ix: i64 = 0;
        while (ix < s.nxi()) : (ix += 1) try s.initCell(ix, iy, 0, pp);
    }
    try s.finishInit();
    try s.step(1e-6);
    return s.cflDt();
}

fn expectRel(got: f64, want: f64, tol: f64) !void {
    const rel = @abs(got - want) / @abs(want);
    if (rel > tol) {
        std.debug.print("cfl: got {e:.15} want {e:.15} rel {e:.2}\n", .{ got, want, rel });
        return error.TestUnexpectedResult;
    }
}

test "CFL: uniform gas at rest, 1D — dt = tsteplim·dx/c_s" {
    const a = std.testing.allocator;
    const cs = soundSpeed();
    const dt = try cflDt(a, 8, 1, 0.125, 1.0, 0.0, 0.5);
    try expectRel(dt, 0.5 * 0.125 / cs, 1e-12);
}

test "CFL: uniform gas at rest, 2D with dx ≠ dy — per-dimension speeds sum" {
    const a = std.testing.allocator;
    const cs = soundSpeed();
    const dt = try cflDt(a, 8, 4, 0.125, 0.25, 0.0, 0.5);
    try expectRel(dt, 0.5 / (cs / 0.125 + cs / 0.25), 1e-12);
}

test "CFL: boosted gas — relativistic velocity addition along the boost, transverse c_s√(1−v²)/√(1−v²c_s²)" {
    const a = std.testing.allocator;
    const cs = soundSpeed();
    const v: f64 = 0.6;
    const lam_x = (v + cs) / (1.0 + v * cs);
    const lam_y = cs * @sqrt(1.0 - v * v) / @sqrt(1.0 - v * v * cs * cs);
    const dt = try cflDt(a, 8, 4, 0.125, 0.25, v, 0.5);
    try expectRel(dt, 0.5 / (lam_x / 0.125 + lam_y / 0.25), 1e-12);
}

test "CFL: tsteplim scales dt linearly" {
    const a = std.testing.allocator;
    const dt_half = try cflDt(a, 8, 4, 0.125, 0.25, 0.3, 0.5);
    const dt_quarter = try cflDt(a, 8, 4, 0.125, 0.25, 0.3, 0.25);
    try expectRel(dt_quarter, 0.5 * dt_half, 1e-13);
}
