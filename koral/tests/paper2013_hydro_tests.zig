//! Reproduction gate for Sadowski et al. (2013), section 4.1 and Figure 1.
//!
//! The paper evolves the Table 1 relativistic shock tube on 500 cells to
//! t = 50 and compares linear minmod reconstruction with theta = 1 and 2
//! against an exact Riemann solution.  Both reconstruction choices exist in
//! the current solver.  MP5 and the paper's RK3 integrator do not, so this
//! test deliberately covers only the two like-for-like linear runs.

const std = @import("std");
const config = @import("../config.zig");
const grid_mod = @import("../grid.zig");
const sim_mod = @import("../sim.zig");
const hydro = @import("../physics/hydro.zig");

const gam: f64 = 5.0 / 3.0;

const cfg = config.Config{
    .modules = &.{.hydro},
    .reconstruction = .linear,
    .flux = .laxf,
    .timestepping = .rk2imex,
    .coords = .mink,
};
const SimT = sim_mod.Sim(cfg);
const L = SimT.Layout;
const NV = SimT.nv;
const Grid = grid_mod.Grid;

const State = struct { rho: f64, p: f64, v: f64 };

fn soundSpeed(s: State) f64 {
    const h = 1.0 + gam / (gam - 1.0) * s.p / s.rho;
    return @sqrt(gam * s.p / (s.rho * h));
}

fn invariantG(cs: f64) f64 {
    const root = @sqrt(gam - 1.0);
    return 2.0 / root * std.math.atanh(cs / root);
}

fn rarefactionVelocity(left: State, p: f64) f64 {
    const k = left.p / std.math.pow(f64, left.rho, gam);
    const rho = std.math.pow(f64, p / k, 1.0 / gam);
    const rapidity = std.math.atanh(left.v) + invariantG(soundSpeed(left)) -
        invariantG(soundSpeed(.{ .rho = rho, .p = p, .v = 0.0 }));
    return std.math.tanh(rapidity);
}

const Shock = struct { rho: f64, v: f64, speed: f64 };

fn shockState(right: State, p: f64) Shock {
    const gp = gam / (gam - 1.0);
    const ha = 1.0 + gp * right.p / right.rho;
    const wa = 1.0 / @sqrt(1.0 - right.v * right.v);
    const alpha = (p - right.p) / (gp * p);
    const aq = 1.0 - alpha;
    const cq = -(ha * ha + ha / right.rho * (p - right.p));
    const hb = (-alpha + @sqrt(alpha * alpha - 4.0 * aq * cq)) / (2.0 * aq);
    const rho = gp * p / (hb - 1.0);
    const ea = right.rho * ha - right.p;
    const eb = rho * hb - p;
    const vrel = @sqrt((p - right.p) * (eb - ea) / ((ea + p) * (eb + right.p)));
    const v = (right.v + vrel) / (1.0 + right.v * vrel);
    const wb = 1.0 / @sqrt(1.0 - v * v);
    const speed = (rho * wb * v - right.rho * wa * right.v) /
        (rho * wb - right.rho * wa);
    return .{ .rho = rho, .v = v, .speed = speed };
}

const Exact = struct {
    left: State,
    right: State,
    pstar: f64,
    vstar: f64,
    rho_l_star: f64,
    rho_r_star: f64,
    shock_speed: f64,

    fn init(left: State, right: State) Exact {
        var lo = right.p;
        var hi = left.p;
        for (0..200) |_| {
            const p = 0.5 * (lo + hi);
            if (rarefactionVelocity(left, p) > shockState(right, p).v)
                lo = p
            else
                hi = p;
        }
        const pstar = 0.5 * (lo + hi);
        const shock = shockState(right, pstar);
        const k = left.p / std.math.pow(f64, left.rho, gam);
        return .{
            .left = left,
            .right = right,
            .pstar = pstar,
            .vstar = rarefactionVelocity(left, pstar),
            .rho_l_star = std.math.pow(f64, pstar / k, 1.0 / gam),
            .rho_r_star = shock.rho,
            .shock_speed = shock.speed,
        };
    }

    fn sample(self: Exact, xi: f64) State {
        const cs_l = soundSpeed(self.left);
        const head = (self.left.v - cs_l) / (1.0 - self.left.v * cs_l);
        const cs_star = soundSpeed(.{ .rho = self.rho_l_star, .p = self.pstar, .v = self.vstar });
        const tail = (self.vstar - cs_star) / (1.0 - self.vstar * cs_star);

        if (xi <= head) return self.left;
        if (xi >= self.shock_speed) return self.right;
        if (xi >= self.vstar) return .{ .rho = self.rho_r_star, .p = self.pstar, .v = self.vstar };
        if (xi >= tail) return .{ .rho = self.rho_l_star, .p = self.pstar, .v = self.vstar };

        const k = self.left.p / std.math.pow(f64, self.left.rho, gam);
        const invariant = std.math.atanh(self.left.v) + invariantG(cs_l);
        var lo: f64 = 1.0e-14;
        var hi = cs_l;
        for (0..160) |_| {
            const cs = 0.5 * (lo + hi);
            const v = std.math.tanh(invariant - invariantG(cs));
            const characteristic = (v - cs) / (1.0 - v * cs);
            if (characteristic > xi) lo = cs else hi = cs;
        }
        const cs = 0.5 * (lo + hi);
        const v = std.math.tanh(invariant - invariantG(cs));
        const rho = std.math.pow(
            f64,
            cs * cs * (gam - 1.0) / (gam * k * (gam - 1.0 - cs * cs)),
            1.0 / (gam - 1.0),
        );
        return .{ .rho = rho, .p = k * std.math.pow(f64, rho, gam), .v = v };
    }
};

fn primitives(s: State) [NV]f64 {
    var pp: [NV]f64 = @splat(0);
    pp[L.index(.rho)] = s.rho;
    pp[L.index(.uu)] = s.p / (gam - 1.0);
    pp[L.index(.vx)] = s.v / @sqrt(1.0 - s.v * s.v);
    pp[L.index(.entr)] = hydro.sFromU(s.rho, pp[L.index(.uu)], gam);
    return pp;
}

const Result = struct {
    density_l1: f64,
    velocity_l1: f64,
    shock_error_cells: f64,
};

fn run(theta: f64) !Result {
    const nx = 500;
    const t_end: f64 = 50.0;
    const grid = Grid.init(.{ .nx = nx, .ng = 2, .minx = -50.0, .maxx = 50.0 });
    var sim = try SimT.init(std.testing.allocator, grid, .{
        .phys = .{ .coords = .mink, .gam = gam },
        .num = .{ .minmod_theta = theta },
        .bc = .{ .x = .copy },
    });
    defer sim.deinit();

    const left = State{ .rho = 10.0, .p = 13.33, .v = 0.0 };
    const right = State{ .rho = 1.0, .p = 1.0e-8, .v = 0.0 };
    var ix: i64 = 0;
    while (ix < sim.nxi()) : (ix += 1) {
        try sim.initCell(ix, 0, 0, primitives(if (grid.xc(ix) < 0.0) left else right));
    }
    try sim.finishInit();

    while (sim.t < t_end) {
        var dt: ?f64 = null;
        const cfl_dt = 1.0 / sim.core.tstepdenmax;
        if (sim.t + cfl_dt > t_end) dt = t_end - sim.t;
        try sim.step(dt);
    }

    const exact = Exact.init(left, right);
    var density_l1: f64 = 0.0;
    var velocity_l1: f64 = 0.0;
    var steepest: f64 = 0.0;
    var shock_x: f64 = 0.0;
    ix = 0;
    while (ix < sim.nxi()) : (ix += 1) {
        const ex = exact.sample(grid.xc(ix) / t_end);
        const ux = sim.core.p.get(L.index(.vx), ix, 0, 0);
        const lorentz = @sqrt(1.0 + ux * ux);
        const density = sim.core.p.get(L.index(.rho), ix, 0, 0) * lorentz;
        const exact_lorentz = 1.0 / @sqrt(1.0 - ex.v * ex.v);
        density_l1 += @abs(density - ex.rho * exact_lorentz);
        velocity_l1 += @abs(ux / lorentz - ex.v);

        if (ix + 1 < sim.nxi() and grid.xc(ix) > exact.vstar * t_end) {
            const ux_next = sim.core.p.get(L.index(.vx), ix + 1, 0, 0);
            const d_next = sim.core.p.get(L.index(.rho), ix + 1, 0, 0) * @sqrt(1.0 + ux_next * ux_next);
            const gradient = density - d_next;
            if (gradient > steepest) {
                steepest = gradient;
                shock_x = 0.5 * (grid.xc(ix) + grid.xc(ix + 1));
            }
        }
    }

    return .{
        .density_l1 = density_l1 / @as(f64, @floatFromInt(nx)) / (left.rho - right.rho),
        .velocity_l1 = velocity_l1 / @as(f64, @floatFromInt(nx)),
        .shock_error_cells = @abs(shock_x - exact.shock_speed * t_end) / grid.dx,
    };
}

test "Sadowski 2013 section 4.1: 500-cell relativistic shock matches Figure 1" {
    const theta1 = try run(1.0);
    const theta2 = try run(2.0);
    std.debug.print(
        "paper 2013 hydro: theta=1 D-L1={e:.3} v-L1={e:.3}; theta=2 D-L1={e:.3} v-L1={e:.3}; shock errors={d:.2}/{d:.2} cells\n",
        .{ theta1.density_l1, theta1.velocity_l1, theta2.density_l1, theta2.velocity_l1, theta1.shock_error_cells, theta2.shock_error_cells },
    );

    // Figure 1's two claims that are representable by today's solver:
    // both linear schemes track the exact solution, and theta=1 is the more
    // diffusive density reconstruction.  Tolerances are numerical, not a
    // digitized comparison against the published plot.
    try std.testing.expect(theta1.density_l1 < 1.5e-2);
    try std.testing.expect(theta2.density_l1 < 1.5e-2);
    try std.testing.expect(theta1.velocity_l1 < 1.0e-2);
    try std.testing.expect(theta2.velocity_l1 < 1.0e-2);
    try std.testing.expect(theta2.density_l1 < theta1.density_l1);
    try std.testing.expect(theta1.shock_error_cells <= 3.0);
    try std.testing.expect(theta2.shock_error_cells <= 3.0);
}
