//! Reproduction gate for Sadowski et al. (2013), section 4.4 and Figure 9.
//!
//! This uses the paper's 101-cell domain, temperature profile, radiative
//! constant, scattering opacity, and four plotted output times.  The exact
//! diffusion profile is evaluated analytically by expanding
//! (1 + 100 exp(-x^2/w^2))^4 into four Gaussian terms.  The current code uses
//! its production PPM, M1, and RK2IMEX choices rather than the paper's
//! reconstruction/RK3 combination.

const std = @import("std");
const build_options = @import("build_options");
const config = @import("../config.zig");
const grid_mod = @import("../grid.zig");
const sim_mod = @import("../sim.zig");
const units_mod = @import("../units.zig");
const hydro = @import("../physics/hydro.zig");
const thermo = @import("../physics/thermo.zig");
const radforce = @import("../physics/radforce.zig");
const invert_rad = @import("../solve/invert_rad.zig");

const cfg = config.Config{
    .modules = &.{ .hydro, .radiation },
    .reconstruction = .ppm,
    .flux = .laxf,
    .timestepping = .rk2imex,
    .coords = .mink,
};
const SimT = sim_mod.Sim(cfg);
const L = SimT.Layout;
const NV = SimT.nv;
const Grid = grid_mod.Grid;

const gam: f64 = 5.0 / 3.0;
const rho_paper: f64 = 1.0;
const t0: f64 = 1.0e6;
const width: f64 = 5.0;
const amplitude: f64 = 100.0;
const sigma_paper: f64 = 1.56e-64;
const kappa_es: f64 = 1.0e3;
const diffusion: f64 = 1.0 / (3.0 * kappa_es * rho_paper);
const e_background_paper: f64 = 4.0 * sigma_paper * t0 * t0 * t0 * t0;

// Pure-scattering grey RHD is homogeneous under
// (rho, u, E) -> scale (rho, u, E), kappa_es -> kappa_es / scale.
// Apply that exact symmetry so the paper's E ~ 1e-40 is not judged through
// absolute Newton tolerances.  Temperature, E/rho, optical depth, D, and the
// dimensionless solution are unchanged.
const state_scale: f64 = 1.0e30;
const rho_sim: f64 = state_scale * rho_paper;
const e_background_sim: f64 = state_scale * e_background_paper;

fn paperMass() f64 {
    // sigma_rad scales exactly as M^2 in geometrized units.
    return @sqrt(sigma_paper / units_mod.Units.init(1.0).sigmaRad());
}

fn initialEnergy(x: f64) f64 {
    const q = 1.0 + amplitude * @exp(-x * x / (width * width));
    return e_background_sim * q * q * q * q;
}

/// Infinite-domain solution of dE/dt = D d2E/dx2 for the paper's profile.
/// At the latest plotted time the pulse is still well inside [-50, 50].
fn exactEnergy(x: f64, t: f64) f64 {
    const choose = [_]f64{ 1.0, 4.0, 6.0, 4.0, 1.0 };
    var sum: f64 = choose[0];
    var amp_power: f64 = 1.0;
    for (1..5) |k_usize| {
        amp_power *= amplitude;
        const k: f64 = @floatFromInt(k_usize);
        const spread = width * width + 4.0 * k * diffusion * t;
        sum += choose[k_usize] * amp_power * width / @sqrt(spread) *
            @exp(-k * x * x / spread);
    }
    return e_background_sim * sum;
}

const Moments = struct { centroid: f64, variance: f64 };

fn moments(sim: *const SimT) Moments {
    var m0: f64 = 0.0;
    var m1: f64 = 0.0;
    var ix: i64 = 0;
    while (ix < sim.nxi()) : (ix += 1) {
        const excess = @max(sim.core.p.get(L.index(.ee), ix, 0, 0) - e_background_sim, 0.0);
        m0 += excess;
        m1 += excess * sim.core.grid.xc(ix);
    }
    const centroid = m1 / m0;
    var m2: f64 = 0.0;
    ix = 0;
    while (ix < sim.nxi()) : (ix += 1) {
        const excess = @max(sim.core.p.get(L.index(.ee), ix, 0, 0) - e_background_sim, 0.0);
        const dx = sim.core.grid.xc(ix) - centroid;
        m2 += excess * dx * dx;
    }
    return .{ .centroid = centroid, .variance = m2 / m0 };
}

fn profileError(sim: *const SimT, t: f64) f64 {
    var numerator: f64 = 0.0;
    var denominator: f64 = 0.0;
    var ix: i64 = 0;
    while (ix < sim.nxi()) : (ix += 1) {
        const exact = exactEnergy(sim.core.grid.xc(ix), t);
        numerator += @abs(sim.core.p.get(L.index(.ee), ix, 0, 0) - exact);
        denominator += exact - e_background_sim;
    }
    return numerator / denominator;
}

const RunSummary = struct {
    worst_profile: f64,
    last_profile: f64,
    last_d_ratio: f64,
};

fn runPaperPulse(output_times: []const f64) !RunSummary {
    const mass = paperMass();
    const units = units_mod.Units.init(mass);
    const comp = thermo.Composition.cdefault;
    const opac = radforce.Params.grey(units, comp, 0.0, kappa_es / state_scale);

    // The published background E/rho is about 6e-40.  KORAL's generic
    // ratio floors target accretion problems and would replace that valid
    // state, so lower only those test-problem floors; the absolute 1e-79
    // radiation floor remains in force.
    var rad = invert_rad.RadParams.cdefault;
    rad.eerhoratiomin = 1.0e-50;
    rad.eeuuratiomin = 1.0e-50;

    const grid = Grid.init(.{ .nx = 101, .ng = 3, .minx = -50.0, .maxx = 50.0 });
    var sim = try SimT.init(std.testing.allocator, grid, .{
        .phys = .{ .coords = .mink, .gam = gam, .rad = rad, .opac = opac },
        .bc = .{ .x = .copy },
    });
    defer sim.deinit();

    try std.testing.expectApproxEqRel(sigma_paper, units.sigmaRad(), 2.0e-15);
    const gas_u = thermo.uFromTrho(&opac.consts, t0, rho_sim, gam);
    var ix: i64 = 0;
    while (ix < sim.nxi()) : (ix += 1) {
        var pp: [NV]f64 = @splat(0);
        pp[L.index(.rho)] = rho_sim;
        pp[L.index(.uu)] = gas_u;
        pp[L.index(.entr)] = hydro.sFromU(rho_sim, gas_u, gam);
        pp[L.index(.ee)] = initialEnergy(grid.xc(ix));
        try sim.initCell(ix, 0, 0, pp);
    }
    try sim.finishInit();

    const initial = moments(&sim);
    var worst_profile: f64 = 0.0;
    var last_profile: f64 = 0.0;
    var last_d_ratio: f64 = 0.0;
    for (output_times) |target| {
        while (sim.t < target) {
            var dt: ?f64 = null;
            const cfl_dt = 1.0 / sim.core.tstepdenmax;
            if (sim.t + cfl_dt > target) dt = target - sim.t;
            try sim.step(dt);
        }
        const err = profileError(&sim, target);
        const now = moments(&sim);
        const measured_d = (now.variance - initial.variance) / (2.0 * target);
        const d_ratio = measured_d / diffusion;
        worst_profile = @max(worst_profile, err);
        last_profile = err;
        last_d_ratio = d_ratio;
        std.debug.print(
            "paper 2013 thick pulse: t={e:.0}, profile L1={e:.3}, D/D_exact={d:.4}, centroid={e:.2}\n",
            .{ target, err, d_ratio, now.centroid },
        );
        try std.testing.expect(@abs(now.centroid) <= 0.1 * grid.dx);
    }

    return .{
        .worst_profile = worst_profile,
        .last_profile = last_profile,
        .last_d_ratio = last_d_ratio,
    };
}

test "Sadowski 2013 section 4.4: thick pulse has the early Figure 9 diffusion rate" {
    const summary = try runPaperPulse(&.{3.0e3});

    // The first published output is the most sensitive to truncation error.
    // Measured with the current PPM path: profile L1 = 0.0118 and
    // D/D_exact = 0.9735 in both Debug and ReleaseFast.
    try std.testing.expect(summary.worst_profile <= 0.03);
    try std.testing.expect(@abs(summary.last_d_ratio - 1.0) <= 0.06);
}

test "Sadowski 2013 section 4.4: full four-time Figure 9 reproduction" {
    if (!build_options.slow_tests) return error.SkipZigTest;
    const summary = try runPaperPulse(&.{ 3.0e3, 1.0e4, 3.0e4, 1.0e5 });

    // The profile comparison is the Figure 9 result.  The moment check is
    // independent of its normalization and directly gates D = 1/(3 chi).
    try std.testing.expect(summary.worst_profile <= 0.03);
    try std.testing.expect(summary.last_profile <= 0.01);
    try std.testing.expect(@abs(summary.last_d_ratio - 1.0) <= 0.02);
}
