//! Reproduction gates for Sadowski et al. (2013), section 4.4: the optically
//! thick planar pulse (Figure 9) and, at the end of the file, the optically
//! thin point-like pulse in 3D (Figure 8).
//!
//! This uses the paper's 101-cell domain, temperature profile, radiative
//! constant, scattering opacity, and four plotted output times.  The exact
//! diffusion profile is evaluated analytically by expanding
//! (1 + 100 exp(-x^2/w^2))^4 into four Gaussian terms.  The current code uses
//! its production PPM, M1, and RK2IMEX choices rather than the paper's
//! reconstruction/RK3 combination.

const std = @import("std");
const figdata = @import("../testing/figdata.zig");
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

fn runPaperPulse(output_times: []const f64, fig: *figdata.Table) !RunSummary {
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

    fig.meta("source", "Sadowski et al. 2013, section 4.4, Figure 9: optically thick pulse, 101 cells", .{});
    fig.meta("columns", "x E/E_background E_exact/E_background", .{});
    fig.block("t = 0", .{});
    ix = 0;
    while (ix < sim.nxi()) : (ix += 1) {
        const e = sim.core.p.get(L.index(.ee), ix, 0, 0) / e_background_sim;
        fig.row(&.{ grid.xc(ix), e, e });
    }

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
        fig.block("t = {d}", .{target});
        ix = 0;
        while (ix < sim.nxi()) : (ix += 1) {
            fig.row(&.{
                grid.xc(ix),
                sim.core.p.get(L.index(.ee), ix, 0, 0) / e_background_sim,
                exactEnergy(grid.xc(ix), target) / e_background_sim,
            });
        }
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
    var fig = figdata.Table.init("k13_pulse_thick_fast");
    defer fig.finish();
    const summary = try runPaperPulse(&.{3.0e3}, &fig);

    // The first published output is the most sensitive to truncation error.
    // Measured with the current PPM path: profile L1 = 0.0118 and
    // D/D_exact = 0.9735 in both Debug and ReleaseFast.
    try std.testing.expect(summary.worst_profile <= 0.03);
    try std.testing.expect(@abs(summary.last_d_ratio - 1.0) <= 0.06);
}

test "Sadowski 2013 section 4.4: full four-time Figure 9 reproduction" {
    if (!build_options.slow_tests) return error.SkipZigTest;
    var fig = figdata.Table.init("k13_pulse_thick");
    defer fig.finish();
    const summary = try runPaperPulse(&.{ 3.0e3, 1.0e4, 3.0e4, 1.0e5 }, &fig);

    // The profile comparison is the Figure 9 result.  The moment check is
    // independent of its normalization and directly gates D = 1/(3 chi).
    try std.testing.expect(summary.worst_profile <= 0.03);
    try std.testing.expect(summary.last_profile <= 0.01);
    try std.testing.expect(@abs(summary.last_d_ratio - 1.0) <= 0.02);
}

// ---- optically thin pulse in 3D (Figure 8) ----------------------------------
//
// The same temperature profile, now a point-like Gaussian
//   T_rad = T0 (1 + 100 exp(-(x^2 + y^2 + z^2) / w^2))
// at the centre of a 51^3 Cartesian box [-50, 50]^3 (the extent is read off
// the axes of the paper's Figure 8; the text gives only the cell count), with
// kappa = 0 and a negligible kappa_es = 1e-6. Linear reconstruction with
// minmod theta = 2 as in the paper. The pulse must expand isotropically at
// the speed of light with its energy density falling as 1 / r^2.
//
// The paper shows this by eye (an expanding circle and a 1/x^2 guide line).
// The gates here are the same two statements as numbers, plus the
// conservation law behind the second one:
//   * the shell on the x axis moves at dr/dt = 1 between the paper's two
//     output times t = 15 and t = 35,
//   * its peak falls as r^-2 between them,
//   * the total radiation energy in the box is conserved while the shell is
//     inside it,
//   * the peak heights agree with the paper's Figure 8, digitised from its
//     EPS sources, within a factor of two,
//   * the Cartesian-grid anisotropy (axis against body diagonal at equal
//     radius) is measured and bounded.
// Differences from the paper's run: RK2 here, RK3 there.

const cfg_thin = config.Config{
    .modules = &.{ .hydro, .radiation },
    .reconstruction = .linear,
    .flux = .laxf,
    .timestepping = .rk2imex,
    .coords = .mink,
};
const SimThin = sim_mod.Sim(cfg_thin);
const LT = SimThin.Layout;
const kappa_es_thin: f64 = 1.0e-6;

const Shell = struct {
    /// radius and height of the energy-density maximum on the +x axis
    r_peak: f64,
    e_peak: f64,
    /// conserved radiation energy summed over the box (lab frame, -R^t_t)
    energy: f64,
    /// excess energy density at radius r_peak on the axis / on the body
    /// diagonal (linear interpolation along each line)
    axis_over_diag: f64,
};

/// Lab-frame radiation energy density above the background. The primitive
/// `ee` is the energy density in the RADIATION rest frame, which for a
/// streaming shell (u_R ~ 14 here) is smaller by ~gamma_R^2; the paper plots
/// the lab-frame E = -R^t_t, which is the conserved variable.
fn excessAt(sim: *const SimThin, ix: i64, iy: i64, iz: i64) f64 {
    return -sim.core.u.get(LT.index(.ee), ix, iy, iz) - e_background_sim;
}

fn measureShell(sim: *const SimThin, n: usize) Shell {
    const g = &sim.core.grid;
    const mid: i64 = @intCast(n / 2);
    const ni: i64 = @intCast(n);

    // peak on the +x axis, refined with a parabola through three cells
    var best: i64 = mid;
    var ix: i64 = mid;
    while (ix < ni) : (ix += 1) {
        if (excessAt(sim, ix, mid, mid) > excessAt(sim, best, mid, mid)) best = ix;
    }
    var r_peak = g.xc(best);
    var e_peak = excessAt(sim, best, mid, mid);
    if (best > mid and best < ni - 1) {
        const em = excessAt(sim, best - 1, mid, mid);
        const ep = excessAt(sim, best + 1, mid, mid);
        const den = em - 2.0 * e_peak + ep;
        if (den < 0) {
            const off = 0.5 * (em - ep) / den;
            r_peak += off * g.dx;
            e_peak -= 0.25 * (em - ep) * off;
        }
    }

    var energy: f64 = 0;
    var iz: i64 = 0;
    while (iz < ni) : (iz += 1) {
        var iy: i64 = 0;
        while (iy < ni) : (iy += 1) {
            ix = 0;
            while (ix < ni) : (ix += 1) energy += -sim.core.u.get(LT.index(.ee), ix, iy, iz);
        }
    }

    // the body diagonal reaches radius r_peak at index offset r_peak / (sqrt 3 dx)
    const fd = r_peak / (@sqrt(3.0) * g.dx);
    const j0: i64 = @intFromFloat(@floor(fd));
    const fr = fd - @floor(fd);
    const d0 = excessAt(sim, mid + j0, mid + j0, mid + j0);
    const d1 = excessAt(sim, mid + j0 + 1, mid + j0 + 1, mid + j0 + 1);
    const e_diag = (1.0 - fr) * d0 + fr * d1;

    return .{ .r_peak = r_peak, .e_peak = e_peak, .energy = energy, .axis_over_diag = e_peak / e_diag };
}

fn runThinPulse(n: usize, output_times: []const f64, out: []Shell, fig: *figdata.Table, map: *figdata.Table) !void {
    const mass = paperMass();
    const units = units_mod.Units.init(mass);
    const comp = thermo.Composition.cdefault;
    const opac = radforce.Params.grey(units, comp, 0.0, kappa_es_thin / state_scale);

    var rad = invert_rad.RadParams.cdefault;
    rad.eerhoratiomin = 1.0e-50;
    rad.eeuuratiomin = 1.0e-50;

    const grid = Grid.init(.{
        .nx = n,
        .ny = n,
        .nz = n,
        .ng = 2,
        .minx = -50.0,
        .maxx = 50.0,
        .miny = -50.0,
        .maxy = 50.0,
        .minz = -50.0,
        .maxz = 50.0,
    });
    var sim = try SimThin.init(std.testing.allocator, grid, .{
        .phys = .{ .coords = .mink, .gam = gam, .rad = rad, .opac = opac },
        .num = .{ .minmod_theta = 2.0 },
        .bc = .{ .x = .copy, .y = .copy, .z = .copy },
    });
    defer sim.deinit();

    const gas_u = thermo.uFromTrho(&opac.consts, t0, rho_sim, gam);
    var iz: i64 = 0;
    while (iz < sim.nzi()) : (iz += 1) {
        var iy: i64 = 0;
        while (iy < sim.nyi()) : (iy += 1) {
            var ix: i64 = 0;
            while (ix < sim.nxi()) : (ix += 1) {
                const x = grid.xc(ix);
                const y = grid.yc(iy);
                const z = grid.zc(iz);
                var pp: [SimThin.nv]f64 = @splat(0);
                pp[LT.index(.rho)] = rho_sim;
                pp[LT.index(.uu)] = gas_u;
                pp[LT.index(.entr)] = hydro.sFromU(rho_sim, gas_u, gam);
                pp[LT.index(.ee)] = initialEnergy(@sqrt(x * x + y * y + z * z));
                try sim.initCell(ix, iy, iz, pp);
            }
        }
    }
    try sim.finishInit();

    for (output_times, 0..) |target, k| {
        while (sim.t < target) {
            var dt: ?f64 = null;
            const cfl_dt = 1.0 / sim.core.tstepdenmax;
            if (sim.t + cfl_dt > target) dt = target - sim.t;
            try sim.step(dt);
        }
        out[k] = measureShell(&sim, n);
        // figure data, in units of the initial central lab-frame energy
        // density: the profile along y = z = 0 and the z = 0 plane
        const e_centre = initialEnergy(0.0);
        const mid: i64 = @intCast(n / 2);
        fig.block("t = {d}", .{target});
        map.block("t = {d}", .{target});
        var jx: i64 = 0;
        while (jx < sim.nxi()) : (jx += 1) {
            fig.row(&.{ grid.xc(jx), -sim.core.u.get(LT.index(.ee), jx, mid, mid) / e_centre });
        }
        var jy: i64 = 0;
        while (jy < sim.nyi()) : (jy += 1) {
            jx = 0;
            while (jx < sim.nxi()) : (jx += 1) {
                map.row(&.{ grid.xc(jx), grid.yc(jy), -sim.core.u.get(LT.index(.ee), jx, jy, mid) / e_centre });
            }
        }
        std.debug.print(
            "paper 2013 thin pulse N={d}^3: t={d:.0}, shell r={d:.3} E_peak={e:.4}, total E={e:.8}, axis/diagonal={d:.3}\n",
            .{ n, target, out[k].r_peak, out[k].e_peak, out[k].energy, out[k].axis_over_diag },
        );
    }
}

/// Peak lab-frame energy density on the x axis in Figure 8 (bottom panels),
/// relative to the initial central value 6.5e-32, digitised from the paper's
/// EPS sources (plot_p3d_02/03.eps): 2.1e-34 at x ~ 18 (t = 15) and 6.1e-35
/// at x ~ 38 (t = 35), i.e. the paper's own shell falls as r^-1.8.
const fig8_peak_t15: f64 = 2.13e-34 / 6.49e-32;
const fig8_peak_t35: f64 = 6.15e-35 / 6.49e-32;

fn checkThinPulse(n: usize) !void {
    var sh: [3]Shell = undefined;
    var fig = figdata.Table.init("k13_pulse_thin_axis");
    defer fig.finish();
    fig.meta("source", "Sadowski et al. 2013, section 4.4, Figure 8 (bottom): optically thin pulse along y = z = 0", .{});
    fig.meta("columns", "x E_lab/E_lab(centre, t=0)", .{});
    fig.meta("paper_peaks", "t=15: x {d} E {e}; t=35: x {d} E {e}  (digitised from plot_p3d_02/03.eps, same units)", .{ 17.6, fig8_peak_t15, 37.3, fig8_peak_t35 });
    var map = figdata.Table.init("k13_pulse_thin_map");
    defer map.finish();
    map.meta("source", "Sadowski et al. 2013, section 4.4, Figure 8 (top): optically thin pulse in the z = 0 plane", .{});
    map.meta("columns", "x y E_lab/E_lab(centre, t=0)", .{});
    map.meta("shape", "{d} {d}", .{ n, n });
    try runThinPulse(n, &.{ 0.0, 15.0, 35.0 }, &sh, &fig, &map);

    const speed = (sh[2].r_peak - sh[1].r_peak) / 20.0;
    const slope = @log(sh[2].e_peak / sh[1].e_peak) / @log(sh[2].r_peak / sh[1].r_peak);
    const de = @abs(sh[2].energy / sh[0].energy - 1.0);
    const rel15 = sh[1].e_peak / sh[0].e_peak;
    const rel35 = sh[2].e_peak / sh[0].e_peak;
    std.debug.print(
        "paper 2013 thin pulse N={d}^3: shell speed {d:.4} c, peak ~ r^{d:.3}, energy drift {e:.2}; peak/initial {e:.3} (paper {e:.3}) and {e:.3} (paper {e:.3})\n",
        .{ n, speed, slope, de, rel15, fig8_peak_t15, rel35, fig8_peak_t35 },
    );
    // Measured at 51^3: speed 1.048 c (the peak is located to about half a
    // cell, 1 of the 20 travelled), slope -2.28, drift 1.1e-7, peak ratios
    // 4.3e-3 and 6.1e-4 against the paper's 3.3e-3 and 9.5e-4.
    try std.testing.expect(@abs(speed - 1.0) <= 0.08);
    try std.testing.expect(slope <= -1.6 and slope >= -2.6);
    try std.testing.expect(de <= 1.0e-6);
    try std.testing.expect(rel15 / fig8_peak_t15 >= 0.5 and rel15 / fig8_peak_t15 <= 2.0);
    try std.testing.expect(rel35 / fig8_peak_t35 >= 0.5 and rel35 / fig8_peak_t35 <= 2.0);
    // Cartesian-grid anisotropy at equal radius: 1.04 at t = 15, 0.90 at
    // t = 35 (the paper reports the axes brighter; here it is within 10 %
    // either way).
    for (sh[1..]) |x| try std.testing.expect(x.axis_over_diag >= 0.8 and x.axis_over_diag <= 1.25);
}

test "Sadowski 2013 section 4.4: thin 3D pulse expands at c and falls as 1/r^2 (Figure 8)" {
    if (!build_options.slow_tests) return error.SkipZigTest;
    try checkThinPulse(51);
}
