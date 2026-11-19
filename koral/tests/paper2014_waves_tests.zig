//! Reproduction gate for Sadowski et al. (2014, MNRAS 439, 503), section 4.1,
//! Table 2 and Figure 3: the nine radiation-modified MHD linear waves.
//!
//! A uniform background (rho0 = 1, u0 = 9.13706e-3, i.e. c_s = 0.1) carries
//! one eigenmode of the linearized relativistic radiation-MHD system,
//! q = q0 + Re(dq exp(i(omega t - k x))), k = 2 pi on a periodic unit domain.
//! After one period t = 2 pi / Re(omega) the density is compared with the
//! analytic solution, the initial profile damped by exp(-Im(omega) t), in
//! the paper's L1 norm, and the convergence order with resolution is gated.
//! This is the only quantitative convergence result in either KORAL paper.
//!
//! Three rows of three modes: no radiation (sonic at B = 0, fast, slow),
//! optically thin (tau = 0.1, P = 0.1) and optically thick (tau = 10, P = 10).
//!
//! RESULT (linear minmod theta = 1.5 as in the paper, LAXF, RK2 IMEX, CFL
//! 0.5): all nine modes converge at second order. Measured L1, N = 64 -> 512:
//!   tau = 0    sonic 1.0e-8 -> 1.7e-10 (orders 1.95 2.00 2.00), fast, slow alike
//!   tau = 0.1  sonic 6.0e-9 -> 3.0e-10 at N = 256 (orders 2.2 2.1), then a floor
//!   tau = 10   sonic 1.2e-9 -> 1.9e-11 (orders 2.06 2.00 1.94), fast alike,
//!              slow 1.7e-9 -> 2.7e-11 (orders 2.22 2.00 1.76)
//! The paper reports FIRST order for the thick sonic and fast modes, its
//! source term being operator-split inside each RK2 stage; the IMEX
//! (Pareschi-Russo SSP2) coupling inherited from koral_lite keeps second
//! order there. The thin row reaches an amplitude-independent floor of
//! about 4e-10 (0.06 % of the wave) at N = 512; with the eigenmode scaled
//! x10 the same runs hold order 2.2 through N = 512, so the floor is
//! absolute noise, not truncation. The paper's own thin curves end at the
//! same level.
//!
//! THREE THINGS THIS TEST HAD TO ESTABLISH THAT THE PAPER DOES NOT STATE.
//! The first two are fixed by substituting the published eigenvectors into
//! the linearized equations (the `Table 2` test below).
//!
//!  1. The coupling is ABSORPTION, not scattering. The caption lists
//!     kappa_a = 0 and the text says tau = kappa_es, but the published dE,
//!     du are not adiabatic: they satisfy the gas energy equation
//!         i omega du - i k Gamma u0 du^x = kappa (dE - 4 E0 dT/T0)
//!     and the momentum equation
//!         i omega (rho0 + Gamma u0) du^x = i k dp + kappa dF^x
//!     with one and the same kappa = tau, to the table's printed precision.
//!     A pure-scattering medium exchanges no heat (du = Gamma u0 drho would
//!     be real) and damps ~100x more slowly than the published Im(omega).
//!     So kappa_abs = tau, kappa_es = 0, as in Jiang et al. (2012), whose
//!     test this follows. P is the pressure ratio p_rad / p_gas, E0 = 3 P p0,
//!     with the background in LTE (E0 = a T0^4, T = p / rho).
//!  2. One sign in the table is wrong: Im(dB^y) of the optically thick slow
//!     mode (see `dby_thick_slow_im`). As printed, that mode is not an
//!     eigenmode and its L1 error is 1.4e-7 at every resolution.
//!  3. C's gas-inversion tolerance is a noise floor for this problem. The
//!     1-D Newton solve for W = rho h gamma^2 starts from the previous
//!     step's primitives and returns them UNCHANGED when the relative
//!     residual is already below U2PCONV = 1e-10. The wave is a 1e-6
//!     perturbation and radiation sets the step to 1e-2 of a period, so
//!     most cells never update: the density is then noise at 5 % of the
//!     wave (amplitude ratios 0.94..1.06 with no trend in N), with the
//!     radiation decoupled or not. `FloorParams.u2pconv` (default: C's
//!     1e-10) is tightened to 1e-14 here. Production runs are unaffected
//!     at the 1e-10 level this implies, but it bounds what any small-
//!     amplitude test of the C code can show.
//!
//! PRODUCTION PPM. PUFFY runs PPM, not linear reconstruction. With PPM at
//! CFL 0.5 the no-radiation modes fall BELOW second order from N = 128 on
//! (orders 0.6-0.8, the limiter engaging on the weakly unstable PPM + RK2
//! pair); at CFL 0.1 they recover (amplitude error < 1e-5 at N = 256). The
//! radiative rows are unaffected because the radiative wavespeed already
//! holds the gas CFL near 0.1, which is also the regime PUFFY runs in.
//! Pinned by the `production PPM` test.

const std = @import("std");
const figdata = @import("../testing/figdata.zig");
const build_options = @import("build_options");
const config = @import("../config.zig");
const grid_mod = @import("../grid.zig");
const sim_mod = @import("../sim.zig");
const units_mod = @import("../units.zig");
const hydro = @import("../physics/hydro.zig");
const radiation = @import("../physics/radiation.zig");
const radforce = @import("../physics/radforce.zig");
const invert_rad = @import("../solve/invert_rad.zig");
const implicit = @import("../solve/implicit.zig");
const tubes = @import("../testing/tubes.zig");

const Grid = grid_mod.Grid;
const expect = std.testing.expect;
const C = std.math.Complex(f64);

fn c(re: f64, im: f64) C {
    return C.init(re, im);
}

const gam: f64 = 5.0 / 3.0;
const rho0: f64 = 1.0;
const uint0: f64 = 9.13706e-3;
const p0: f64 = (gam - 1.0) * uint0;
const b0: f64 = 0.100759;
const k_wave: f64 = 2.0 * std.math.pi;

pub const Mode = struct {
    name: []const u8,
    /// optical depth of the unit domain, = absorption opacity (see header)
    tau: f64,
    /// p_rad / p_gas of the background; 0 = no radiation module
    prat: f64,
    magnetized: bool,
    drho: C,
    du: C,
    dux: C,
    duy: C,
    dby: C,
    de: C,
    dfx: C,
    dfy: C,
    omega: C,
};

/// |Im dB^y| of the optically thick slow mode. The paper prints
/// dB^y = -8.03823e-8 + 3.03114e-7 i; the induction equation
/// omega dB^y = -k B0 (du^y - du^x), which every other magnetized mode of the
/// table satisfies to its printed precision, holds for this mode only with
/// the opposite sign of the imaginary part. With the printed sign the mode
/// is not an eigenmode and the L1 error stays at 1.4e-7, 20 % of the wave,
/// at every resolution. Pinned by the table test below.
const dby_thick_slow_im: f64 = 3.03114e-7;

/// Sadowski et al. (2014) Table 2, transcribed from the arXiv source
/// (1311.5900v1, article.tex lines 956-989), with the one correction above.
pub const modes = [_]Mode{
    // ---- tau = 0 ----
    .{ .name = "tau=0 sonic", .tau = 0, .prat = 0, .magnetized = false, .drho = c(1e-6, 0), .du = c(1.52284e-8, 0), .dux = c(1.0e-7, 0), .duy = c(0, 0), .dby = c(0, 0), .de = c(0, 0), .dfx = c(0, 0), .dfy = c(0, 0), .omega = c(0.628319, 0) },
    .{ .name = "tau=0 fast", .tau = 0, .prat = 0, .magnetized = true, .drho = c(1e-6, 0), .du = c(1.52284e-8, 0), .dux = c(1.60294e-7, 0), .duy = c(-9.79087e-8, 0), .dby = c(1.62303e-7, 0), .de = c(0, 0), .dfx = c(0, 0), .dfy = c(0, 0), .omega = c(1.00716, 0) },
    .{ .name = "tau=0 slow", .tau = 0, .prat = 0, .magnetized = true, .drho = c(1e-6, 0), .du = c(1.52284e-8, 0), .dux = c(6.17707e-8, 0), .duy = c(1.00118e-7, 0), .dby = c(-6.25516e-8, 0), .de = c(0, 0), .dfx = c(0, 0), .dfy = c(0, 0), .omega = c(0.388117, 0) },
    // ---- tau = 0.1, P = 0.1 ----
    .{ .name = "tau=0.1 sonic", .tau = 0.1, .prat = 0.1, .magnetized = false, .drho = c(1e-6, 0), .du = c(1.51557e-8, 7.69693e-10), .dux = c(9.97992e-8, 2.55207e-9), .duy = c(0, 0), .dby = c(0, 0), .de = c(1.33148e-13, 3.60017e-11), .dfx = c(-2.52471e-10, 7.40041e-11), .dfy = c(0, 0), .omega = c(0.627057, 0.0160351) },
    .{ .name = "tau=0.1 fast", .tau = 0.1, .prat = 0.1, .magnetized = true, .drho = c(1e-6, 0), .du = c(1.51984e-8, 4.81575e-10), .dux = c(1.60251e-7, 7.23831e-10), .duy = c(-9.79544e-8, 9.83679e-10), .dby = c(1.62344e-7, -8.96662e-10), .de = c(1.48421e-12, 6.06322e-11), .dfx = c(-3.95433e-10, 8.51051e-11), .dfy = c(2.36680e-10, 2.11182e-11), .omega = c(1.00689, 0.00454797) },
    .{ .name = "tau=0.1 slow", .tau = 0.1, .prat = 0.1, .magnetized = true, .drho = c(1e-6, 0), .du = c(1.50174e-8, 1.22299e-9), .dux = c(6.15333e-8, 1.83144e-9), .duy = c(9.89772e-8, 6.54186e-9), .dby = c(-6.14882e-8, -5.88315e-9), .de = c(1.91703e-13, 2.18721e-11), .dfx = c(-1.65181e-12, 7.17520e-11), .dfy = c(-2.23679e-10, -7.43141e-11), .omega = c(0.386625, 0.011507) },
    // ---- tau = 10, P = 10 ----
    .{ .name = "tau=10 sonic", .tau = 10, .prat = 10, .magnetized = false, .drho = c(1e-6, 0), .du = c(1.17070e-8, 1.88153e-9), .dux = c(2.66251e-7, 6.33514e-8), .duy = c(0, 0), .dby = c(0, 0), .de = c(2.05419e-7, 1.49859e-7), .dfx = c(-2.07308e-8, 3.77556e-8), .dfy = c(0, 0), .omega = c(1.6729, 0.398049) },
    .{ .name = "tau=10 fast", .tau = 10, .prat = 10, .magnetized = true, .drho = c(1e-6, 0), .du = c(1.17305e-8, 1.71290e-9), .dux = c(2.78499e-7, 5.23804e-8), .duy = c(-2.81093e-8, 6.25588e-9), .dby = c(1.10170e-7, -4.03337e-9), .de = c(2.07294e-7, 1.36364e-7), .dfx = c(-1.83331e-8, 3.63664e-8), .dfy = c(2.67581e-10, 1.24272e-9), .omega = c(1.74986, 0.329116) },
    .{ .name = "tau=10 slow", .tau = 10, .prat = 10, .magnetized = true, .drho = c(1e-6, 0), .du = c(9.46189e-9, 1.21376e-9), .dux = c(8.34269e-8, 1.20829e-8), .duy = c(1.13633e-7, 2.72697e-7), .dby = c(-8.03823e-8, -dby_thick_slow_im), .de = c(2.59666e-8, 9.67891e-8), .dfx = c(-1.98263e-8, 5.47610e-9), .dfy = c(3.66075e-9, -1.14750e-9), .omega = c(0.524187, 0.075919) },
};

fn e0Of(m: *const Mode) f64 {
    return 3.0 * m.prat * p0;
}

// ---- the table against the linearized equations ----------------------------

fn relResidual(lhs: C, rhs: C) f64 {
    return lhs.sub(rhs).magnitude() / @max(lhs.magnitude(), rhs.magnitude());
}

test "Sadowski 2014 Table 2: eigenmodes satisfy the linearized equations with absorption" {
    const i = c(0, 1);
    for (&modes) |*m| {
        // continuity, exact for every mode: omega drho = k rho0 du^x
        const cont = relResidual(m.omega.mul(m.drho), m.dux.mul(c(k_wave * rho0, 0)));
        try expect(cont <= 2e-5);
        // induction, exact at linear order: omega dB^y = -k B0 (du^y - du^x)
        if (m.magnetized) {
            const rhs_b = m.duy.sub(m.dux).mul(c(-k_wave * b0, 0));
            const ind = relResidual(m.omega.mul(m.dby), rhs_b);
            std.debug.print("table 2 [{s}]: induction {e:.1}\n", .{ m.name, ind });
            try expect(ind <= 2e-5);
            if (m.tau == 10 and m.duy.im > 1e-7) {
                // the sign as printed in the paper fails by O(1)
                const printed = c(m.dby.re, dby_thick_slow_im);
                try expect(relResidual(m.omega.mul(printed), rhs_b) >= 0.5);
            }
        }
        if (m.prat == 0) continue;

        const e0 = e0Of(m);
        const kap = c(m.tau, 0);
        // gas energy: i omega du - i k Gamma u0 du^x = kappa (dE - 4 E0 dT/T0),
        // dT/T0 = du/u0 - drho/rho0.
        const dtt = m.du.mul(c(1.0 / uint0, 0)).sub(m.drho.mul(c(1.0 / rho0, 0)));
        const lhs_e = i.mul(m.omega).mul(m.du).sub(i.mul(c(k_wave * gam * uint0, 0)).mul(m.dux));
        const rhs_abs = kap.mul(m.de.sub(dtt.mul(c(4.0 * e0, 0))));
        const res_e = relResidual(lhs_e, rhs_abs);

        // x-momentum (no magnetic term at B = 0; the magnetized modes add
        // the B^y pressure gradient): i omega (rho0 h0 + b_perp^2) du^x ...
        // gated only for the B = 0 modes where it has no field terms:
        //   i omega w0 du^x = i k dp + kappa dF^x,   w0 = rho0 + Gamma u0.
        var res_m: f64 = 0;
        if (!m.magnetized) {
            const w0 = rho0 + gam * uint0;
            const lhs_m = i.mul(m.omega).mul(m.dux).mul(c(w0, 0));
            const rhs_m = i.mul(c(k_wave * (gam - 1.0), 0)).mul(m.du).add(kap.mul(m.dfx));
            res_m = relResidual(lhs_m, rhs_m);
        }
        std.debug.print("table 2 [{s}]: continuity {e:.1}, gas energy {e:.1}, momentum {e:.1}\n", .{ m.name, cont, res_e, res_m });
        // The exchange term is a small difference of large terms in the thin
        // row, so the table's six digits leave a few 1e-3 there.
        try expect(res_e <= 5e-3);
        try expect(res_m <= 2e-5);

        // The scattering reading fails by construction: it has no heat
        // exchange, so the left-hand side would have to vanish.
        try expect(lhs_e.magnitude() >= 0.5 * rhs_abs.magnitude());
    }
}

// ---- evolution --------------------------------------------------------------

const RunOpts = struct {
    tsteplim: f64 = 0.5,
};

fn Harness(comptime with_rad: bool, comptime recon: config.Reconstruction) type {
    return struct {
        const cfg = config.Config{
            .modules = if (with_rad) &.{ .hydro, .mhd, .radiation } else &.{ .hydro, .mhd },
            .reconstruction = recon,
            .flux = .laxf,
            .timestepping = .rk2imex,
            .coords = .mink,
        };
        const SimT = sim_mod.Sim(cfg);
        const L = SimT.Layout;
        const NV = SimT.nv;

        fn part(z: C, phase: f64) f64 {
            // Re(z exp(-i phase)) at t = 0
            return z.re * @cos(phase) + z.im * @sin(phase);
        }

        /// L1 error of the density after one period, the paper's norm.
        fn run(a: std.mem.Allocator, m: *const Mode, nx: usize, ro: RunOpts) !f64 {
            const g = Grid.init(.{ .nx = nx, .ng = 3, .minx = 0, .maxx = 1 });
            const e0 = e0Of(m);
            var opt: SimT.Options = .{
                .phys = .{ .coords = .mink, .gam = gam },
                .bc = .{ .x = .periodic },
            };
            opt.phys.floors.u2pconv = 1.0e-14; // header, point 3
            opt.num.minmod_theta = 1.5; // the paper's limiter (linear only)
            opt.num.tsteplim = ro.tsteplim;
            if (with_rad) {
                // LTE background in paper units T = p / rho: a = E0 / T0^4.
                const t0 = p0 / rho0;
                const a_paper = e0 / (t0 * t0 * t0 * t0);
                const units = units_mod.Units.init(tubes.massForA(a_paper));
                opt.phys.opac = radforce.Params.grey(units, tubes.comp, m.tau, 0.0);
                opt.phys.rad = invert_rad.RadParams.cdefault;
                // Same reasoning as u2pconv: C's production Newton exits
                // (relative change 1e-6) would sit on top of a 1e-6 wave.
                opt.phys.implicit = implicit.ImplicitParams.cdefault;
                opt.phys.implicit.conv = 1.0e-15;
                opt.phys.implicit.convrel = 1.0e-14;
                opt.phys.implicit.convrelerr = 1.0e-15;
            }
            var s = try SimT.init(a, g, opt);
            defer s.deinit();

            const bx: f64 = if (m.magnetized) b0 else 0.0;
            var ix: i64 = 0;
            while (ix < s.nxi()) : (ix += 1) {
                const ph = k_wave * g.xc(ix);
                var pp: [NV]f64 = @splat(0);
                const rho = rho0 + part(m.drho, ph);
                const uint = uint0 + part(m.du, ph);
                pp[L.index(.rho)] = rho;
                pp[L.index(.uu)] = uint;
                pp[L.index(.vx)] = part(m.dux, ph);
                pp[L.index(.vy)] = part(m.duy, ph);
                pp[L.index(.entr)] = hydro.sFromU(rho, uint, gam);
                pp[L.index(.b1)] = bx;
                pp[L.index(.b2)] = bx + part(m.dby, ph);
                if (with_rad) {
                    // Fluid-frame moments -> rad-frame velocity relative to
                    // the gas (F = (4/3) E u_R to first order), then the
                    // code's own fluid -> lab conversion.
                    const ehat = e0 + part(m.de, ph);
                    pp[L.index(.ee)] = ehat;
                    pp[L.index(.fx)] = 0.75 * part(m.dfx, ph) / ehat;
                    pp[L.index(.fy)] = 0.75 * part(m.dfy, ph) / ehat;
                    const geom = s.core.cache.fillGeometry(ix, 0, 0);
                    pp = try radiation.pradFf2Lab(cfg, pp, &geom, opt.phys.rad);
                }
                try s.initCell(ix, 0, 0, pp);
            }
            try s.finishInit();

            const tend = 2.0 * std.math.pi / m.omega.re;
            while (s.t < tend) {
                var dt: ?f64 = null;
                if (s.t + 1.0 / s.core.tstepdenmax > tend) dt = tend - s.t;
                try s.step(dt);
            }

            const damp = @exp(-m.omega.im * tend);
            var l1: f64 = 0;
            ix = 0;
            while (ix < s.nxi()) : (ix += 1) {
                const exact = rho0 + damp * part(m.drho, k_wave * g.xc(ix));
                l1 += @abs(s.core.p.get(L.index(.rho), ix, 0, 0) - exact);
            }
            return l1 / @as(f64, @floatFromInt(nx));
        }
    };
}

fn runMode(a: std.mem.Allocator, m: *const Mode, comptime recon: config.Reconstruction, nx: usize, ro: RunOpts) !f64 {
    return if (m.prat == 0)
        Harness(false, recon).run(a, m, nx, ro)
    else
        Harness(true, recon).run(a, m, nx, ro);
}

const Ladder = struct {
    errs: [8]f64 = undefined,
    n: usize = 0,

    fn order(l: *const Ladder, r: usize) f64 {
        return std.math.log2(l.errs[r - 1] / l.errs[r]);
    }
    fn last(l: *const Ladder) f64 {
        return l.order(l.n - 1);
    }
};

fn ladder(m: *const Mode, comptime recon: config.Reconstruction, resolutions: []const usize, ro: RunOpts, fig: *figdata.Table) !Ladder {
    var l = Ladder{ .n = resolutions.len };
    for (resolutions, 0..) |nx, r| l.errs[r] = try runMode(std.testing.allocator, m, recon, nx, ro);
    fig.block("{s} | {s} | cfl {d}", .{ m.name, @tagName(recon), ro.tsteplim });
    for (resolutions, 0..) |nx, r| fig.row(&.{ @floatFromInt(nx), l.errs[r] });
    std.debug.print("paper 2014 waves [{s}, {s}, cfl {d}]: L1", .{ m.name, @tagName(recon), ro.tsteplim });
    for (resolutions, 0..) |nx, r| std.debug.print(" N={d}: {e:.3}", .{ nx, l.errs[r] });
    std.debug.print("; orders", .{});
    for (1..resolutions.len) |r| std.debug.print(" {d:.2}", .{l.order(r)});
    std.debug.print("\n", .{});
    return l;
}

/// One row of Table 2 in the paper's configuration. Every mode's error must
/// fall monotonically and the finest pair must be second order.
fn checkRow(first: usize, resolutions: []const usize, fig: *figdata.Table) !void {
    fig.meta("source", "Sadowski et al. 2014, section 4.1, Figure 3: L1 density error of the linear waves after one period", .{});
    fig.meta("columns", "NX L1", .{});
    var ok = true;
    for (0..3) |j| {
        const l = try ladder(&modes[first + j], .linear, resolutions, .{}, fig);
        for (1..l.n) |r| ok = ok and l.errs[r] < l.errs[r - 1];
        ok = ok and l.last() >= 1.7 and l.last() <= 2.8;
        // The wave is 1e-6 (mean |cos| = 6.4e-7): a wrong eigenmode or a
        // wrong reading of the table leaves an O(0.1) fraction as error
        // (1.4e-7 for the thick slow mode with the paper's printed sign).
        ok = ok and l.errs[l.n - 1] <= 1.5e-8;
    }
    try expect(ok); // after all three modes are printed
}

test "Sadowski 2014 section 4.1: tau = 0 sonic, fast, slow waves converge at second order" {
    var fig = figdata.Table.init("k14_waves_fast_tau0");
    defer fig.finish();
    try checkRow(0, &.{ 16, 32, 64 }, &fig);
}

test "Sadowski 2014 section 4.1: optically thin (tau = 0.1) waves converge at second order" {
    var fig = figdata.Table.init("k14_waves_fast_thin");
    defer fig.finish();
    try checkRow(3, &.{ 16, 32, 64 }, &fig);
}

test "Sadowski 2014 section 4.1: optically thick (tau = 10) waves converge at second order" {
    var fig = figdata.Table.init("k14_waves_fast_thick");
    defer fig.finish();
    try checkRow(6, &.{ 16, 32, 64 }, &fig);
}

test "Sadowski 2014 section 4.1: production PPM, order depends on the gas CFL" {
    if (!build_options.slow_tests) return error.SkipZigTest;
    const res = [_]usize{ 64, 128, 256 };
    var fig = figdata.Table.init("k14_waves_ppm");
    defer fig.finish();
    fig.meta("source", "Sadowski et al. 2014, section 4.1, Figure 3 variant with production PPM: L1 density error of the linear waves after one period", .{});
    fig.meta("columns", "NX L1", .{});
    // no radiation, CFL 0.5: below second order (measured 0.59, 0.84)
    const stalled = try ladder(&modes[0], .ppm, &res, .{}, &fig);
    try expect(stalled.last() <= 1.5);
    try expect(stalled.errs[2] < stalled.errs[0]);
    // same mode, CFL 0.1: recovered, and far more accurate at N = 256
    const fine = try ladder(&modes[0], .ppm, &res, .{ .tsteplim = 0.1 }, &fig);
    try expect(fine.last() >= 1.8);
    try expect(fine.errs[2] <= 0.2 * stalled.errs[2]);
    // optically thin sonic mode, CFL 0.5: radiation holds the gas CFL near
    // 0.1 and PPM is second order (measured 2.22, 1.70 with the floor near)
    const thin = try ladder(&modes[3], .ppm, &res, .{}, &fig);
    try expect(thin.order(1) >= 1.8);
    try expect(thin.errs[2] <= 1.0e-9);
}

test "Sadowski 2014 section 4.1: full Figure 3 resolution ladder" {
    if (!build_options.slow_tests) return error.SkipZigTest;
    const res = [_]usize{ 16, 32, 64, 128, 256, 512 };
    var fig = figdata.Table.init("k14_waves");
    defer fig.finish();
    fig.meta("source", "Sadowski et al. 2014, section 4.1, Figure 3: L1 density error of the linear waves after one period", .{});
    fig.meta("columns", "NX L1", .{});
    var ok = true;
    for (&modes) |*m| {
        const l = try ladder(m, .linear, &res, .{}, &fig);
        // second order through N = 256 everywhere
        for (3..5) |r| ok = ok and l.order(r) >= 1.8 and l.order(r) <= 2.8;
        // N = 512: still second order without radiation and when thick; the
        // thin row sits on its ~4e-10 absolute floor (header)
        if (m.tau == 0.1) {
            ok = ok and l.errs[5] <= 8.0e-10;
        } else {
            ok = ok and l.order(5) >= 1.7; // thick slow: 1.76
        }
    }
    try expect(ok);
}
