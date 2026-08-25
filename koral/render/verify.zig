//! Gold et al. 2020 (ApJ 897, 148) standardized GRRT verification scenes;
//! the EHT radiative-transfer code-comparison tests, implemented so this
//! imager sits on the same validation footing as BHOSS/ipole/RAPTOR/
//! GRTRANS/ODYSSEY/RAIKOU/VRT2.
//!
//! The five tests (their §3.2, Table 1) are fully analytic single-fluid
//! scenes on Kerr, in Boyer-Lindquist language:
//!
//!   n/n0 = exp{−½[(r/10)² + (h·cosθ)²]}          (eq 5; h=0 → sphere,
//!                                                  h=100/3 → razor disk)
//!   l    = l0·R^{3/2}/(1+R),  R = r·sinθ          (eq 6, q = 1/2)
//!   u_μ  = ū·(−1, 0, 0, l)  in BL,  ū from u·u=−1 (eqs 7-8)
//!   j_ν  = C·n·ν̂^{−α},  C·n0 = 3e−18 cgs         (eq 9, ν̂ ≡ ν/ν_p,
//!                                                  ν_p = 230 GHz)
//!   α_ν  = A·C·n·ν̂^{−(β+α)} per cm,  β = 5/2     (eq 11)
//!
//!   Test:      1      2      3      4       5
//!   A          0      0      0      1e5     1e6
//!   α         −3     −2      0      0       0
//!   h          0      0      10/3   10/3    100/3
//!   l0         0      1      1      1       1
//!   a          0.9    0      0.9    0.9     0.9
//!   S_exact    1.6465 1.4360 0.4418 0.2710  0.0255 Jy   (their Table 2)
//!
//! Geometry: M = 6e11 cm (~4e6 M☉), d = 2.4e22 cm (~7.78 kpc), camera at
//! r = 1000 M, i = 60°, fov 30 M × 30 M, spin axis up, stationary observer,
//! center ray k_φ = 0; all of which are ALREADY this renderer's
//! conventions (the tetrad's (t,φ,r,θ) Gram-Schmidt ordering is exactly
//! their "zero angular momentum at image center" condition). Rays stop
//! within 1e−4 M of the horizon or past the camera radius.
//!
//! The scene is specified in BL; the integrator runs in MKS2 (ingoing-KS
//! based). The only chart-dependent object is u_μ: with t_KS = t_BL +
//! ∫2Mr/Δ dr and φ_KS = φ_BL + ∫a/Δ dr,
//!     u^KS_t = −ū,   u^KS_r = ū(2Mr − a·l)/Δ,   u^KS_φ = ū·l,
//! (verified: u·u = −1 recovers exactly in KS), and x1 = ln(r − r0) gives
//! u_x1 = u^KS_r·(r − r0). n, l, ū are scalars of (r, θ); chart-free.
//! Near the horizon Δ → 0 makes ū → 0 while ū/Δ stays finite via
//! ū ∝ √Δ, so the medium is regular down to the capture radius.
//!
//! The transfer step mirrors traceRay's monochromatic branch (invariant
//! I_ν/ν̂³, dτ = α_ν·dl with dl = ν̂·h·M_cm the fluid-frame path in cm),
//! and the geodesic stepping IS render.zig's (rk4Step/flatPoleStep/
//! stepSize), so this validates the production integration scheme, not a
//! lookalike.

const std = @import("std");
const render = @import("render.zig");
const metric = @import("../metric/metric.zig");
const forms = @import("../metric/forms.zig");
const Dual3 = @import("../math/dual.zig").Dual3;
const grid_mod = @import("../grid.zig");

pub const TestDef = struct {
    /// absorption strength A
    a_abs: f64,
    /// spectral index α (j ∝ ν̂^{−α})
    alpha: f64,
    /// vertical concentration h
    h: f64,
    /// rotation switch l0
    l0: f64,
    /// BH spin a
    spin: f64,
    /// their Table 2 EXACT total flux [Jy]
    s_exact: f64,
};

pub const tests = [5]TestDef{
    .{ .a_abs = 0, .alpha = -3, .h = 0, .l0 = 0, .spin = 0.9, .s_exact = 1.6465 },
    .{ .a_abs = 0, .alpha = -2, .h = 0, .l0 = 1, .spin = 0.0, .s_exact = 1.4360 },
    .{ .a_abs = 0, .alpha = 0, .h = 10.0 / 3.0, .l0 = 1, .spin = 0.9, .s_exact = 0.4418 },
    .{ .a_abs = 1.0e5, .alpha = 0, .h = 10.0 / 3.0, .l0 = 1, .spin = 0.9, .s_exact = 0.2710 },
    .{ .a_abs = 1.0e6, .alpha = 0, .h = 100.0 / 3.0, .l0 = 1, .spin = 0.9, .s_exact = 0.0255 },
};

pub const beta = 2.5;
/// C·n0 [erg cm⁻³ s⁻¹ sr⁻¹ Hz⁻¹]
pub const cn0 = 3.0e-18;
/// BH mass as a length [cm] (~4e6 M☉)
pub const m_cm = 6.0e11;
/// source distance [cm] (~7.78 kpc)
pub const d_cm = 2.4e22;
pub const incl_deg = 60.0;
pub const fov_m = 30.0;
pub const r_cam = 1000.0;

/// Total flux [Jy] of an intensity image: F = ΣI·ΔΩ_pix, ΔΩ from the
/// paper's M and d.
pub fn totalFluxJy(img: []const f64, npix: usize) f64 {
    const pix_rad = fov_m * m_cm / d_cm / @as(f64, @floatFromInt(npix));
    var s: f64 = 0;
    for (img) |v| s += v;
    return s * pix_rad * pix_rad * 1.0e23;
}

const Local = struct { j: f64, chi: f64, nuhat: f64 };

/// Emissivity/absorptivity/frequency at MKS2 event x for photon k
/// (contravariant MKS2, camera-normalized). See file doc for the chart
/// transform of the BL-specified fluid.
pub fn medium(td: *const TestDef, mp: metric.MetricParams, x: [4]f64, k: [4]f64) Local {
    const r = @exp(x[1]) + mp.mksr0;
    const th = forms.mks2Theta(Dual3.constant(x[2]), mp.mksh0).v;
    const sinth = @sin(th);
    const costh = @cos(th);
    const a = td.spin;

    const zz = td.h * costh;
    const rr = r / 10.0;
    const n = @exp(-0.5 * (rr * rr + zz * zz));

    const bigr = r * sinth;
    const l = td.l0 * bigr * @sqrt(bigr) / (1.0 + bigr);

    // BL contravariant (t,φ) block for ū (eq 8)
    const sig = r * r + a * a * costh * costh;
    const del = r * r - 2.0 * r + a * a;
    const aa = (r * r + a * a) * (r * r + a * a) - a * a * del * sinth * sinth;
    const gtt = -aa / (sig * del);
    var den = gtt;
    if (l != 0) {
        const gtp = -2.0 * a * r / (sig * del);
        const gpp = (del - a * a * sinth * sinth) / (sig * del * sinth * sinth);
        den += -2.0 * gtp * l + gpp * l * l;
    }
    // u must be timelike (den < 0); the test parameters keep it so outside
    // the capture radius — guard anyway (vacuum on violation)
    if (!(den < 0)) return .{ .j = 0, .chi = 0, .nuhat = 1 };
    const ubar = @sqrt(-1.0 / den);

    // covariant u in MKS2 (see file doc)
    const u_t = -ubar;
    const u_x1 = ubar * (2.0 * r - a * l) / del * (r - mp.mksr0);
    const u_ph = ubar * l;
    const nuhat = -(k[0] * u_t + k[1] * u_x1 + k[3] * u_ph);
    if (!(nuhat > 1e-12)) return .{ .j = 0, .chi = 0, .nuhat = 1 };

    return .{
        .j = cn0 * n * std.math.pow(f64, nuhat, -td.alpha),
        .chi = td.a_abs * cn0 * n * std.math.pow(f64, nuhat, -(beta + td.alpha)),
        .nuhat = nuhat,
    };
}

pub const TraceOpts = struct {
    eps: f64 = 0.25,
    tau_max: f64 = 60.0,
    max_steps: usize = 200_000,
};

/// One backward ray through a Gold scene: the geodesic machinery is
/// render.zig's, the transfer step mirrors traceRay's monochromatic branch.
/// Returns I_ν at the camera [cgs].
pub fn traceGold(td: *const TestDef, mp: metric.MetricParams, g: *const grid_mod.Grid, x0: [4]f64, k0: [4]f64, opts: TraceOpts) f64 {
    const r_hor = metric.rHorizonBL(td.spin);
    const r_capture = r_hor + 1.0e-4; // their stop criterion
    const r_escape = 1.05 * r_cam;

    var x = x0;
    var k = k0;
    var intensity: f64 = 0;
    var tau: f64 = 0;
    var steps: usize = 0;
    while (steps < opts.max_steps) : (steps += 1) {
        const r = @exp(x[1]) + mp.mksr0;
        if (r < r_capture) break;
        if (r > r_escape) break;
        if (tau > opts.tau_max) break;

        const cd = metric.compute(.mks2, mp, x);
        const h = render.stepSize(g, x, k, opts.eps);
        var next = render.rk4Step(mp, &cd, x, k, -h);
        if (!next.ok) next = render.flatPoleStep(mp, &cd, x, k, -h);

        const med = medium(td, mp, x, k);
        if (med.j > 0 or med.chi > 0) {
            const dl = med.nuhat * h * m_cm; // fluid-frame path [cm]
            const nup = med.nuhat * med.nuhat * med.nuhat; // I_ν/ν̂³ invariant
            const dtau = med.chi * dl;
            const att = @exp(-tau);
            if (dtau > 1e-8) {
                intensity += att * (med.j / (med.chi * nup)) * (1.0 - @exp(-dtau));
            } else {
                intensity += att * (med.j / nup) * dl;
            }
            tau += dtau;
        }

        x = next.x;
        k = next.k;
    }
    return intensity;
}

/// Synthetic step-control grid for the analytic scenes (nothing is sampled
/// from it; stepSize only reads the spacings). Resolves the emitting cloud
/// (r ≲ 50) at production-like cell sizes.
pub fn goldGrid(mp: metric.MetricParams) grid_mod.Grid {
    return grid_mod.Grid.init(.{
        .nx = 192,
        .ny = 96,
        .nz = 1,
        .ng = 0,
        .minx = @log(1.2 - mp.mksr0),
        .maxx = @log(1.1e3 - mp.mksr0),
        .miny = 0.001,
        .maxy = 0.999,
        .minz = -std.math.pi / 4.0,
        .maxz = std.math.pi / 4.0,
    });
}

/// Render one Gold test into a npix×npix intensity map [cgs], ss×ss
/// box-averaged, threaded by row (deterministic: row-owned writes).
pub fn renderGold(td: *const TestDef, npix: usize, ss: usize, opts: TraceOpts, nthreads: usize, out: []f64) void {
    std.debug.assert(out.len == npix * npix);
    const mp = metric.MetricParams{ .a = td.spin, .mksr0 = 0.1, .mksh0 = 0.9 };
    const g = goldGrid(mp);
    var cam = render.Camera{
        .r = r_cam,
        .incl_deg = incl_deg,
        .fov = fov_m,
        .width = npix,
        .height = npix,
        .ss = @max(ss, 1),
    };
    cam.setup(mp);

    const Ctx = struct {
        td: *const TestDef,
        mp: metric.MetricParams,
        g: *const grid_mod.Grid,
        cam: *const render.Camera,
        out: []f64,
        next: *std.atomic.Value(usize),
        opts: TraceOpts,
        fn run(c: *const @This()) void {
            const ssn = @max(c.cam.ss, 1);
            const ssf: f64 = @floatFromInt(ssn);
            const inv = 1.0 / (ssf * ssf);
            while (true) {
                const py = c.next.fetchAdd(1, .monotonic);
                if (py >= c.cam.height) return;
                for (0..c.cam.width) |px| {
                    var acc: f64 = 0;
                    for (0..ssn) |sy| {
                        for (0..ssn) |sx| {
                            const fx = @as(f64, @floatFromInt(px)) + (@as(f64, @floatFromInt(sx)) + 0.5) / ssf;
                            const fy = @as(f64, @floatFromInt(py)) + (@as(f64, @floatFromInt(sy)) + 0.5) / ssf;
                            acc += traceGold(c.td, c.mp, c.g, c.cam.x0, c.cam.rayAt(fx, fy), c.opts);
                        }
                    }
                    c.out[py * c.cam.width + px] = acc * inv;
                }
            }
        }
    };
    var next: std.atomic.Value(usize) = .init(0);
    const ctx = Ctx{ .td = td, .mp = mp, .g = &g, .cam = &cam, .out = out, .next = &next, .opts = opts };

    var threads: [63]std.Thread = undefined;
    const want: usize = @min(@max(nthreads, 1) - 1, threads.len);
    var spawned: usize = 0;
    for (0..want) |i| {
        threads[i] = std.Thread.spawn(.{}, Ctx.run, .{&ctx}) catch break;
        spawned = i + 1;
    }
    Ctx.run(&ctx);
    for (threads[0..spawned]) |t| t.join();
}
