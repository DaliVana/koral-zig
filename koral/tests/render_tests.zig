//! Theory gates for the GRRT renderer (render/render.zig + render/image.zig):
//!
//!  * camera tetrad orthonormality in Kerr,
//!  * null-geodesic integration: E = −k_t and L = k_φ conservation and the
//!    null norm through a strong-field passage (Kerr a = 0.9375, MKS2),
//!  * the Schwarzschild photon-capture threshold b_crit = √27 M — rays
//!    inside are swallowed, outside escape,
//!  * LTE consistency of the emission model: source function j/χ = B(T)
//!    when gas and radiation are in equilibrium,
//!  * φ-wedge periodic sampling,
//!  * the PNG encoder (chunk CRCs, zlib stored blocks, pixel roundtrip)
//!    and the afmhot colormap,
//!  * SLOW LIGHT (series/sweep/adaptive): time-lerp + edge clamping of the
//!    window sampler; static-series identity BIT-FOR-BIT through both
//!    traceRayWith and the full 3-phase sweep; a time-localized flare
//!    arriving retarded by the Shapiro-corrected travel time; the radial
//!    KS flight time Δt = Δr + 4M ln(...); consecutive photon-ring windings
//!    delayed by the photon-orbit period 2π·3√3 M; and the adaptive
//!    quadtree plan (exact pixel weights, capture-boundary marking,
//!    antialiased shadow edge).

const std = @import("std");
const render = @import("../render/render.zig");
const image = @import("../render/image.zig");
const metric = @import("../metric/metric.zig");
const forms = @import("../metric/forms.zig");
const grid_mod = @import("../grid.zig");
const config = @import("../config.zig");
const layout = @import("../layout.zig");
const thermo = @import("../physics/thermo.zig");
const opacities = @import("../physics/opacities.zig");
const units_mod = @import("../units.zig");
const Dual3 = @import("../math/dual.zig").Dual3;

const cfg = config.puffy;
const L = layout.VarLayout(cfg);
const pi = std.math.pi;

// ---- helpers ---------------------------------------------------------------

fn puffyGrid(nx: usize, ny: usize, nz: usize, mp: metric.MetricParams, rmin: f64, rmax: f64) grid_mod.Grid {
    return grid_mod.Grid.init(.{
        .nx = nx,
        .ny = ny,
        .nz = nz,
        .ng = 0,
        .minx = @log(rmin - mp.mksr0),
        .maxx = @log(rmax - mp.mksr0),
        .miny = 0.001,
        .maxy = 0.999,
        .minz = -pi / 4.0,
        .maxz = pi / 4.0,
    });
}

fn zeroDump(allocator: std.mem.Allocator, nx: usize, ny: usize, nz: usize) !render.DumpData {
    const n = nx * ny * nz * L.count;
    const body = try allocator.alloc(f64, n);
    @memset(body, 0);
    return .{
        .header = .{ .nx = @intCast(nx), .ny = @intCast(ny), .nz = @intCast(nz), .nv = @intCast(L.count), .t = 0, .nstep = 0, .out_idx = 0 },
        .body = body,
    };
}

fn testConsts() thermo.Consts {
    return thermo.Consts.init(units_mod.Units.init(10.0), thermo.Composition.puffy);
}

/// Launch a ray from a static observer at (r_cam, θ) with local arrival
/// direction d = (d1 outward-radial, d2 equatorward, d3 azimuthal).
fn launch(mp: metric.MetricParams, r_cam: f64, theta: f64, d_in: [3]f64) struct { x: [4]f64, k: [4]f64 } {
    const x0 = [4]f64{ 0, @log(r_cam - mp.mksr0), forms.mks2X2FromTheta(theta, mp.mksh0), 0 };
    const cd = metric.compute(.mks2, mp, x0);
    const e = render.tetradFromMetric(cd.gcov);
    var d = d_in;
    const n = @sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
    for (&d) |*v| v.* /= n;
    var k: [4]f64 = undefined;
    for (0..4) |mu| k[mu] = e[0][mu] + d[0] * e[1][mu] + d[1] * e[2][mu] + d[2] * e[3][mu];
    return .{ .x = x0, .k = k };
}

fn kcovAt(mp: metric.MetricParams, x: [4]f64, k: [4]f64) [4]f64 {
    const cd = metric.compute(.mks2, mp, x);
    var kc: [4]f64 = @splat(0);
    for (0..4) |i| {
        for (0..4) |j| kc[i] += cd.gcov[i][j] * k[j];
    }
    return kc;
}

fn knormAt(mp: metric.MetricParams, x: [4]f64, k: [4]f64) f64 {
    const kc = kcovAt(mp, x, k);
    var s: f64 = 0;
    for (0..4) |i| s += kc[i] * k[i];
    return s;
}

// ---- tetrad ----------------------------------------------------------------

test "render: camera tetrad is orthonormal in Kerr (a=0.9375)" {
    const mp = metric.MetricParams{ .a = 0.9375, .mksr0 = 0.1, .mksh0 = 0.9 };
    const x = [4]f64{ 0, @log(1000.0 - mp.mksr0), forms.mks2X2FromTheta(pi / 3.0, mp.mksh0), 0.1 };
    const cd = metric.compute(.mks2, mp, x);
    const e = render.tetradFromMetric(cd.gcov);
    for (0..4) |a| {
        for (0..4) |b| {
            var s: f64 = 0;
            for (0..4) |i| {
                for (0..4) |j| s += cd.gcov[i][j] * e[a][i] * e[b][j];
            }
            const want: f64 = if (a != b) 0.0 else if (a == 0) -1.0 else 1.0;
            try std.testing.expectApproxEqAbs(want, s, 1e-10);
        }
    }
}

// ---- geodesics -------------------------------------------------------------

test "render: Kerr null geodesic conserves E, L and stays null through the strong field" {
    const allocator = std.testing.allocator;
    const mp = metric.MetricParams{ .a = 0.9375, .mksr0 = 0.1, .mksh0 = 0.9 };
    var data = try zeroDump(allocator, 64, 32, 4);
    defer allocator.free(data.body);
    const g = puffyGrid(64, 32, 4, mp, 1.25, 1000.0);
    const consts = testConsts();
    const scene = render.Scene.init(g, mp, consts, opacities.Channels.puffy, 5.0 / 3.0, &data, 1000.0, 1000.0);

    // aimed to pass within a few M of the hole, off the equator
    const ray = launch(mp, 1000.0, pi / 3.0, .{ 1.0, 0.002, -0.005 });
    const kc0 = kcovAt(mp, ray.x, ray.k);
    try std.testing.expectApproxEqAbs(@as(f64, 0), knormAt(mp, ray.x, ray.k), 1e-12);

    const res = render.traceRay(cfg, &scene, ray.x, ray.k, .{ .eps = 0.1 });
    try std.testing.expect(res.steps > 100);

    const kc1 = kcovAt(mp, res.x, res.k);
    // E = −k_t, L = k_φ (stationary + axisymmetric metric)
    try std.testing.expectApproxEqRel(kc0[0], kc1[0], 1e-5);
    try std.testing.expectApproxEqRel(kc0[3], kc1[3], 1e-5);
    // null norm, relative to E²
    try std.testing.expect(@abs(knormAt(mp, res.x, res.k)) / (kc0[0] * kc0[0]) < 1e-4);
}

test "render: E and L survive a polar-axis crossing (the meridian-artifact rays)" {
    const allocator = std.testing.allocator;
    const mp = metric.MetricParams{ .a = 0.9375, .mksr0 = 0.1, .mksh0 = 0.9 };
    var data = try zeroDump(allocator, 64, 32, 4);
    defer allocator.free(data.body);
    const g = puffyGrid(64, 32, 4, mp, 1.25, 1000.0);
    const scene = render.Scene.init(g, mp, testConsts(), opacities.Channels.puffy, 5.0 / 3.0, &data, 1000.0, 1000.0);

    // near-face-on camera: rays at a few M of impact parameter travel almost
    // along the spin axis behind the hole and must cross the x2 = 0 pole.
    // eps = 0.1: RK4 truncation on these strong-field near-polar paths
    // scales ~h⁵ (measured: 2.9e-3 at eps 0.25 → 4.5e-7 at 0.05), so the
    // norm tolerance below carries a ~10× margin over truncation while
    // still catching structural errors (the old stage-mixing pole bug sat
    // orders of magnitude above it).
    const theta_cam = 0.06; // rad from the axis
    for ([_]f64{ 6.0, 9.0, 14.0 }) |b| {
        const alpha = std.math.asin(b / 1000.0);
        const ray = launch(mp, 1000.0, theta_cam, .{ @cos(alpha), @sin(alpha), 0 });
        const kc0 = kcovAt(mp, ray.x, ray.k);
        const res = render.traceRay(cfg, &scene, ray.x, ray.k, .{ .eps = 0.1 });
        const kc1 = kcovAt(mp, res.x, res.k);
        try std.testing.expectApproxEqRel(kc0[0], kc1[0], 2e-5);
        // L may be ~0 for a polar ray — compare absolutely against E
        try std.testing.expect(@abs(kc1[3] - kc0[3]) < 2e-5 * @abs(kc0[0]));
        try std.testing.expect(@abs(knormAt(mp, res.x, res.k)) / (kc0[0] * kc0[0]) < 3e-4);
    }
}

test "render: Schwarzschild photon capture threshold b_crit = sqrt(27) M" {
    const allocator = std.testing.allocator;
    const mp = metric.MetricParams{ .a = 0.0, .mksr0 = 0.1, .mksh0 = 0.9 };
    var data = try zeroDump(allocator, 64, 32, 1);
    defer allocator.free(data.body);
    const g = puffyGrid(64, 32, 1, mp, 1.85, 1000.0);
    const consts = testConsts();
    const scene = render.Scene.init(g, mp, consts, opacities.Channels.puffy, 5.0 / 3.0, &data, 2000.0, 1000.0);

    const r_cam = 2000.0;
    // b_crit = 3√3 ≈ 5.196: b = 4.5 must fall in, b = 6 must fly by.
    for ([_]struct { b: f64, captured: bool }{
        .{ .b = 4.5, .captured = true },
        .{ .b = 6.0, .captured = false },
    }) |case| {
        const alpha = std.math.asin(case.b / r_cam);
        const ray = launch(mp, r_cam, pi / 2.0, .{ @cos(alpha), 0, -@sin(alpha) });
        // impact parameter of the launched ray: b = L/E = k_φ / (−k_t)
        const kc = kcovAt(mp, ray.x, ray.k);
        const b_measured = @abs(kc[3] / kc[0]);
        try std.testing.expectApproxEqRel(case.b, b_measured, 5e-3);

        const res = render.traceRay(cfg, &scene, ray.x, ray.k, .{ .eps = 0.25 });
        try std.testing.expectEqual(case.captured, res.captured);
        if (!case.captured) {
            const r_end = @exp(res.x[1]) + mp.mksr0;
            try std.testing.expect(r_end > 1000.0);
        }
    }
}

// ---- emission model --------------------------------------------------------

test "render: LTE source function j/chi equals the Planck function B(T)" {
    const mp = metric.MetricParams{ .a = 0.0, .mksr0 = 0.1, .mksh0 = 0.9 };
    var body = [_]f64{0} ** L.count;
    var data = render.DumpData{
        .header = .{ .nx = 1, .ny = 1, .nz = 1, .nv = @intCast(L.count), .t = 0, .nstep = 0, .out_idx = 0 },
        .body = &body,
    };
    const g = puffyGrid(1, 1, 1, mp, 1.85, 1000.0);
    const consts = testConsts();
    const scene = render.Scene.init(g, mp, consts, opacities.Channels.puffy, 5.0 / 3.0, &data, 1000.0, 1000.0);

    // gas at rest, B = 0, radiation in LTE with the gas: Ê = 4σT⁴
    // (temperatures are in Kelvin throughout thermo, the C convention)
    const tK = 1.0e7;
    const rho = 1.0e-6;
    const uu = thermo.uFromTrho(&consts, tK, rho, 5.0 / 3.0);

    var pp = [_]f64{0} ** L.count;
    pp[L.index(.rho)] = rho;
    pp[L.index(.uu)] = uu;
    pp[L.index(.ee)] = consts.lteEfromT(tK);

    const x = [4]f64{ 0, @log(20.0 - mp.mksr0), 0.5, 0 };
    const cd = metric.compute(.mks2, mp, x);
    const geom = render.geomFromCoordData(x, &cd);
    const st = render.localState(cfg, &scene, &geom, pp) orelse return error.TestUnexpectedResult;

    // total gray emissivity as traceRay assembles it: thermal + scattering
    // source with the dipole factor, which is exactly 1 here (F̂ = 0 for an
    // isotropic field around gas at rest — also assert that).
    for (st.fhat) |f| try std.testing.expectApproxEqAbs(@as(f64, 0), f, 1e-18);
    const jem = st.j_therm + st.chi_es * (st.ehat / consts.fourmpi);
    const planck = consts.sigma_rad_over_pi * tK * tK * tK * tK;
    try std.testing.expect(st.chi > 0);
    try std.testing.expectApproxEqRel(planck, jem / st.chi, 1e-10);
}

test "render: fluid-frame flux gives a dipole boost along the flux direction" {
    const mp = metric.MetricParams{ .a = 0.0, .mksr0 = 0.1, .mksh0 = 0.9 };
    var body = [_]f64{0} ** L.count;
    var data = render.DumpData{
        .header = .{ .nx = 1, .ny = 1, .nz = 1, .nv = @intCast(L.count), .t = 0, .nstep = 0, .out_idx = 0 },
        .body = &body,
    };
    const g = puffyGrid(1, 1, 1, mp, 1.85, 1000.0);
    const consts = testConsts();
    const scene = render.Scene.init(g, mp, consts, opacities.Channels.puffy, 5.0 / 3.0, &data, 1000.0, 1000.0);

    const x = [4]f64{ 0, @log(20.0 - mp.mksr0), 0.5, 0 };
    const cd = metric.compute(.mks2, mp, x);
    const geom = render.geomFromCoordData(x, &cd);

    // gas at rest, radiation frame moving radially outward → F̂ along +x1
    const tK = 1.0e7;
    var pp = [_]f64{0} ** L.count;
    pp[L.index(.rho)] = 1.0e-6;
    pp[L.index(.uu)] = thermo.uFromTrho(&consts, tK, 1.0e-6, 5.0 / 3.0);
    pp[L.index(.ee)] = consts.lteEfromT(tK);
    pp[L.index(.fx)] = 0.3; // radiative VELR primitive: rad frame boosted in +x1
    const st = render.localState(cfg, &scene, &geom, pp) orelse return error.TestUnexpectedResult;
    try std.testing.expect(st.fhat[1] > 0);
    // F̂ ⊥ u (with the actual gas frame — VELR = 0 is the NORMAL observer,
    // which is not ∂_t-aligned in Kerr-Schild coordinates)
    var udotf: f64 = 0;
    for (0..4) |i| {
        for (0..4) |j| udotf += cd.gcov[i][j] * st.ucon[j] * st.fhat[i];
    }
    try std.testing.expectApproxEqAbs(@as(f64, 0), udotf / st.ehat, 1e-12);

    // dipole factor 1 + 3(k·F̂)/(ν̂Ê): outgoing radial photon boosted,
    // ingoing suppressed — and their deviations from 1 are symmetric.
    const e = render.tetradFromMetric(cd.gcov);
    inline for ([_]f64{ 1.0, -1.0 }) |dir| {
        var k: [4]f64 = undefined;
        for (0..4) |mu| k[mu] = e[0][mu] + dir * e[1][mu];
        var kcov: [4]f64 = @splat(0);
        for (0..4) |i| {
            for (0..4) |j| kcov[i] += cd.gcov[i][j] * k[j];
        }
        var nuhat: f64 = 0;
        for (0..4) |i| nuhat -= kcov[i] * st.ucon[i];
        var kdotf: f64 = 0;
        for (0..4) |i| kdotf += kcov[i] * st.fhat[i];
        const dip = 1.0 + 3.0 * kdotf / (nuhat * st.ehat);
        if (dir > 0) {
            try std.testing.expect(dip > 1.01);
        } else {
            try std.testing.expect(dip < 0.99);
        }
    }
}

test "render: scattering=false (AGN preset) zeroes kappa_es everywhere it enters" {
    const mp = metric.MetricParams{ .a = 0.0, .mksr0 = 0.1, .mksh0 = 0.9 };
    var body = [_]f64{0} ** L.count;
    var data = render.DumpData{
        .header = .{ .nx = 1, .ny = 1, .nz = 1, .nv = @intCast(L.count), .t = 0, .nstep = 0, .out_idx = 0 },
        .body = &body,
    };
    const g = puffyGrid(1, 1, 1, mp, 1.85, 1000.0);
    const consts = testConsts();
    var scene = render.Scene.init(g, mp, consts, opacities.Channels.puffy, 5.0 / 3.0, &data, 1000.0, 1000.0);

    const x = [4]f64{ 0, @log(20.0 - mp.mksr0), 0.5, 0 };
    const cd = metric.compute(.mks2, mp, x);
    const geom = render.geomFromCoordData(x, &cd);
    const tK = 1.0e7;
    var pp = [_]f64{0} ** L.count;
    pp[L.index(.rho)] = 1.0e-6;
    pp[L.index(.uu)] = thermo.uFromTrho(&consts, tK, 1.0e-6, 5.0 / 3.0);
    pp[L.index(.ee)] = consts.lteEfromT(tK);

    const st_on = render.localState(cfg, &scene, &geom, pp) orelse return error.TestUnexpectedResult;
    scene.scattering = false;
    const st_off = render.localState(cfg, &scene, &geom, pp) orelse return error.TestUnexpectedResult;

    try std.testing.expect(st_on.chi_es > 0);
    try std.testing.expectEqual(@as(f64, 0), st_off.chi_es);
    // extinction drops by exactly the scattering coefficient
    try std.testing.expectApproxEqRel(st_on.chi - st_on.chi_es, st_off.chi, 1e-13);
    // thermal emissivity (κ_abs·B) is untouched
    try std.testing.expectEqual(st_on.j_therm, st_off.j_therm);
}

test "render: sigma-cut and floor-cut mask emission but keep extinction" {
    const mp = metric.MetricParams{ .a = 0.0, .mksr0 = 0.1, .mksh0 = 0.9 };
    var body = [_]f64{0} ** L.count;
    var data = render.DumpData{
        .header = .{ .nx = 1, .ny = 1, .nz = 1, .nv = @intCast(L.count), .t = 0, .nstep = 0, .out_idx = 0 },
        .body = &body,
    };
    const g = puffyGrid(1, 1, 1, mp, 1.85, 1000.0);
    const consts = testConsts();
    var scene = render.Scene.init(g, mp, consts, opacities.Channels.puffy, 5.0 / 3.0, &data, 1000.0, 1000.0);

    const x = [4]f64{ 0, @log(20.0 - mp.mksr0), 0.5, 0 };
    const cd = metric.compute(.mks2, mp, x);
    const geom = render.geomFromCoordData(x, &cd);

    const tK = 1.0e7;
    var pp = [_]f64{0} ** L.count;
    pp[L.index(.rho)] = 1.0e-6;
    pp[L.index(.uu)] = thermo.uFromTrho(&consts, tK, 1.0e-6, 5.0 / 3.0);
    pp[L.index(.ee)] = consts.lteEfromT(tK);

    // σ-cut: a strongly magnetized cell (b²/ρ ≫ 1) emits nothing when the
    // cut is on, and emits when it is off. Extinction survives either way.
    pp[L.index(.b1)] = 1.0;
    scene.sigma_cut = 1.0;
    const st_cut = render.localState(cfg, &scene, &geom, pp) orelse return error.TestUnexpectedResult;
    try std.testing.expect(st_cut.masked);
    try std.testing.expect(st_cut.chi > 0);
    scene.sigma_cut = 0;
    const st_free = render.localState(cfg, &scene, &geom, pp) orelse return error.TestUnexpectedResult;
    try std.testing.expect(!st_free.masked);
    try std.testing.expect(st_free.j_therm > 0);
    pp[L.index(.b1)] = 0;

    // floor-cut at 10³×: ρ near the atmosphere profile is masked, real
    // (torus-like) ρ far above it is not.
    scene.floor = .{ .rho0 = 1.0e-24, .r0 = 2.0, .power = -1.5, .factor = 1.0e3 };
    pp[L.index(.rho)] = 1.0e-24; // ~30× the r=20 floor value — inside the cut
    pp[L.index(.uu)] = thermo.uFromTrho(&consts, tK, 1.0e-24, 5.0 / 3.0);
    const st_atm = render.localState(cfg, &scene, &geom, pp) orelse return error.TestUnexpectedResult;
    try std.testing.expect(st_atm.masked);
    pp[L.index(.rho)] = 1.0e-6; // ~3e19× the floor — real material
    pp[L.index(.uu)] = thermo.uFromTrho(&consts, tK, 1.0e-6, 5.0 / 3.0);
    const st_torus = render.localState(cfg, &scene, &geom, pp) orelse return error.TestUnexpectedResult;
    try std.testing.expect(!st_torus.masked);
}

// ---- analytic Kerr shadow (render/shadow.zig) ------------------------------

test "shadow: parametric xi/eta solve R = R' = 0 of the radial potential" {
    const shadow = @import("../render/shadow.zig");
    const a = 0.9375;
    const shell = shadow.photonShell(a);
    // known values for a = 0.9375: prograde ~1.43, retrograde ~3.94
    try std.testing.expect(shell.r_pro > 1.3 and shell.r_pro < 1.6);
    try std.testing.expect(shell.r_retro > 3.8 and shell.r_retro < 4.0);

    for ([_]f64{ 1.6, 2.0, 2.6, 3.2, 3.8 }) |rp| {
        const ce = shadow.xiEta(rp, a);
        const delta = rp * rp - 2.0 * rp + a * a;
        const p = rp * rp + a * a - a * ce.xi;
        const q = ce.eta + (ce.xi - a) * (ce.xi - a);
        const bigR = p * p - delta * q;
        const bigRp = 4.0 * rp * p - (2.0 * rp - 2.0) * q;
        try std.testing.expect(@abs(bigR) / (rp * rp * rp * rp) < 1e-10);
        try std.testing.expect(@abs(bigRp) / (rp * rp * rp) < 1e-10);
    }

    // a → 0 limit: the curve degenerates to ξ² + η = 27
    const ce0 = shadow.xiEta(3.0001, 1e-4);
    try std.testing.expectApproxEqRel(@as(f64, 27.0), ce0.xi * ce0.xi + ce0.eta, 1e-3);
}

test "render: camera center ray is impact-parameter-clean in Kerr (tetrad ordering)" {
    // In Kerr-Schild coordinates g_rφ → −a·sin²θ at ANY radius, so a tetrad
    // whose radial leg is not orthogonalized against ∂_φ launches every ray
    // with a spurious ξ ≈ −a·sin²θ — a rigid ~a·sinθ shift of the whole
    // image plane (caught by the Bardeen overlay). The (t, φ, r, θ)
    // Gram-Schmidt ordering must keep the center ray's ξ at O(1/r).
    const a = 0.9375;
    const mp = metric.MetricParams{ .a = a, .mksr0 = 0.1, .mksh0 = 0.9 };
    const r_cam = 4000.0;
    const x = [4]f64{ 0, @log(r_cam - mp.mksr0), forms.mks2X2FromTheta(pi / 3.0, mp.mksh0), 0 };
    const cd = metric.compute(.mks2, mp, x);
    const e = render.tetradFromMetric(cd.gcov);
    // e1 carries no azimuthal momentum at all ...
    var e1_dot_phi: f64 = 0;
    for (0..4) |i| e1_dot_phi += cd.gcov[3][i] * e[1][i];
    try std.testing.expectApproxEqAbs(@as(f64, 0), e1_dot_phi / r_cam, 1e-12);
    // ... so the center ray's ξ = L/E is O(1/r), not O(a)
    var k: [4]f64 = undefined;
    for (0..4) |mu| k[mu] = e[0][mu] + e[1][mu];
    const kc = kcovAt(mp, x, k);
    try std.testing.expect(@abs(kc[3] / kc[0]) < 5.0 / r_cam);
}

test "render: capture boundary lies on the analytic Kerr shadow curve (Bardeen)" {
    const allocator = std.testing.allocator;
    const shadow = @import("../render/shadow.zig");
    const a = 0.9375;
    const mp = metric.MetricParams{ .a = a, .mksr0 = 0.1, .mksh0 = 0.9 };
    var data = try zeroDump(allocator, 64, 32, 4);
    defer allocator.free(data.body);
    const g = puffyGrid(64, 32, 4, mp, 1.25, 1000.0);
    const scene = render.Scene.init(g, mp, testConsts(), opacities.Channels.puffy, 5.0 / 3.0, &data, 2000.0, 1000.0);

    const r_cam = 2000.0;
    const theta_cam = pi / 3.0; // 60° inclination
    const x2cam = forms.mks2X2FromTheta(theta_cam, mp.mksh0);
    const dthdx2 = forms.mks2DThetaDx2Metric(Dual3.constant(x2cam), mp.mksh0).v;
    const sth = @sin(theta_cam);
    const cth = @cos(theta_cam);

    // For 6 directions around the shadow: bisect the capture boundary in
    // impact angle, then check the boundary ray's EXACT conserved (ξ, η)
    // against the Bardeen critical curve. Conserved quantities make the
    // comparison exact at finite camera radius — no image-plane mapping.
    var idir: usize = 0;
    while (idir < 6) : (idir += 1) {
        const ang = 2.0 * pi * (@as(f64, @floatFromInt(idir)) + 0.31) / 6.0;
        const trace = struct {
            fn at(mp_: metric.MetricParams, sc: *const render.Scene, s_ang: f64, dir: f64) struct { captured: bool, x: [4]f64, k: [4]f64 } {
                const ax = s_ang * @cos(dir);
                const ay = s_ang * @sin(dir);
                const ray = launch(mp_, 2000.0, pi / 3.0, .{ @cos(ax) * @cos(ay), @sin(ay), -@sin(ax) });
                const res = render.traceRay(cfg, sc, ray.x, ray.k, .{ .eps = 0.3 });
                return .{ .captured = res.captured, .x = ray.x, .k = ray.k };
            }
        };

        var s_lo: f64 = 1.0 / r_cam; // b = 1 M: safely inside the shadow
        var s_hi: f64 = 12.0 / r_cam; // b = 12 M: safely outside
        try std.testing.expect(trace.at(mp, &scene, s_lo, ang).captured);
        try std.testing.expect(!trace.at(mp, &scene, s_hi, ang).captured);
        var iter: usize = 0;
        while (iter < 16) : (iter += 1) {
            const mid = 0.5 * (s_lo + s_hi);
            if (trace.at(mp, &scene, mid, ang).captured) s_lo = mid else s_hi = mid;
        }

        const bnd = trace.at(mp, &scene, 0.5 * (s_lo + s_hi), ang);
        const kc = kcovAt(mp, bnd.x, bnd.k);
        const e_ray = -kc[0];
        const l_ray = kc[3];
        const ktheta = kc[2] / dthdx2;
        const xi = l_ray / e_ray;
        const carter = ktheta * ktheta + cth * cth * (l_ray * l_ray / (sth * sth) - a * a * e_ray * e_ray);
        const eta = carter / (e_ray * e_ray);

        const crit = shadow.etaCritForXi(xi, a) orelse return error.TestUnexpectedResult;
        const rel = @abs(eta - crit) / (eta + xi * xi);
        try std.testing.expect(rel < 5e-3);
    }
}

// ---- monochromatic emission (render/emission.zig) --------------------------

test "emission: Bessel K2 matches small- and large-x asymptotics" {
    const em = @import("../render/emission.zig");
    // x → 0: K2 ~ 2/x²
    try std.testing.expectApproxEqRel(@as(f64, 2.0 / (0.01 * 0.01)), em.besselK2(0.01), 1e-3);
    // x ≫ 1: K2 ~ √(π/2x)·e^{−x}·(1 + 15/(8x) + 105/(128x²))
    const x = 10.0;
    const asym = @sqrt(std.math.pi / (2.0 * x)) * @exp(-x) *
        (1.0 + 15.0 / (8.0 * x) + 105.0 / (128.0 * x * x));
    try std.testing.expectApproxEqRel(asym, em.besselK2(x), 1e-3);
    // continuity across the x = 2 branch switch
    try std.testing.expectApproxEqRel(em.besselK2(1.999999), em.besselK2(2.000001), 1e-5);
}

test "emission: Planck function RJ limit and Stefan-Boltzmann integral" {
    const em = @import("../render/emission.zig");
    const temp = 1.0e7;
    // Rayleigh-Jeans at hν ≪ kT
    const nu_rj = 1.0e10;
    const rj = 2.0 * nu_rj * nu_rj * em.k_cgs * temp / (em.c_cgs * em.c_cgs);
    try std.testing.expectApproxEqRel(rj, em.planckNu(nu_rj, temp), 1e-4);
    // ∫B_ν dν = σT⁴/π (log-trapezoid over 12 decades)
    const n = 4000;
    var sum: f64 = 0;
    var i: usize = 0;
    const lo = @log(1.0e10);
    const hi = @log(1.0e22);
    const dln = (hi - lo) / @as(f64, @floatFromInt(n));
    while (i < n) : (i += 1) {
        const nu = @exp(lo + (@as(f64, @floatFromInt(i)) + 0.5) * dln);
        sum += em.planckNu(nu, temp) * nu * dln;
    }
    const sigma_cgs = 5.670367e-5;
    try std.testing.expectApproxEqRel(sigma_cgs * temp * temp * temp * temp / std.math.pi, sum, 1e-3);
}

test "emission: monochromatic LTE source function is exactly Planck" {
    const em = @import("../render/emission.zig");
    // T_e = T_rad, dipole 1: every channel obeys Kirchhoff, so j/χ = B_ν
    // regardless of the channel mix (synchrotron on at this θ_e ≈ 0.85).
    const temp = 5.0e9;
    const in = em.MonoIn{
        .nu = 2.3e11,
        .ne_cgs = 1.0e8,
        .ni_cgs = 1.0e8,
        .te = temp,
        .trad = temp,
        .b_gauss = 100.0,
        .sin_pitch = 0.7,
        .chi_es_cgs = 1.0e-10,
        .dip = 1.0,
    };
    const out = em.monoJChi(in);
    try std.testing.expect(out.chi > in.chi_es_cgs); // synch+ff actually on
    try std.testing.expectApproxEqRel(em.planckNu(in.nu, temp), out.j / out.chi, 1e-12);
}

test "emission: synchrotron has the exponential high-frequency cutoff" {
    const em = @import("../render/emission.zig");
    const base = em.MonoIn{
        .nu = 0,
        .ne_cgs = 1.0e7,
        .ni_cgs = 0,
        .te = 5.0e10, // θ_e ≈ 8.5 — synchrotron-dominated
        .trad = 1.0e4, // negligible scattering source
        .b_gauss = 30.0,
        .sin_pitch = 0.8,
        .chi_es_cgs = 0,
        .dip = 1.0,
    };
    var lo = base;
    lo.nu = 1.0e11;
    var hi = base;
    hi.nu = 1.0e16;
    const j_lo = em.monoJChi(lo).j;
    const j_hi = em.monoJChi(hi).j;
    try std.testing.expect(j_lo > 0);
    try std.testing.expect(j_hi < 1e-6 * j_lo);
}

test "render: supersampled rays — pixel center matches ray(), ss=2 averages cleanly" {
    const allocator = std.testing.allocator;
    const mp = metric.MetricParams{ .a = 0.0, .mksr0 = 0.1, .mksh0 = 0.9 };
    var cam = render.Camera{ .r = 1000, .incl_deg = 60, .fov = 40, .width = 8, .height = 8, .ss = 2 };
    cam.setup(mp);
    // rayAt at the pixel center is exactly ray()
    const k_a = cam.ray(3, 5);
    const k_b = cam.rayAt(3.5, 5.5);
    for (0..4) |mu| try std.testing.expectEqual(k_a[mu], k_b[mu]);

    // a vacuum scene renders to zeros through the ss=2 path (indexing sane)
    var data = try zeroDump(allocator, 8, 6, 1);
    defer allocator.free(data.body);
    const g = puffyGrid(8, 6, 1, mp, 1.85, 1000.0);
    const scene = render.Scene.init(g, mp, testConsts(), opacities.Channels.puffy, 5.0 / 3.0, &data, 1000.0, 1000.0);
    var img = [_]f64{-1} ** 64;
    render.renderImage(cfg, &scene, &cam, &img, .{ .eps = 1.0, .max_steps = 20_000 }, 2);
    for (img) |v| try std.testing.expectEqual(@as(f64, 0), v);
}

// ---- sampling --------------------------------------------------------------

test "render: sampling tiles the phi wedge periodically" {
    const allocator = std.testing.allocator;
    const mp = metric.MetricParams{ .a = 0.0, .mksr0 = 0.1, .mksh0 = 0.9 };
    const nx = 8;
    const ny = 6;
    const nz = 4;
    var data = try zeroDump(allocator, nx, ny, nz);
    defer allocator.free(data.body);
    // distinct rho per cell
    for (0..nz) |iz| {
        for (0..ny) |iy| {
            for (0..nx) |ix| {
                const idx = ((iz * ny + iy) * nx + ix) * L.count;
                data.body[idx + L.index(.rho)] = 1.0 + @as(f64, @floatFromInt(ix + 10 * iy + 100 * iz));
            }
        }
    }
    const g = puffyGrid(nx, ny, nz, mp, 1.85, 1000.0);
    const consts = testConsts();
    const scene = render.Scene.init(g, mp, consts, opacities.Channels.puffy, 5.0 / 3.0, &data, 1000.0, 1000.0);

    const wedge = pi / 2.0;
    var pp_a: [L.count]f64 = undefined;
    var pp_b: [L.count]f64 = undefined;
    const x_a = [4]f64{ 0, @log(30.0 - mp.mksr0), 0.42, 0.13 };
    var x_b = x_a;
    x_b[3] = x_a[3] + 2.0 * wedge; // two wedge periods away
    try std.testing.expect(render.sample(cfg, &scene, x_a, &pp_a));
    try std.testing.expect(render.sample(cfg, &scene, x_b, &pp_b));
    try std.testing.expectApproxEqRel(pp_a[L.index(.rho)], pp_b[L.index(.rho)], 1e-13);

    // radially outside the data → vacuum
    const x_out = [4]f64{ 0, @log(2000.0), 0.5, 0 };
    try std.testing.expect(!render.sample(cfg, &scene, x_out, &pp_a));
}

// ---- image backend ---------------------------------------------------------

test "image: afmhot endpoints and monotonicity" {
    try std.testing.expectEqual([3]u8{ 0, 0, 0 }, image.afmhot(0.0));
    try std.testing.expectEqual([3]u8{ 255, 255, 255 }, image.afmhot(1.0));
    const mid = image.afmhot(0.5);
    try std.testing.expectEqual(@as(u8, 255), mid[0]);
    try std.testing.expectEqual(@as(u8, 0), mid[2]);
    var prev = [3]u8{ 0, 0, 0 };
    var i: usize = 0;
    while (i <= 100) : (i += 1) {
        const c = image.afmhot(@as(f64, @floatFromInt(i)) / 100.0);
        try std.testing.expect(c[0] >= prev[0] and c[1] >= prev[1] and c[2] >= prev[2]);
        prev = c;
    }
}

test "image: gaussian blur preserves total flux of an interior impulse" {
    const allocator = std.testing.allocator;
    const w = 32;
    var img = [_]f64{0} ** (w * w);
    img[16 * w + 16] = 1.0;
    try image.gaussianBlur(allocator, &img, w, w, 2.0);
    var sum: f64 = 0;
    var maxv: f64 = 0;
    for (img) |v| {
        sum += v;
        maxv = @max(maxv, v);
    }
    try std.testing.expectApproxEqRel(@as(f64, 1.0), sum, 1e-9);
    try std.testing.expect(maxv < 0.1); // actually spread out
}

test "image: PNG encoder produces valid chunks and roundtrips pixels" {
    const allocator = std.testing.allocator;
    const w = 3;
    const h = 2;
    const rgb = [_]u8{
        255, 0,  0,  0, 255, 0, 0, 0, 255,
        10,  20, 30, 0, 0,   0, 9, 8, 7,
    };
    const png = try image.encodePng(allocator, w, h, &rgb);
    defer allocator.free(png);

    try std.testing.expect(std.mem.eql(u8, png[0..8], &image.png_signature));

    // walk the chunks, checking each CRC and collecting IDAT
    var idat: std.ArrayList(u8) = .empty;
    defer idat.deinit(allocator);
    var off: usize = 8;
    var saw_ihdr = false;
    var saw_iend = false;
    while (off < png.len) {
        const len = std.mem.readInt(u32, png[off..][0..4], .big);
        const typ = png[off + 4 ..][0..4];
        const data = png[off + 8 ..][0..len];
        const crc_file = std.mem.readInt(u32, png[off + 8 + len ..][0..4], .big);
        var crc = std.hash.Crc32.init();
        crc.update(typ);
        crc.update(data);
        try std.testing.expectEqual(crc.final(), crc_file);
        if (std.mem.eql(u8, typ, "IHDR")) {
            saw_ihdr = true;
            try std.testing.expectEqual(@as(u32, w), std.mem.readInt(u32, data[0..4], .big));
            try std.testing.expectEqual(@as(u32, h), std.mem.readInt(u32, data[4..8], .big));
            try std.testing.expectEqual(@as(u8, 8), data[8]); // bit depth
            try std.testing.expectEqual(@as(u8, 2), data[9]); // RGB
        } else if (std.mem.eql(u8, typ, "IDAT")) {
            try idat.appendSlice(allocator, data);
        } else if (std.mem.eql(u8, typ, "IEND")) {
            saw_iend = true;
        }
        off += 12 + len;
    }
    try std.testing.expect(saw_ihdr and saw_iend);
    try std.testing.expectEqual(png.len, off);

    // decode the zlib stream of stored deflate blocks
    const z = idat.items;
    try std.testing.expectEqual(@as(u8, 0x78), z[0]);
    var raw: std.ArrayList(u8) = .empty;
    defer raw.deinit(allocator);
    var zo: usize = 2;
    while (true) {
        const final = z[zo] & 1;
        try std.testing.expectEqual(@as(u8, 0), z[zo] >> 1); // stored block
        const n = std.mem.readInt(u16, z[zo + 1 ..][0..2], .little);
        const nlen = std.mem.readInt(u16, z[zo + 3 ..][0..2], .little);
        try std.testing.expectEqual(n, ~nlen);
        try raw.appendSlice(allocator, z[zo + 5 ..][0..n]);
        zo += 5 + n;
        if (final == 1) break;
    }
    const adler_file = std.mem.readInt(u32, z[zo..][0..4], .big);
    try std.testing.expectEqual(std.hash.Adler32.hash(raw.items), adler_file);

    // unfilter (all rows use filter 0) and compare pixels
    const stride = 1 + 3 * w;
    try std.testing.expectEqual(@as(usize, h * stride), raw.items.len);
    for (0..h) |y| {
        try std.testing.expectEqual(@as(u8, 0), raw.items[y * stride]);
        try std.testing.expect(std.mem.eql(
            u8,
            raw.items[y * stride + 1 ..][0 .. 3 * w],
            rgb[y * 3 * w ..][0 .. 3 * w],
        ));
    }
}

test "image: white point and stretch normalize to [0,1]" {
    const allocator = std.testing.allocator;
    var img: [100]f64 = undefined;
    for (&img, 0..) |*v, i| v.* = @floatFromInt(i);
    const wp = try image.whitePoint(allocator, &img, 99.0);
    try std.testing.expect(wp >= 97.0 and wp <= 99.0);
    image.stretch(&img, wp, 0.5);
    for (img) |v| try std.testing.expect(v >= 0 and v <= 1.0);
    try std.testing.expectEqual(@as(f64, 1.0), img[99]);
}

// ---- slow light: series, sweep, timing -------------------------------------

/// Fill every cell of `d` with a uniform emissive LTE-ish state (gas at
/// rest, no B): enough matter to make intensities nonzero, thin enough
/// that rays cross the whole domain.
fn fillUniform(d: *render.DumpData, rho: f64, tK: f64, consts: *const thermo.Consts) void {
    const nv = d.header.nv;
    const ncell = d.body.len / nv;
    const uu = thermo.uFromTrho(consts, tK, rho, 5.0 / 3.0);
    const ee = consts.lteEfromT(tK);
    for (0..ncell) |c| {
        d.body[c * nv + L.index(.rho)] = rho;
        d.body[c * nv + L.index(.uu)] = uu;
        d.body[c * nv + L.index(.ee)] = ee;
    }
}

test "series: window sampler lerps primitives linearly in time and clamps at the edges" {
    const allocator = std.testing.allocator;
    const mp = metric.MetricParams{ .a = 0.0, .mksr0 = 0.1, .mksh0 = 0.9 };
    var a = try zeroDump(allocator, 1, 1, 1);
    defer allocator.free(a.body);
    var b = try zeroDump(allocator, 1, 1, 1);
    defer allocator.free(b.body);
    for (a.body) |*v| v.* = 2.0;
    for (b.body) |*v| v.* = 6.0;
    const g = puffyGrid(1, 1, 1, mp, 1.85, 1000.0);
    const scene = render.Scene.init(g, mp, testConsts(), opacities.Channels.puffy, 5.0 / 3.0, &a, 1000.0, 1000.0);

    var smp = render.series.WindowSampler{ .lo = &a, .hi = &b, .t_lo = 10.0, .t_hi = 20.0 };
    var pp: [L.count]f64 = undefined;

    // interior: exact linear interpolation
    var x = [4]f64{ 15.0, @log(20.0 - mp.mksr0), 0.5, 0 };
    try std.testing.expect(smp.sample(cfg, &scene, x, &pp));
    for (pp) |v| try std.testing.expectEqual(@as(f64, 4.0), v);
    x[0] = 12.5;
    try std.testing.expect(smp.sample(cfg, &scene, x, &pp));
    for (pp) |v| try std.testing.expectEqual(@as(f64, 3.0), v);
    try std.testing.expectEqual(@as(u64, 0), smp.n_below + smp.n_above);

    // edges: clamp-and-count (hold the end frame)
    x[0] = 5.0;
    try std.testing.expect(smp.sample(cfg, &scene, x, &pp));
    for (pp) |v| try std.testing.expectEqual(@as(f64, 2.0), v);
    try std.testing.expectEqual(@as(u64, 1), smp.n_below);
    x[0] = 25.0;
    try std.testing.expect(smp.sample(cfg, &scene, x, &pp));
    for (pp) |v| try std.testing.expectEqual(@as(f64, 6.0), v);
    try std.testing.expectEqual(@as(u64, 1), smp.n_above);

    // degenerate window (same frame twice) = plain snapshot sampling
    var one = render.series.WindowSampler{ .lo = &a, .hi = &a, .t_lo = 0, .t_hi = 0 };
    x[0] = -1e9;
    try std.testing.expect(one.sample(cfg, &scene, x, &pp));
    for (pp) |v| try std.testing.expectEqual(@as(f64, 2.0), v);
    try std.testing.expectEqual(@as(u64, 0), one.n_below + one.n_above);
}

test "series: a static series reproduces the fast-light trace bit-for-bit" {
    const allocator = std.testing.allocator;
    const mp = metric.MetricParams{ .a = 0.9375, .mksr0 = 0.1, .mksh0 = 0.9 };
    const consts = testConsts();
    var a = try zeroDump(allocator, 32, 24, 1);
    defer allocator.free(a.body);
    fillUniform(&a, 1.0e-26, 1.0e7, &consts);
    var b = try zeroDump(allocator, 32, 24, 1);
    defer allocator.free(b.body);
    @memcpy(b.body, a.body);

    const g = puffyGrid(32, 24, 1, mp, 1.25, 500.0);
    const scene = render.Scene.init(g, mp, consts, opacities.Channels.puffy, 5.0 / 3.0, &a, 1000.0, 500.0);
    var cam = render.Camera{ .r = 1000, .incl_deg = 60, .fov = 30, .width = 3, .height = 3, .ss = 1 };
    cam.setup(mp);

    // distinct frame pointers with identical bodies: the lerp path runs and
    // must still be exact (the a + w*(b-a) form). Also pin stationarity:
    // shifting the camera's coordinate time must change nothing.
    var smp = render.series.WindowSampler{ .lo = &a, .hi = &b, .t_lo = 0.0, .t_hi = 50.0 };
    for (0..3) |py| {
        for (0..3) |px| {
            const k0 = cam.ray(px, py);
            var x0 = cam.x0;
            const fast = render.traceRay(cfg, &scene, x0, k0, .{});
            try std.testing.expect(fast.intensity > 0);
            x0[0] = 500.0; // mid-window at depth, above it near the camera
            const slow = render.traceRayWith(cfg, &scene, &smp, x0, k0, .{});
            try std.testing.expectEqual(fast.intensity, slow.intensity);
            try std.testing.expectEqual(fast.tau, slow.tau);
            try std.testing.expectEqual(fast.steps, slow.steps);
            for (0..4) |mu| {
                if (mu != 0) try std.testing.expectEqual(fast.x[mu], slow.x[mu]);
                try std.testing.expectEqual(fast.k[mu], slow.k[mu]);
            }
        }
    }
}

test "sweep: static series through the full 3-phase sweep is bit-identical to direct tracing" {
    const allocator = std.testing.allocator;
    const mp = metric.MetricParams{ .a = 0.9375, .mksr0 = 0.1, .mksh0 = 0.9 };
    const consts = testConsts();
    var a = try zeroDump(allocator, 32, 24, 1);
    defer allocator.free(a.body);
    fillUniform(&a, 1.0e-26, 1.0e7, &consts);
    var b = try zeroDump(allocator, 32, 24, 1);
    defer allocator.free(b.body);
    @memcpy(b.body, a.body);

    const g = puffyGrid(32, 24, 1, mp, 1.25, 500.0);
    const scene = render.Scene.init(g, mp, consts, opacities.Channels.puffy, 5.0 / 3.0, &a, 1000.0, 500.0);
    var cam = render.Camera{ .r = 1000, .incl_deg = 60, .fov = 30, .width = 6, .height = 6, .ss = 2 };
    cam.setup(mp);

    const specs = try render.sweep.uniformPlan(allocator, &cam);
    defer allocator.free(specs);

    // alternate two identical-content frames so every window pair has
    // distinct pointers (the lerp path, not the degenerate shortcut)
    const ts = [_]f64{ 0.0, 15.0, 30.0, 45.0 };
    const frames = [_]*const render.DumpData{ &a, &b, &a, &b };
    var src = render.series.SliceSource{ .ts = ts[0..], .frames = frames[0..] };

    const opts = render.TraceOpts{};
    var out = [_]f64{-1} ** 36;
    const stats = try render.sweep.renderSlow(cfg, allocator, &scene, &cam, &src, specs, out[0..], opts, .{ .t_cam = 1030.0, .r_slow = 40.0 }, 3, false);

    // the sweep must actually have exercised its machinery
    try std.testing.expect(stats.entered > 0);
    try std.testing.expect(stats.rounds >= 3);
    try std.testing.expect(stats.tails > 0);
    try std.testing.expect(stats.past_start > 0);

    // reference: plain fast-light tracing of the same plan
    var want = [_]f64{0} ** 36;
    for (specs) |sp| {
        const res = render.traceRay(cfg, &scene, cam.x0, cam.rayAt(sp.fx, sp.fy), opts);
        want[sp.pix] += sp.weight * res.intensity;
    }
    for (want, out) |w, o| try std.testing.expectEqual(w, o);
    var sum: f64 = 0;
    for (out) |o| sum += o;
    try std.testing.expect(sum > 0);
}

test "sweep: a time-localized flare arrives retarded by the (Shapiro-corrected) travel time" {
    const allocator = std.testing.allocator;
    const mp = metric.MetricParams{ .a = 0.0, .mksr0 = 0.1, .mksh0 = 0.9 };
    const consts = testConsts();
    const nx = 64;
    const ny = 32;

    var zero = try zeroDump(allocator, nx, ny, 1);
    defer allocator.free(zero.body);
    var blob = try zeroDump(allocator, nx, ny, 1);
    defer allocator.free(blob.body);

    const g = puffyGrid(nx, ny, 1, mp, 1.85, 500.0);
    // light up a small ring of cells around (r = 10, θ = π/3) — on the
    // line of sight of a camera at incl 60°
    const uu = thermo.uFromTrho(&consts, 1.0e7, 1.0e-26, 5.0 / 3.0);
    const ee = consts.lteEfromT(1.0e7);
    const th_c = pi / 3.0;
    for (0..ny) |iy| {
        const x2 = g.miny + (@as(f64, @floatFromInt(iy)) + 0.5) * g.dy;
        const th = forms.mks2Theta(Dual3.constant(x2), mp.mksh0).v;
        if (@abs(th - th_c) > 0.12) continue;
        for (0..nx) |ix| {
            const x1 = g.minx + (@as(f64, @floatFromInt(ix)) + 0.5) * g.dx;
            const r = @exp(x1) + mp.mksr0;
            if (r < 9.0 or r > 11.0) continue;
            const base = (iy * nx + ix) * L.count;
            blob.body[base + L.index(.rho)] = 1.0e-26;
            blob.body[base + L.index(.uu)] = uu;
            blob.body[base + L.index(.ee)] = ee;
        }
    }

    // series: dark everywhere except ONE lit frame at t = 20 (linear ramps
    // to the dark neighbors: emission is nonzero only for t_sample in (15, 25))
    const ts = [_]f64{ 0, 5, 10, 15, 20, 25, 30, 35, 40 };
    var frames: [9]*const render.DumpData = @splat(&zero);
    frames[4] = &blob;
    var src = render.series.SliceSource{ .ts = ts[0..], .frames = frames[0..] };

    // reference frame (static zone + tails) = dark
    const scene = render.Scene.init(g, mp, consts, opacities.Channels.puffy, 5.0 / 3.0, &zero, 200.0, 500.0);
    var cam = render.Camera{ .r = 200, .incl_deg = 60, .fov = 2, .width = 1, .height = 1, .ss = 1 };
    cam.setup(mp);
    const specs = try render.sweep.uniformPlan(allocator, &cam);
    defer allocator.free(specs);

    // outgoing radial ray in Schwarzschild KS time: Δt = Δr + 4M ln((r_cam−2M)/(r_e−2M))
    const r_b = 10.0;
    const dt_travel = (cam.r - r_b) + 4.0 * @log((cam.r - 2.0) / (r_b - 2.0));

    const I = struct {
        fn at(alloc: std.mem.Allocator, sc: *const render.Scene, c: *const render.Camera, sr: *render.series.SliceSource, sp: []const render.sweep.RaySpec, t_cam: f64) !f64 {
            var out = [_]f64{0} ** 1;
            _ = try render.sweep.renderSlow(cfg, alloc, sc, c, sr, sp, out[0..], .{}, .{ .t_cam = t_cam, .r_slow = 40.0 }, 1, false);
            return out[0];
        }
    };

    const t_peak = 20.0 + dt_travel;
    const bright = try I.at(allocator, &scene, &cam, &src, specs, t_peak);
    const early = try I.at(allocator, &scene, &cam, &src, specs, t_peak - 12.0);
    const late = try I.at(allocator, &scene, &cam, &src, specs, t_peak + 12.0);
    const shoulder = try I.at(allocator, &scene, &cam, &src, specs, t_peak + 3.0);

    // lit exactly when the RETARDED time at the blob falls in the lit window
    try std.testing.expect(bright > 0);
    try std.testing.expectEqual(@as(f64, 0), early);
    try std.testing.expectEqual(@as(f64, 0), late);
    try std.testing.expect(shoulder > 0 and shoulder < bright);
}

// ---- slow light: spacetime timing gates ------------------------------------

test "render: radial flight time carries the Shapiro delay — Δt = Δr + 4M ln in KS time" {
    const allocator = std.testing.allocator;
    const mp = metric.MetricParams{ .a = 0.0, .mksr0 = 0.1, .mksh0 = 0.9 };
    var data = try zeroDump(allocator, 64, 32, 1);
    defer allocator.free(data.body);
    const g = puffyGrid(64, 32, 1, mp, 1.85, 1000.0);
    var scene = render.Scene.init(g, mp, testConsts(), opacities.Channels.puffy, 5.0 / 3.0, &data, 1000.0, 1000.0);

    // stop the trace at r_e and interpolate the crossing with the endpoint k
    const r_e = 10.0;
    scene.r_capture = r_e;
    const ray = launch(mp, 1000.0, pi / 3.0, .{ 1.0, 0.0, 0.0 });
    const res = render.traceRay(cfg, &scene, ray.x, ray.k, .{ .eps = 0.1 });
    try std.testing.expect(res.captured);
    const r_end = @exp(res.x[1]) + mp.mksr0;
    const t_cross = res.x[0] + (r_e - r_end) * res.k[0] / ((r_end - mp.mksr0) * res.k[1]);
    const t_flight = -t_cross; // launched at x⁰ = 0, traced into the past

    // ingoing-KS time along an OUTGOING radial null ray:
    //   dt/dr = (r + 2M)/(r − 2M)  ⇒  Δt = Δr + 4M ln((r_cam−2M)/(r_e−2M)).
    // The 4M ln term is 19.3 M here — the Shapiro delay slow light must
    // carry (flat-space retardation would be just Δr).
    const want = (1000.0 - r_e) + 4.0 * @log((1000.0 - 2.0) / (r_e - 2.0));
    try std.testing.expectApproxEqAbs(want, t_flight, 0.05);
}

test "render: successive photon-ring windings are delayed by the photon-orbit period 2pi*3sqrt(27)... (a=0)" {
    const allocator = std.testing.allocator;
    const mp = metric.MetricParams{ .a = 0.0, .mksr0 = 0.1, .mksh0 = 0.9 };
    var data = try zeroDump(allocator, 64, 32, 1);
    defer allocator.free(data.body);
    const g = puffyGrid(64, 32, 1, mp, 1.85, 1000.0);
    var scene = render.Scene.init(g, mp, testConsts(), opacities.Channels.puffy, 5.0 / 3.0, &data, 100.0, 100.0);
    scene.r_escape = 110.0;

    const r_cam: f64 = 100.0;
    const r_ref: f64 = 105.0; // interpolate the exit through this radius

    // equatorial ray at impact angle α: Δφ and flight time at the exit
    // sphere, or null for a captured ray. x[3] is never wrapped by the
    // integrator, so it IS the unwrapped winding angle.
    const Probe = struct {
        fn shoot(sc: *const render.Scene, mpar: metric.MetricParams, alpha: f64) ?struct { dphi: f64, t: f64 } {
            const ray = launch(mpar, r_cam, pi / 2.0, .{ @cos(alpha), 0, -@sin(alpha) });
            const res = render.traceRay(cfg, sc, ray.x, ray.k, .{ .eps = 0.1, .max_steps = 400_000 });
            if (res.captured) return null;
            const r_end = @exp(res.x[1]) + mpar.mksr0;
            const s_ = (r_ref - r_end) / ((r_end - mpar.mksr0) * res.k[1]);
            return .{
                .dphi = @abs(res.x[3] + s_ * res.k[3]),
                .t = -(res.x[0] + s_ * res.k[0]),
            };
        }
    };

    // bisect α to a target total winding Δφ (monotone: smaller α → closer
    // to b_crit → more winding; captured counts as "infinite winding")
    const Find = struct {
        fn at(sc: *const render.Scene, mpar: metric.MetricParams, target: f64) f64 {
            var lo = std.math.asin(4.5 / r_cam); // captured side
            var hi = std.math.asin(8.0 / r_cam); // weak-bending side
            var t_best: f64 = 0;
            for (0..80) |_| {
                const mid = 0.5 * (lo + hi);
                if (Probe.shoot(sc, mpar, mid)) |p| {
                    if (p.dphi > target) {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                    if (@abs(p.dphi - target) < 1e-4) return p.t;
                    t_best = p.t;
                } else {
                    lo = mid; // captured: even more winding than target
                }
            }
            return t_best;
        }
    };

    // rays whose total Δφ differs by exactly 2π exit with IDENTICAL
    // geometry — the extra turn happens on the photon shell, so their
    // flight times differ by one photon-orbit period T = 2π·3√3 M.
    // This is the subring delay slow light resolves (T/2 per subring).
    const t3 = Find.at(&scene, mp, 3.0 * pi);
    const t5 = Find.at(&scene, mp, 5.0 * pi);
    const t7 = Find.at(&scene, mp, 7.0 * pi);
    const period = 2.0 * pi * 3.0 * @sqrt(3.0);
    try std.testing.expectApproxEqAbs(period, t5 - t3, 0.5);
    try std.testing.expectApproxEqAbs(period, t7 - t5, 0.5);
}

// ---- adaptive refinement ---------------------------------------------------

test "adaptive: quadtree plan conserves pixel weight, marks the capture boundary, antialiases it" {
    const allocator = std.testing.allocator;
    const mp = metric.MetricParams{ .a = 0.9375, .mksr0 = 0.1, .mksh0 = 0.9 };
    var data = try zeroDump(allocator, 32, 24, 1);
    defer allocator.free(data.body);
    const g = puffyGrid(32, 24, 1, mp, 1.25, 500.0);
    var scene = render.Scene.init(g, mp, testConsts(), opacities.Channels.puffy, 5.0 / 3.0, &data, 300.0, 500.0);
    // test-time economy: a close camera + early escape + coarse eps keep the
    // Debug-mode step count down; the plan logic is resolution-independent
    scene.r_escape = 330.0;
    const n = 16;
    var cam = render.Camera{ .r = 300, .incl_deg = 60, .fov = 18, .width = n, .height = n, .ss = 2 };
    cam.setup(mp);

    const opts = render.TraceOpts{ .eps = 1.0, .max_steps = 12_000 };
    var plan = try render.adaptive.planRays(cfg, allocator, &scene, &cam, opts, .{ .depth = 2, .dt_thresh = 5.0, .probe_max_steps = 3_000 }, 2);
    defer plan.deinit(allocator);
    try std.testing.expect(plan.vacuum_rays > (n + 1) * (n + 1));

    // Σ weight per pixel is EXACTLY 1 (powers of 1/4)
    var wsum = [_]f64{0} ** (n * n);
    for (plan.specs) |sp| wsum[sp.pix] += sp.weight;
    for (wsum) |v| try std.testing.expectEqual(@as(f64, 1.0), v);

    // the refined set is a band, not empty and not the whole frame
    var nmark: usize = 0;
    for (plan.marked) |m| nmark += @intFromBool(m);
    try std.testing.expect(nmark > 4 and nmark < n * n / 2);

    // the capture boundary (bisected independently along the middle row,
    // both directions) lies in a marked pixel; a marked pixel carries at
    // least the leaves of one subdivision
    const Cap = struct {
        fn at(sc: *const render.Scene, c: *const render.Camera, fx: f64, fy: f64) bool {
            var o = render.TraceOpts{ .eps = 1.0, .max_steps = 12_000 };
            o.screen = true;
            return render.traceRay(cfg, sc, c.x0, c.rayAt(fx, fy), o).captured;
        }
    };
    const fy_mid = @as(f64, n) / 2.0 + 0.5;
    try std.testing.expect(Cap.at(&scene, &cam, @as(f64, n) / 2.0, fy_mid)); // hole center
    for ([_]f64{ n - 0.1, 0.1 }) |fx_out| {
        try std.testing.expect(!Cap.at(&scene, &cam, fx_out, fy_mid));
        var in: f64 = @as(f64, n) / 2.0;
        var out_: f64 = fx_out;
        for (0..40) |_| {
            const mid = 0.5 * (in + out_);
            if (Cap.at(&scene, &cam, mid, fy_mid)) in = mid else out_ = mid;
        }
        const px: usize = @intFromFloat(std.math.clamp(0.5 * (in + out_), 0, n - 1));
        const row = (n / 2) * n;
        const hit = plan.marked[row + px] or
            (px > 0 and plan.marked[row + px - 1]) or
            (px + 1 < n and plan.marked[row + px + 1]);
        try std.testing.expect(hit);
        var count: usize = 0;
        for (plan.specs) |sp| {
            if (sp.pix == row + px and plan.marked[row + px]) count += 1;
        }
        if (plan.marked[row + px]) try std.testing.expect(count >= 4);
    }

    // planned screen render: fractional shadow-edge coverage, exact 0/1 core
    var img = [_]f64{0} ** (n * n);
    var o = render.TraceOpts{ .eps = 1.0, .max_steps = 12_000 };
    o.screen = true;
    try render.adaptive.renderPlan(cfg, allocator, &scene, &cam, plan.specs, img[0..], o, 2);
    var frac: usize = 0;
    for (img) |v| {
        try std.testing.expect(v >= 0.0 and v <= 1.0);
        if (v > 0.0 and v < 1.0) frac += 1;
    }
    try std.testing.expect(frac >= 6);
    try std.testing.expectEqual(@as(f64, 0.0), img[(n / 2) * n + n / 2]);
    try std.testing.expectEqual(@as(f64, 1.0), img[0]);
}

// ---- FITS export -----------------------------------------------------------

test "fits: header cards, geometry keywords, big-endian data, row flip, block padding" {
    const allocator = std.testing.allocator;
    const fits = render.fits;

    // 3x2 image, distinct values; row 0 is the TOP row in renderer order
    const img = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    const bytes = try fits.encode(allocator, img[0..], 3, 2, .{
        .cdelt_deg = 1.25e-10,
        .ra_deg = 266.4168371,
        .dec_deg = -29.0078106,
        .freq_hz = 230.0e9,
        .mjd = 57850.0,
    });
    defer allocator.free(bytes);

    // sizes: one header block + one (padded) data block
    try std.testing.expectEqual(@as(usize, 2 * fits.block), bytes.len);

    // card 0 is SIMPLE with T at column 30 (index 29)
    try std.testing.expect(std.mem.startsWith(u8, bytes, "SIMPLE  ="));
    try std.testing.expectEqual(@as(u8, 'T'), bytes[29]);

    // mandatory + WCS cards present, with FITS-legal uppercase exponents
    const header = bytes[0..fits.block];
    for ([_][]const u8{
        "BITPIX  =                  -64",
        "NAXIS   =                    2",
        "NAXIS1  =                    3",
        "NAXIS2  =                    2",
        "BUNIT   = 'JY/PIXEL'",
        "CTYPE1  = 'RA---SIN'",
"CDELT1  =  -1.250000000000E-10",
        "CDELT2  =   1.250000000000E-10",
        "OBSRA   =     2.664168371000E2",
        "OBSDEC  =    -2.900781060000E1",
        "FREQ    =    2.300000000000E11",
        "MJD     =     5.785000000000E4",
        "END",
    }) |want| {
        try std.testing.expect(std.mem.indexOf(u8, header, want) != null);
    }
    // every header byte is printable ASCII (blank-filled, no zeros)
    for (header) |b| try std.testing.expect(b >= 32 and b <= 126);

    // data: big-endian f64, bottom row first — FITS row 0 = renderer row 1
    const D = struct {
        fn at(bs: []const u8, i: usize) f64 {
            return @bitCast(std.mem.readInt(u64, bs[fits.block + 8 * i ..][0..8], .big));
        }
    };
    for ([_]f64{ 4.0, 5.0, 6.0, 1.0, 2.0, 3.0 }, 0..) |want, i| {
        try std.testing.expectEqual(want, D.at(bytes, i));
    }
    // padding after the 48 data bytes is zero
    for (bytes[fits.block + 48 ..]) |b| try std.testing.expectEqual(@as(u8, 0), b);
}

test "sweep: batched epochs (t_cam_of) reproduce per-epoch sweeps — a two-point light curve" {
    const allocator = std.testing.allocator;
    const mp = metric.MetricParams{ .a = 0.0, .mksr0 = 0.1, .mksh0 = 0.9 };
    const consts = testConsts();
    const nx = 64;
    const ny = 32;

    // same flare scene as the retarded-time gate: blob at (r=10, th=pi/3)
    // lit only in the t=20 frame
    var zero = try zeroDump(allocator, nx, ny, 1);
    defer allocator.free(zero.body);
    var blob = try zeroDump(allocator, nx, ny, 1);
    defer allocator.free(blob.body);
    const g = puffyGrid(nx, ny, 1, mp, 1.85, 500.0);
    const uu = thermo.uFromTrho(&consts, 1.0e7, 1.0e-26, 5.0 / 3.0);
    const ee = consts.lteEfromT(1.0e7);
    for (0..ny) |iy| {
        const x2 = g.miny + (@as(f64, @floatFromInt(iy)) + 0.5) * g.dy;
        const th = forms.mks2Theta(Dual3.constant(x2), mp.mksh0).v;
        if (@abs(th - pi / 3.0) > 0.12) continue;
        for (0..nx) |ix| {
            const x1 = g.minx + (@as(f64, @floatFromInt(ix)) + 0.5) * g.dx;
            const r = @exp(x1) + mp.mksr0;
            if (r < 9.0 or r > 11.0) continue;
            const base = (iy * nx + ix) * L.count;
            blob.body[base + L.index(.rho)] = 1.0e-26;
            blob.body[base + L.index(.uu)] = uu;
            blob.body[base + L.index(.ee)] = ee;
        }
    }
    const ts = [_]f64{ 0, 5, 10, 15, 20, 25, 30, 35, 40 };
    var frames: [9]*const render.DumpData = @splat(&zero);
    frames[4] = &blob;
    var src = render.series.SliceSource{ .ts = ts[0..], .frames = frames[0..] };

    const scene = render.Scene.init(g, mp, consts, opacities.Channels.puffy, 5.0 / 3.0, &zero, 200.0, 500.0);
    var cam = render.Camera{ .r = 200, .incl_deg = 60, .fov = 2, .width = 1, .height = 1, .ss = 1 };
    cam.setup(mp);
    const base = try render.sweep.uniformPlan(allocator, &cam);
    defer allocator.free(base);

    const dt_travel = (cam.r - 10.0) + 4.0 * @log((cam.r - 2.0) / (10.0 - 2.0));
    const t_bright = 20.0 + dt_travel;
    const t_dark = t_bright + 12.0;

    // reference: two independent single-epoch sweeps
    var want: [2]f64 = undefined;
    for ([_]f64{ t_bright, t_dark }, 0..) |tc, e| {
        var o1 = [_]f64{0} ** 1;
        _ = try render.sweep.renderSlow(cfg, allocator, &scene, &cam, &src, base, o1[0..], .{}, .{ .t_cam = tc, .r_slow = 40.0 }, 1, false);
        want[e] = o1[0];
    }
    try std.testing.expect(want[0] > 0);
    try std.testing.expectEqual(@as(f64, 0), want[1]);

    // batched: both epochs in ONE sweep, each aimed at its own pixel
    var specs: [2]render.sweep.RaySpec = undefined;
    for (base, 0..) |sp, j| {
        specs[0 + j] = sp;
        var s2 = sp;
        s2.pix += 1;
        specs[1 + j] = s2;
    }
    const t_of = [_]f64{ t_bright, t_dark };
    var out = [_]f64{ -1, -1 };
    const stats = try render.sweep.renderSlow(cfg, allocator, &scene, &cam, &src, specs[0..], out[0..], .{}, .{ .t_cam = 0, .r_slow = 40.0, .t_cam_of = t_of[0..] }, 1, false);
    try std.testing.expectEqual(@as(usize, 2), stats.rays);

    // bit-identical to the per-epoch runs
    try std.testing.expectEqual(want[0], out[0]);
    try std.testing.expectEqual(want[1], out[1]);
}
