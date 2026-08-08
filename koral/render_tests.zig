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
//!    and the afmhot colormap.

const std = @import("std");
const render = @import("render/render.zig");
const image = @import("render/image.zig");
const metric = @import("metric/metric.zig");
const forms = @import("metric/forms.zig");
const grid_mod = @import("grid.zig");
const config = @import("config.zig");
const layout = @import("layout.zig");
const thermo = @import("physics/thermo.zig");
const opacities = @import("physics/opacities.zig");
const units_mod = @import("units.zig");
const Dual3 = @import("math/dual.zig").Dual3;

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
    const shadow = @import("render/shadow.zig");
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
    const shadow = @import("render/shadow.zig");
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
    const em = @import("render/emission.zig");
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
    const em = @import("render/emission.zig");
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
    const em = @import("render/emission.zig");
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
    const em = @import("render/emission.zig");
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
