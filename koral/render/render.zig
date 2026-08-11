//! GRRT renderer core: general-relativistic ray tracing + radiative
//! transfer over a KDMP snapshot, producing a camera-plane intensity map.
//!
//! Method (the standard unpolarized GRRT recipe of ipole/RAPTOR class
//! codes, specialized to KORAL's gray M1 data):
//!
//!  * Rays are launched from a distant static observer's orthonormal
//!    tetrad, one per pixel, and integrated BACKWARD (Δλ < 0, into the
//!    past) as null geodesics dk^μ/dλ = −Γ^μ_αβ k^α k^β, directly in the
//!    grid's MKS2 coordinates — Christoffels come exact-to-roundoff from
//!    metric.compute's dual numbers, and the log-r/concentrated-θ
//!    coordinates give near-horizon step refinement for free.
//!  * Along the ray the frequency-integrated transfer equation is
//!    accumulated in invariant form: with ν̂ = −k_μ u^μ the local photon
//!    frequency (camera-normalized to 1),
//!        I_cam = ∫ e^{−τ} (j/ν̂³) dλ,   dτ = χ ν̂ dλ,
//!    which bakes in gravitational redshift and Doppler beaming (g⁴ for
//!    the frequency-integrated intensity).
//!  * Emission and extinction come from the SAME microphysics the sim
//!    evolved (physics/opacities.zig): thermal emission κ_abs·B(T_e),
//!    plus scattering source κ_es·Ê/4π — the M1 radiation field the code
//!    solved for IS the local mean intensity, so the formal solution needs
//!    no iteration. Extinction χ = κ_abs(T_rad) + κ_es. All in GU; the
//!    image is normalized for display, so only ratios matter.
//!
//! The φ-wedge (PHIWEDGE = π/2, periodic) is tiled automatically by the
//! periodic wrap in `sample`; a 2D axisymmetric dump (nz = 1) renders
//! through the identical path (the revolve is the φ-independence of the
//! interpolant). Everything is problem-agnostic: Scene carries the grid,
//! metric parameters, thermo constants and opacity channels; only the CLI
//! wires in PUFFY.
//!
//! The integrator is RESUMABLE (advanceRay + RayState) and generic over a
//! sampler: the fast-light path binds the single snapshot (SnapshotSampler,
//! bit-identical to the old traceRay loop), while SLOW LIGHT binds a
//! time-interpolating sampler over a KDMP series and drives rays through a
//! streaming time window — see series.zig / sweep.zig, and adaptive.zig for
//! photon-ring image-plane refinement (docs/SLOWLIGHT_2026-08-11.md).

const std = @import("std");
const config = @import("../config.zig");
const layout = @import("../layout.zig");
const grid_mod = @import("../grid.zig");
const geometry = @import("../geometry.zig");
const metric = @import("../metric/metric.zig");
const forms = @import("../metric/forms.zig");
const Dual3 = @import("../math/dual.zig").Dual3;
const relele = @import("../relele.zig");
const radiation = @import("../physics/radiation.zig");
const mhd = @import("../physics/bfield.zig");
const thermo = @import("../physics/thermo.zig");
const opacities = @import("../physics/opacities.zig");
const dump = @import("../io/dump.zig");

pub const image = @import("image.zig");
pub const emission = @import("emission.zig");
pub const shadow = @import("shadow.zig");
pub const series = @import("series.zig");
pub const sweep = @import("sweep.zig");
pub const adaptive = @import("adaptive.zig");
pub const fits = @import("fits.zig");

pub const Grid = grid_mod.Grid;
pub const MetricParams = metric.MetricParams;
pub const Geometry = geometry.Geometry;

// ---- dump payload ----------------------------------------------------------

pub const LooseHeader = struct { h: dump.DumpHeader, body_off: usize };

/// Parse a KDMP header, accepting v2 (the restart contract) AND the legacy
/// v1 (32-byte header, no nstep/out_idx): rendering is read-only, so the
/// strict version gate of io/dump.zig — which protects restart bookkeeping —
/// does not apply here. Shared by DumpData.fromBytes and the slow-light
/// series scanner (series.zig), which reads headers without bodies.
pub fn parseHeaderLoose(bytes: []const u8) !LooseHeader {
    const h = dump.parseDumpHeader(bytes) catch |err| switch (err) {
        error.BadVersion => {
            const v1_header = 32;
            if (bytes.len < v1_header) return error.Truncated;
            if (std.mem.readInt(u32, bytes[4..8], .little) != 1) return err;
            return .{ .body_off = v1_header, .h = .{
                .nx = std.mem.readInt(u32, bytes[8..12], .little),
                .ny = std.mem.readInt(u32, bytes[12..16], .little),
                .nz = std.mem.readInt(u32, bytes[16..20], .little),
                .nv = std.mem.readInt(u32, bytes[20..24], .little),
                .t = @bitCast(std.mem.readInt(u64, bytes[24..32], .little)),
                .nstep = 0,
                .out_idx = 0,
            } };
        },
        else => return err,
    };
    return .{ .h = h, .body_off = dump.header_size };
}

/// A KDMP snapshot held as a flat primitive array p[iz][iy][ix][iv]
/// (iv fastest — io/dump.zig's body order), decoupled from Sim so the
/// renderer needs no ghosts, comm or solver state.
pub const DumpData = struct {
    header: dump.DumpHeader,
    body: []f64,

    pub fn fromBytes(allocator: std.mem.Allocator, bytes: []const u8) !DumpData {
        const lh = try parseHeaderLoose(bytes);
        const h = lh.h;
        const body_off = lh.body_off;
        const n = @as(usize, h.nx) * h.ny * h.nz * h.nv;
        if (bytes.len < body_off + n * 8) return error.Truncated;
        const body = try allocator.alloc(f64, n);
        errdefer allocator.free(body);
        for (body, 0..) |*v, i| {
            v.* = @bitCast(std.mem.readInt(u64, bytes[body_off + 8 * i ..][0..8], .little));
        }
        return .{ .header = h, .body = body };
    }

    pub fn deinit(d: *DumpData, allocator: std.mem.Allocator) void {
        allocator.free(d.body);
        d.* = undefined;
    }

    pub inline fn cell(d: *const DumpData, ix: usize, iy: usize, iz: usize) []const f64 {
        const nv = d.header.nv;
        const idx = ((iz * d.header.ny + iy) * d.header.nx + ix) * nv;
        return d.body[idx..][0..nv];
    }
};

// ---- scene -----------------------------------------------------------------

/// Mask parameters for floor-dominated cells: emission is zeroed where
/// ρ < factor·rho0·(r/r0)^power — the problem's floor-atmosphere density
/// profile scaled by `factor` (for PUFFY: rho0 = RHOATMMIN, r0 = 2,
/// power = −1.5, see setHdAtmosphere).
pub const FloorCut = struct {
    rho0: f64,
    r0: f64,
    power: f64,
    factor: f64,
};

/// Everything a ray needs: geometry of the data grid, spacetime, thermo
/// constants (mass-scaled), opacity channels, and the snapshot itself.
pub const Scene = struct {
    grid: Grid,
    mp: MetricParams,
    consts: thermo.Consts,
    channels: opacities.Channels,
    /// gas adiabatic index (for T(u,ρ))
    gam: f64,
    data: *const DumpData,
    /// rays ending below this KS radius are captured (just above r_horizon)
    r_capture: f64,
    /// rays beyond this KS radius are done (past the camera, out of matter)
    r_escape: f64,
    /// electron scattering on/off — mirror of the problem's `scattering`
    /// flag (C: PR_KAPPAES). The AGN preset runs with κ_es ≡ 0; rendering
    /// its dumps with Thomson extinction would overcount χ relative to the
    /// sim's own physics.
    scattering: bool = true,
    /// standard EHT σ-cut: zero emission where b²/ρ > sigma_cut (0 = off).
    /// Floor-dominated funnel material has unreliable thermodynamics.
    sigma_cut: f64 = 0,
    /// zero emission from cells within `factor`× of the floor-atmosphere
    /// density profile (null = off). Extinction is kept either way — masked
    /// material still occults, only its (unphysical) glow is dropped.
    floor: ?FloorCut = null,

    pub fn init(g: Grid, mp: MetricParams, consts: thermo.Consts, channels: opacities.Channels, gam: f64, data: *const DumpData, r_cam: f64, rmax: f64) Scene {
        return .{
            .grid = g,
            .mp = mp,
            .consts = consts,
            .channels = channels,
            .gam = gam,
            .data = data,
            .r_capture = 1.02 * metric.rHorizonBL(mp.a),
            .r_escape = 1.05 * @max(r_cam, rmax),
        };
    }
};

// ---- geometry helpers ------------------------------------------------------

/// Assemble the kernel-facing Geometry (C's 4×5 layout — geometry.zig) from
/// one metric.compute result. The renderer's coordinates are always MKS2.
pub fn geomFromCoordData(x: [4]f64, cd: *const metric.CoordData) Geometry {
    var gg: [4][5]f64 = undefined;
    var GG: [4][5]f64 = undefined;
    for (0..4) |i| {
        for (0..4) |j| {
            gg[i][j] = cd.gcov[i][j];
            GG[i][j] = cd.gcon[i][j];
        }
        GG[i][4] = 0;
    }
    gg[0][4] = cd.dlgdet[0];
    gg[1][4] = cd.dlgdet[1];
    gg[2][4] = cd.dlgdet[2];
    gg[3][4] = cd.gdet;
    GG[3][4] = cd.gttpert;
    return .{
        .coords = .mks2,
        .xxvec = .{ 0, x[1], x[2], x[3] },
        .gg = gg,
        .GG = GG,
        .gdet = cd.gdet,
        .alpha = 1.0 / @sqrt(-cd.gcon[0][0]),
        .gttpert = cd.gttpert,
    };
}

inline fn dotg(gcov: [4][4]f64, u: [4]f64, v: [4]f64) f64 {
    var s: f64 = 0;
    for (0..4) |i| {
        for (0..4) |j| s += gcov[i][j] * u[i] * v[j];
    }
    return s;
}

/// Orthonormal tetrad e[a][μ]: e[0] the static observer's 4-velocity,
/// e[1] outward radial, e[2] +θ (equatorward), e[3] +φ. Gram–Schmidt runs
/// in the order (t, φ, r, θ): orthogonalizing the radial trial vector
/// against ∂_φ is essential in Kerr–Schild coordinates, where
/// g_rφ → −a·sin²θ stays O(a) at ANY radius — a (t, r, θ, φ) ordering
/// leaves the "radial" direction carrying azimuthal momentum, which shifts
/// every launched ray's impact parameter by ~a·sinθ (caught by the Bardeen
/// shadow-overlay validation). Valid outside the ergosphere (∂_t timelike)
/// — always true for a distant camera.
pub fn tetradFromMetric(gcov: [4][4]f64) [4][4]f64 {
    // build order: t, φ, r, θ — indexed via the coordinate axis of each slot
    const axis = [4]usize{ 0, 3, 1, 2 };
    var f: [4][4]f64 = @splat(@splat(0));
    for (0..4) |a| f[a][axis[a]] = 1;
    for (0..4) |a| {
        for (0..a) |b| {
            const eta_bb: f64 = if (b == 0) -1.0 else 1.0;
            const proj = eta_bb * dotg(gcov, f[a], f[b]);
            for (0..4) |mu| f[a][mu] -= proj * f[b][mu];
        }
        const n2 = dotg(gcov, f[a], f[a]);
        const norm = @sqrt(@abs(n2));
        for (0..4) |mu| f[a][mu] /= norm;
    }
    // return in (t, r, θ, φ) slot order
    return .{ f[0], f[2], f[3], f[1] };
}

// ---- data sampling ---------------------------------------------------------

/// Trilinear sample of the primitives at MKS2 point x. Radially outside the
/// data → false (vacuum). θ clamps (the grid reaches x2 = 0.001/0.999,
/// essentially the poles); φ wraps periodically over the wedge, which both
/// tiles a 3D wedge to 2π and revolves a 2D (nz = 1) slice.
pub fn sample(comptime cfg: config.Config, s: *const Scene, x: [4]f64, pp: *[layout.VarLayout(cfg).count]f64) bool {
    return sampleData(cfg, &s.grid, s.data, x, pp);
}

/// The sampling kernel with the snapshot made explicit, so time-aware
/// samplers (series.zig) can interpolate between two of them. x[0] is
/// ignored here — one snapshot has no time axis.
pub fn sampleData(comptime cfg: config.Config, g: *const Grid, data: *const DumpData, x: [4]f64, pp: *[layout.VarLayout(cfg).count]f64) bool {
    const L = layout.VarLayout(cfg);
    const nx = g.nx;
    const ny = g.ny;
    const nz = g.nz;

    const fx = (x[1] - g.minx) / g.dx - 0.5;
    if (fx < -0.5 or fx > @as(f64, @floatFromInt(nx)) - 0.5) return false;
    const fy = (x[2] - g.miny) / g.dy - 0.5;

    const cx = std.math.clamp(fx, 0, @as(f64, @floatFromInt(nx - 1)));
    const cy = std.math.clamp(fy, 0, @as(f64, @floatFromInt(ny - 1)));
    const ix0: usize = @intFromFloat(@floor(cx));
    const iy0: usize = @intFromFloat(@floor(cy));
    const ix1 = @min(ix0 + 1, nx - 1);
    const iy1 = @min(iy0 + 1, ny - 1);
    const tx = cx - @floor(cx);
    const ty = cy - @floor(cy);

    var iz0: usize = 0;
    var iz1: usize = 0;
    var tz: f64 = 0;
    if (nz > 1) {
        const nzf: f64 = @floatFromInt(nz);
        const fz = @mod((x[3] - g.minz) / g.dz - 0.5, nzf);
        iz0 = @intFromFloat(@floor(fz));
        iz0 = @min(iz0, nz - 1);
        iz1 = (iz0 + 1) % nz;
        tz = fz - @floor(fz);
    }

    const c000 = data.cell(ix0, iy0, iz0);
    const c100 = data.cell(ix1, iy0, iz0);
    const c010 = data.cell(ix0, iy1, iz0);
    const c110 = data.cell(ix1, iy1, iz0);
    const c001 = data.cell(ix0, iy0, iz1);
    const c101 = data.cell(ix1, iy0, iz1);
    const c011 = data.cell(ix0, iy1, iz1);
    const c111 = data.cell(ix1, iy1, iz1);

    for (0..L.count) |iv| {
        const v00 = c000[iv] * (1 - tx) + c100[iv] * tx;
        const v10 = c010[iv] * (1 - tx) + c110[iv] * tx;
        const v01 = c001[iv] * (1 - tx) + c101[iv] * tx;
        const v11 = c011[iv] * (1 - tx) + c111[iv] * tx;
        const v0 = v00 * (1 - ty) + v10 * ty;
        const v1 = v01 * (1 - ty) + v11 * ty;
        pp[iv] = v0 * (1 - tz) + v1 * tz;
    }
    return true;
}

// ---- local emission/extinction ---------------------------------------------

pub const LocalState = struct {
    /// gas lab-frame 4-velocity
    ucon: [4]f64,
    /// thermal frequency-integrated emissivity κ_abs·B(T_e) [GU]
    j_therm: f64,
    /// gray extinction χ = κ_abs(T_rad) + κ_es [GU: 1/length]
    chi: f64,
    /// electron-scattering extinction alone [GU: 1/length]
    chi_es: f64,
    /// fluid-frame radiation energy density Ê [GU]
    ehat: f64,
    /// fluid-frame radiation flux F̂^μ = −R^μ_ν u^ν − Ê u^μ (⊥ u) [GU]
    fhat: [4]f64,
    /// emission masked (σ-cut / floor-cut)? Extinction still applies.
    masked: bool,
    // raw state for the monochromatic microphysics
    rho: f64,
    te: f64,
    trad: f64,
    /// electron number density [GU]
    ne: f64,
    bsq: f64,
    /// magnetic 4-vector b^μ (⊥ u) for the photon–B pitch angle
    bcon: [4]f64,
};

/// Emission/extinction state at one sampled point, following radforce.zig's
/// opacity-state recipe: u^μ from the VELR primitives, b² from B, Ê and F̂
/// from the M1 R^μν, T_rad = LTE(Ê), then the full opacity chain. The
/// direction-dependent pieces (M1 dipole, monochromatic j_ν/χ_ν) are
/// assembled per-ray in traceRay from these ingredients. Returns null for
/// vacuum/invalid samples (floor atmosphere handles itself: its tiny ρ
/// gives negligible j and χ).
pub fn localState(comptime cfg: config.Config, s: *const Scene, geom: *const Geometry, pp: [layout.VarLayout(cfg).count]f64) ?LocalState {
    const L = layout.VarLayout(cfg);
    const rho = pp[L.index(.rho)];
    const uu = pp[L.index(.uu)];
    if (!(rho > 0) or !(uu > 0) or !std.math.isFinite(rho) or !std.math.isFinite(uu)) return null;

    const u = relele.uconUcovFromPrims(
        .{ pp[L.index(.vx)], pp[L.index(.vy)], pp[L.index(.vz)] },
        geom,
    ) catch return null;

    var bsq: f64 = 0;
    var bcon: [4]f64 = @splat(0);
    if (comptime L.hasVar(.b1)) {
        const b = mhd.bconBcovBsqFrom4vel(
            .{ pp[L.index(.b1)], pp[L.index(.b2)], pp[L.index(.b3)] },
            u.con,
            u.cov,
            geom,
        );
        bsq = b.bsq;
        bcon = b.bcon;
    }

    const temps = thermo.tempsFromUrho(&s.consts, uu, rho, s.gam);
    const ne = thermo.thermalNe(&s.consts, rho);

    // fluid-frame radiation energy Ê = R^αβ u_α u_β and flux
    // F̂^μ = −R^μ_ν u^ν − Ê u^μ (radforce.zig recipe + first moment)
    var ehat: f64 = 0;
    var fhat: [4]f64 = @splat(0);
    var trad = temps.te;
    var tradbb = temps.te;
    if (comptime L.hasVar(.ee)) {
        const rij = radiation.calcRijG(cfg, f64, pp, geom.cov(), geom.con());
        for (0..4) |i| {
            for (0..4) |j| ehat += rij[i][j] * u.cov[i] * u.cov[j];
        }
        if (!(ehat > 0) or !std.math.isFinite(ehat)) ehat = 0;
        const rud = relele.lowerSecond(rij, geom);
        for (0..4) |mu| {
            var s_: f64 = 0;
            for (0..4) |nu| s_ -= rud[mu][nu] * u.con[nu];
            fhat[mu] = s_ - ehat * u.con[mu];
            if (!std.math.isFinite(fhat[mu])) fhat[mu] = 0;
        }
        tradbb = s.consts.lteTfromE(@max(ehat, 1e-300));
        trad = tradbb;
    }

    const kes = if (s.scattering) opacities.kappaEsPuffy(&s.consts, s.channels, rho, trad) else 0.0;
    const kr = opacities.calcKappaFromState(&s.consts, s.channels, .{
        .rho = rho,
        .tgas = temps.tgas,
        .te = temps.te,
        .trad = trad,
        .tradbb = tradbb,
        .ne = ne,
        .bsq = bsq,
    });

    // Floor masks: σ-cut (magnetized funnel) and atmosphere-profile cut.
    // Masked cells keep their extinction but contribute no emission.
    var masked = false;
    if (s.sigma_cut > 0 and bsq > s.sigma_cut * rho) masked = true;
    if (s.floor) |fc| {
        const r = @exp(geom.xxvec[1]) + s.mp.mksr0;
        if (rho < fc.factor * fc.rho0 * std.math.pow(f64, r / fc.r0, fc.power)) masked = true;
    }

    // thermal source B(T_e) = (σ/π)T⁴; the scattering source is assembled
    // per-ray in traceRay (it carries the M1 dipole, so it depends on the
    // photon direction).
    const planck = s.consts.sigma_rad_over_pi * temps.te * temps.te * temps.te * temps.te;
    const j_therm = kr.opac.gas_abs * planck;
    const chi = kr.opac.rad_abs + kes;
    if (!std.math.isFinite(j_therm) or !std.math.isFinite(chi)) return null;

    return .{
        .ucon = u.con,
        .j_therm = j_therm,
        .chi = chi,
        .chi_es = kes,
        .ehat = ehat,
        .fhat = fhat,
        .masked = masked,
        .rho = rho,
        .te = temps.te,
        .trad = trad,
        .ne = ne,
        .bsq = bsq,
        .bcon = bcon,
    };
}

// ---- geodesic integration --------------------------------------------------

pub const TraceOpts = struct {
    /// step size in units of the local grid cell (RK4; 0.5 ≈ half a cell)
    eps: f64 = 0.5,
    /// stop once the accumulated optical depth exceeds this
    tau_max: f64 = 30.0,
    max_steps: usize = 200_000,
    /// observed frequency [Hz]. 0 = gray frequency-integrated transfer
    /// (intensity in GU, invariant I/ν̂⁴). > 0 = monochromatic transfer at
    /// this camera frequency (invariant I_ν/ν̂³, microphysics from
    /// emission.zig): the traced value is I_ν at the camera in CGS
    /// [erg/s/cm²/sr/Hz], ready for brightness temperatures and fluxes.
    nu_obs: f64 = 0,
    /// Validation mode: ignore the fluid entirely and render a bright
    /// celestial sphere — escaped rays return 1, captured rays 0. The
    /// dark region's boundary is the analytic Kerr shadow (shadow.zig).
    screen: bool = false,
};

pub const TraceResult = struct {
    /// camera-frame frequency-integrated intensity (GU; ν̂_cam = 1)
    intensity: f64,
    tau: f64,
    steps: usize,
    captured: bool,
    /// ray endpoint (for conservation checks / diagnostics)
    x: [4]f64,
    k: [4]f64,
};

fn accel(kris: *const [4][4][4]f64, k: [4]f64) [4]f64 {
    var a: [4]f64 = @splat(0);
    for (0..4) |mu| {
        var acc: f64 = 0;
        for (0..4) |al| {
            for (0..4) |be| acc += kris[mu][al][be] * k[al] * k[be];
        }
        a[mu] = -acc;
    }
    return a;
}

const StepResult = struct { x: [4]f64, k: [4]f64, ok: bool };

/// One RK4 step of the geodesic ODE, REJECTED (ok = false) if any stage
/// or the endpoint leaves the MKS2 chart x2 ∈ (0,1). No reflection happens
/// here: mixing stages from both sides of a pole in one RK4 combination is
/// what produced the meridian artifact — a rejected step falls back to
/// flatPoleStep, which crosses the axis in a regular chart.
fn rk4Step(mp: MetricParams, cd1: *const metric.CoordData, x: [4]f64, k: [4]f64, dl: f64) StepResult {
    const a1 = accel(&cd1.kris, k);

    var x2 = x;
    var k2 = k;
    for (0..4) |i| {
        x2[i] += 0.5 * dl * k[i];
        k2[i] += 0.5 * dl * a1[i];
    }
    if (x2[2] <= 0 or x2[2] >= 1) return .{ .x = x, .k = k, .ok = false };
    const cd2 = metric.compute(.mks2, mp, x2);
    const a2 = accel(&cd2.kris, k2);

    var x3 = x;
    var k3 = k;
    for (0..4) |i| {
        x3[i] += 0.5 * dl * k2[i];
        k3[i] += 0.5 * dl * a2[i];
    }
    if (x3[2] <= 0 or x3[2] >= 1) return .{ .x = x, .k = k, .ok = false };
    const cd3 = metric.compute(.mks2, mp, x3);
    const a3 = accel(&cd3.kris, k3);

    var x4 = x;
    var k4 = k;
    for (0..4) |i| {
        x4[i] += dl * k3[i];
        k4[i] += dl * a3[i];
    }
    if (x4[2] <= 0 or x4[2] >= 1) return .{ .x = x, .k = k, .ok = false };
    const cd4 = metric.compute(.mks2, mp, x4);
    const a4 = accel(&cd4.kris, k4);

    const w = dl / 6.0;
    var xn = x;
    var kn = k;
    for (0..4) |i| {
        xn[i] += w * (k[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        kn[i] += w * (a1[i] + 2.0 * a2[i] + 2.0 * a3[i] + a4[i]);
    }
    if (xn[2] <= 0 or xn[2] >= 1) return .{ .x = x, .k = k, .ok = false };
    return .{ .x = xn, .k = kn, .ok = true };
}

/// x2-distance-to-pole floor in the step controller: below it h stops
/// shrinking, so the ray crosses in a few more steps instead of creeping
/// forever (Zeno), and the flat hop's crossing chord stays within
/// θ ~ dθ/dx2·pole_floor ~ 6e-4 rad, where its O(θ²) elementary-flatness
/// error is ~4e-7.
const pole_floor = 2.0e-5;

/// Cross the polar axis with ONE short chord in locally flat transverse
/// coordinates q = θ·(cos φ, sin φ). Kerr is elementarily flat on the
/// axis (g_θθ = Σ, g_φφ = Σ sin²θ + O(θ⁴), g_tφ = O(θ²)), so over a
/// short chord the geodesic is a straight line in q to O(θ²) and the
/// crossing needs no reflection at all — θ, φ and their momenta re-emerge
/// from atan2 on the far side. This is a CROSSING device, not a regional
/// integrator: the step controller keeps RK4 valid arbitrarily close to
/// the pole (Δθ ≪ θ), and only the final tiny chord goes through here.
/// The south pole maps onto the north-pole code by the x2 → 1−x2 mirror.
fn flatPoleStep(mp: MetricParams, cd: *const metric.CoordData, x: [4]f64, k: [4]f64, dl: f64) StepResult {
    // The (t, x1) sector stays fully dynamical: its acceleration components
    // are regular on the axis (the cot θ divergence lives only in the φ
    // rows, which the q-propagation replaces), and freezing k⁰/k¹ would
    // accumulate a secular energy error on rays that hug the axis.
    const a = accel(&cd.kris, k);
    const k0n = k[0] + dl * a[0];
    const k1n = k[1] + dl * a[1];

    const south = x[2] > 0.5;
    var x2 = x[2];
    var k2 = k[2];
    if (south) {
        x2 = 1.0 - x2;
        k2 = -k2;
    }

    const th = forms.mks2Theta(Dual3.constant(x2), mp.mksh0).v;
    const dthdx2 = forms.mks2DThetaDx2Metric(Dual3.constant(x2), mp.mksh0).v;
    const kth = dthdx2 * k2;
    const c = @cos(x[3]);
    const sn = @sin(x[3]);
    const q1 = th * c;
    const q2 = th * sn;
    const qd1 = kth * c - th * sn * k[3];
    const qd2 = kth * sn + th * c * k[3];

    const q1n = q1 + dl * qd1;
    const q2n = q2 + dl * qd2;
    const thn = @max(@sqrt(q1n * q1n + q2n * q2n), 1e-14);
    const phin = std.math.atan2(q2n, q1n);
    // conformal rescale: the transverse metric is Σ(dq₁²+dq₂²) with
    // Σ ≈ r²+a² on the axis, and the conserved quantity through the chord
    // is the momentum Σ·q̇ — a radially plunging near-axis ray changes r
    // across the step, and freezing q̇ instead leaks null-norm error.
    const r_a = @exp(x[1]) + mp.mksr0;
    const r_b = @exp(x[1] + dl * 0.5 * (k[1] + k1n)) + mp.mksr0;
    const scale = (r_a * r_a + mp.a * mp.a) / (r_b * r_b + mp.a * mp.a);
    const kthn = scale * (qd1 * q1n + qd2 * q2n) / thn;
    const k3n = scale * (qd2 * q1n - qd1 * q2n) / (thn * thn);

    var x2n = forms.mks2X2FromTheta(thn, mp.mksh0);
    const dthdx2n = forms.mks2DThetaDx2Metric(Dual3.constant(x2n), mp.mksh0).v;
    var k2n = kthn / dthdx2n;
    if (south) {
        x2n = 1.0 - x2n;
        k2n = -k2n;
    }

    return .{
        .x = .{ x[0] + dl * 0.5 * (k[0] + k0n), x[1] + dl * 0.5 * (k[1] + k1n), x2n, phin },
        .k = .{ k0n, k1n, k2n, k3n },
        .ok = true,
    };
}

/// Grid-scaled step: ~eps cells of motion per step in the tightest
/// dimension. MKS2's log-r makes this an adaptive Δr automatically. The
/// pole term keeps Δx2 ≲ 10% of the distance to the pole (down to
/// pole_floor), so RK4 resolves the Γ ~ cot θ growth instead of stepping
/// across it — the source of the old meridian artifact.
fn stepSize(g: *const Grid, x: [4]f64, k: [4]f64, eps: f64) f64 {
    const a1 = @abs(k[1]) / @max(g.dx, 0.02);
    const a2 = @abs(k[2]) / @max(g.dy, 0.005);
    const a3 = @abs(k[3]) / @max(g.dz, 0.02);
    const dist_pole = @max(@min(x[2], 1.0 - x[2]), pole_floor);
    const ap = @abs(k[2]) / (0.1 * dist_pole);
    return eps / (a1 + a2 + a3 + ap + 1e-12);
}

/// Why a ray stopped advancing. `.active` means "paused by the controller"
/// (slow-light sweep parking); everything else is terminal.
pub const RayStatus = enum(u8) {
    active,
    /// fell below r_capture (into the hole)
    captured,
    /// passed r_escape (out of the scene)
    escaped,
    /// accumulated optical depth exceeded tau_max
    thick,
    /// ran out of the step budget
    exhausted,
};

/// One ray's full integration state — everything traceRay used to keep in
/// locals, so a trace can be paused and resumed (the slow-light sweep parks
/// rays at time-window boundaries and resumes them when the window slides).
pub const RayState = struct {
    x: [4]f64,
    k: [4]f64,
    intensity: f64 = 0,
    tau: f64 = 0,
    steps: usize = 0,
    status: RayStatus = .active,
};

/// Never pause: run the ray to a terminal status (the fast-light behavior).
pub const NeverPause = struct {
    pub inline fn pause(_: NeverPause, _: *const Scene, _: *const RayState, _: f64) bool {
        return false;
    }
};

/// The fast-light sampler: the Scene's single snapshot, no time axis.
pub const SnapshotSampler = struct {
    pub inline fn sample(_: SnapshotSampler, comptime c: config.Config, s: *const Scene, x: [4]f64, pp: *[layout.VarLayout(c).count]f64) bool {
        return sampleData(c, &s.grid, s.data, x, pp);
    }
};

/// Advance one ray backward (Δλ < 0, into the past) until it terminates —
/// capture, escape, τ > tau_max, step budget — or `ctl.pause(s, st, r)`
/// returns true (status stays .active; the caller may resume later, the
/// state carries everything). `sampler.sample(cfg, s, x, pp)` supplies the
/// primitives at the sampled EVENT — x includes the coordinate time x[0],
/// which a time-aware sampler (series.WindowSampler) interpolates on and
/// the fast-light SnapshotSampler ignores.
///
/// This is traceRay's old loop verbatim (same operation order, so the
/// fast-light path stays bit-identical); only the control seams are new.
pub fn advanceRay(comptime cfg: config.Config, s: *const Scene, sampler: anytype, st: *RayState, opts: TraceOpts, ctl: anytype) void {
    while (true) {
        if (st.steps >= opts.max_steps) {
            st.status = .exhausted;
            return;
        }
        const r = @exp(st.x[1]) + s.mp.mksr0;
        if (r < s.r_capture) {
            st.status = .captured;
            return;
        }
        if (r > s.r_escape) {
            st.status = .escaped;
            return;
        }
        if (st.tau > opts.tau_max) {
            st.status = .thick;
            return;
        }
        if (ctl.pause(s, st, r)) return;

        const cd = metric.compute(.mks2, s.mp, st.x);

        // -- geodesic step with Δλ = −h (into the past): RK4 everywhere;
        // a step that would cross a pole (rejected) becomes one short
        // flat-chart chord through the axis instead --
        const h = stepSize(&s.grid, st.x, st.k, opts.eps);
        var next = rk4Step(s.mp, &cd, st.x, st.k, -h);
        if (!next.ok) next = flatPoleStep(s.mp, &cd, st.x, st.k, -h);

        // -- radiative transfer over the ACCEPTED segment length h --
        var pp: [layout.VarLayout(cfg).count]f64 = undefined;
        if (!opts.screen and sampler.sample(cfg, s, st.x, &pp)) {
            const geom = geomFromCoordData(st.x, &cd);
            if (localState(cfg, s, &geom, pp)) |lst| {
                var kcov: [4]f64 = @splat(0);
                for (0..4) |i| {
                    for (0..4) |j| kcov[i] += cd.gcov[i][j] * st.k[j];
                }
                var nuhat: f64 = 0;
                for (0..4) |i| nuhat -= kcov[i] * lst.ucon[i];
                if (nuhat > 1e-12) {
                    // M1 dipole for the scattering source: the field seen
                    // along the photon direction n̂ is J·(1 + 3 n̂·F̂/Ê) in
                    // the Eddington expansion, with n̂·F̂ = k_μF̂^μ/ν̂
                    // (F̂ ⊥ u). Clamped at 0 — the linear expansion can go
                    // negative at free-streaming flux factors.
                    var kdotf: f64 = 0;
                    for (0..4) |i| kdotf += kcov[i] * lst.fhat[i];
                    const dip = if (lst.ehat > 1e-300)
                        @max(0.0, 1.0 + 3.0 * kdotf / (nuhat * lst.ehat))
                    else
                        0.0;

                    var jl: f64 = undefined;
                    var chil: f64 = undefined;
                    var dl = nuhat * h; // fluid-frame path length (GU)
                    var nup = nuhat * nuhat * nuhat * nuhat;
                    if (opts.nu_obs > 0) {
                        const un = &s.consts.units;
                        var sin_pitch: f64 = 0;
                        if (lst.bsq > 1e-30) {
                            var kdotb: f64 = 0;
                            for (0..4) |i| kdotb += kcov[i] * lst.bcon[i];
                            const cosp = kdotb / (nuhat * @sqrt(lst.bsq));
                            sin_pitch = @sqrt(@max(0.0, 1.0 - cosp * cosp));
                        }
                        const em = emission.monoJChi(.{
                            .nu = opts.nu_obs * nuhat,
                            .ne_cgs = lst.ne * s.consts.numdensgu2cgs,
                            .ni_cgs = lst.rho * s.consts.one_over_mui_mp * s.consts.numdensgu2cgs,
                            .te = lst.te,
                            .trad = lst.trad,
                            .b_gauss = @sqrt(s.consts.fourmpi * un.endenGu2Cgs(lst.bsq)),
                            .sin_pitch = sin_pitch,
                            .chi_es_cgs = lst.chi_es / un.masscm,
                            .dip = dip,
                        });
                        jl = if (lst.masked) 0.0 else em.j;
                        chil = em.chi;
                        dl *= un.masscm; // path in cm for the CGS opacities
                        nup = nuhat * nuhat * nuhat; // I_ν/ν³ invariant
                    } else {
                        jl = if (lst.masked) 0.0 else lst.j_therm + lst.chi_es * (lst.ehat / s.consts.fourmpi) * dip;
                        chil = lst.chi;
                    }

                    const dtau = chil * dl;
                    const att = @exp(-st.tau);
                    if (dtau > 1e-8) {
                        st.intensity += att * (jl / (chil * nup)) * (1.0 - @exp(-dtau));
                    } else {
                        st.intensity += att * (jl / nup) * dl;
                    }
                    st.tau += dtau;
                }
            }
        }

        st.x = next.x;
        st.k = next.k;
        st.steps += 1;
    }
}

/// Package a finished RayState as the caller-facing TraceResult.
fn packResult(st: *const RayState, opts: TraceOpts) TraceResult {
    var intensity = st.intensity;
    if (opts.screen) intensity = if (st.status == .captured) 0.0 else 1.0;
    return .{
        .intensity = intensity,
        .tau = st.tau,
        .steps = st.steps,
        .captured = st.status == .captured,
        .x = st.x,
        .k = st.k,
    };
}

/// Trace one ray start-to-finish with an arbitrary sampler (slow light:
/// series.WindowSampler). Same contract as traceRay otherwise.
pub fn traceRayWith(comptime cfg: config.Config, s: *const Scene, sampler: anytype, x0: [4]f64, k0: [4]f64, opts: TraceOpts) TraceResult {
    var st = RayState{ .x = x0, .k = k0 };
    advanceRay(cfg, s, sampler, &st, opts, NeverPause{});
    return packResult(&st, opts);
}

/// Trace one ray backward from (x0, k0) at the camera, accumulating the
/// attenuated-emission formal solution. k0 must be future-directed null
/// with −k·u_cam = 1 (Camera.ray guarantees this).
pub fn traceRay(comptime cfg: config.Config, s: *const Scene, x0: [4]f64, k0: [4]f64, opts: TraceOpts) TraceResult {
    return traceRayWith(cfg, s, SnapshotSampler{}, x0, k0, opts);
}

// ---- camera ----------------------------------------------------------------

/// Pinhole camera on a static observer at (r, θ = incl, φ). `fov` is the
/// full field of view in GM/c² at the hole's distance; pixel (0,0) is the
/// top-left, screen-up points to the spin axis' north pole.
pub const Camera = struct {
    r: f64 = 1000.0,
    incl_deg: f64 = 60.0,
    phi: f64 = 0.0,
    /// full image width at the black hole, in M
    fov: f64 = 100.0,
    width: usize = 512,
    height: usize = 512,
    /// antialiasing: ss×ss rays per pixel, box-averaged. The sub-pixel
    /// photon ring aliases badly at 1; 2 (4 rays) is usually enough.
    ss: usize = 1,

    x0: [4]f64 = undefined,
    e: [4][4]f64 = undefined,

    pub fn setup(cam: *Camera, mp: MetricParams) void {
        const th = std.math.clamp(
            cam.incl_deg * std.math.pi / 180.0,
            0.05,
            std.math.pi - 0.05,
        );
        cam.x0 = .{
            0,
            @log(cam.r - mp.mksr0),
            forms.mks2X2FromTheta(th, mp.mksh0),
            cam.phi,
        };
        const cd = metric.compute(.mks2, mp, cam.x0);
        cam.e = tetradFromMetric(cd.gcov);
    }

    /// Future-directed null k^μ of the photon ARRIVING at pixel center
    /// (px,py) — see rayAt.
    pub fn ray(cam: *const Camera, px: usize, py: usize) [4]f64 {
        return cam.rayAt(@as(f64, @floatFromInt(px)) + 0.5, @as(f64, @floatFromInt(py)) + 0.5);
    }

    /// Ray at continuous image coordinates (fx, fy) ∈ [0,W]×[0,H] (pixel
    /// centers at half-integers): dominantly outward (+e1), tilted opposite
    /// the pixel's sky offset, normalized so −k·u_cam = 1.
    pub fn rayAt(cam: *const Camera, fx: f64, fy: f64) [4]f64 {
        const fovrad = cam.fov / cam.r;
        const wf: f64 = @floatFromInt(cam.width);
        const hf: f64 = @floatFromInt(cam.height);
        const ax = fovrad * (fx / wf - 0.5);
        const ay = fovrad * (0.5 - fy / hf);
        var d: [3]f64 = .{ @cos(ax) * @cos(ay), @sin(ay), -@sin(ax) };
        const n = @sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
        for (&d) |*v| v.* /= n;
        var k: [4]f64 = undefined;
        for (0..4) |mu| {
            k[mu] = cam.e[0][mu] + d[0] * cam.e[1][mu] + d[1] * cam.e[2][mu] + d[2] * cam.e[3][mu];
        }
        return k;
    }
};

// ---- full-frame render -----------------------------------------------------

/// Render the camera's frame into `out` (row-major, width×height), one
/// ray per pixel, work-stealing rows across `nthreads` OS threads (the
/// main thread participates; failed spawns just narrow the team).
pub fn renderImage(comptime cfg: config.Config, s: *const Scene, cam: *const Camera, out: []f64, opts: TraceOpts, nthreads: usize) void {
    std.debug.assert(out.len == cam.width * cam.height);
    var next_row: std.atomic.Value(usize) = .init(0);

    const Worker = struct {
        fn run(sc: *const Scene, c: *const Camera, o: []f64, op: TraceOpts, rows: *std.atomic.Value(usize)) void {
            const ssn = @max(c.ss, 1);
            const ssf: f64 = @floatFromInt(ssn);
            const inv = 1.0 / (ssf * ssf);
            while (true) {
                const py = rows.fetchAdd(1, .monotonic);
                if (py >= c.height) return;
                for (0..c.width) |px| {
                    var acc: f64 = 0;
                    for (0..ssn) |sy| {
                        for (0..ssn) |sx| {
                            const fx = @as(f64, @floatFromInt(px)) + (@as(f64, @floatFromInt(sx)) + 0.5) / ssf;
                            const fy = @as(f64, @floatFromInt(py)) + (@as(f64, @floatFromInt(sy)) + 0.5) / ssf;
                            const k0 = c.rayAt(fx, fy);
                            acc += traceRay(cfg, sc, c.x0, k0, op).intensity;
                        }
                    }
                    o[py * c.width + px] = acc * inv;
                }
            }
        }
    };

    var threads: [63]std.Thread = undefined;
    const want: usize = @min(@max(nthreads, 1) - 1, threads.len);
    var spawned: usize = 0;
    for (0..want) |i| {
        threads[i] = std.Thread.spawn(.{}, Worker.run, .{ s, cam, out, opts, &next_row }) catch break;
        spawned = i + 1;
    }
    Worker.run(s, cam, out, opts, &next_row);
    for (threads[0..spawned]) |t| t.join();
}
