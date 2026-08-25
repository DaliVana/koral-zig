//! Pure per-cell kernels for radiative shear viscosity (PUFFY's
//! RADVISCOSITY == SHEARVISCOSITY, KORAL rad.c):
//!
//!   shearFromGradients  rad.c:4090: σ_μν and the expansion θ from
//!                                     assembled velocity gradients.
//!   viscCoeff           rad.c:4508: ν = ALPHARADVISC·mfp with the
//!                                     RADVISCMFPSPH mfp limiter and the
//!                                     RADVISCNUDAMP diffusion cap.
//!   rijVisc             rad.c:4670: R^ij_visc = −2 ν Ê σ^ij → R^i_j.
//!   addViscFlux         rad.c:3799: the RADVISCMAXVELDAMP velocity cap
//!                                     and the face addition to the M1 flux.
//!
//! Physics: M1 cannot represent crossing photon streams; near the polar
//! axis radiation converging from both funnel walls collapses into one
//! unphysical radial beam. The patch adds a Navier-Stokes-like viscous
//! stress to the radiation tensor, R^ij_visc = −2 ν Ê σ^ij: the next-order
//! diffusive correction the closure truncates away. σ_μν is the fluid
//! shear: covariant velocity gradients symmetrized, projected orthogonal
//! to u^μ with P_μν = g_μν + u_μu_ν, trace (the expansion θ) removed, time
//! row fixed by σ_μν u^ν = 0. The coefficient ν = α·mfp (mfp = 1/χ)
//! carries three physical limiters: the mfp is capped by the local radius
//! (a photon cannot diffuse farther than the system size) and killed
//! near/inside the horizon; ν ≤ Δx²/(4Δt) keeps the explicit diffusion
//! stable so viscosity never sets the timestep, and the face flux is
//! rescaled so the implied transport velocity stays ≤ MAXRADVISCVEL;
//! viscous transport must never outrun the physical signal.
//!
//! Everything here is values-in/values-out; no grid, no stencil, no
//! threading, like the rest of physics/. The sim-coupled half (the FD gather
//! of the velocity gradients, the χ/cell-size/BL-radius inputs of ν, and the
//! once-per-step threaded domain pass) lives in sim/rijvisc.zig.
//!
//! PUFFY switches (define.h:96-102): RADVISCMFPSPH (no RMIN/MAX overrides →
//! rmin = 1.2·r_horizon), RADVISCNUDAMP, RADVISCMAXVELDAMP, ALPHARADVISC 0.1,
//! MAXRADVISCVEL 0.1.

const std = @import("std");
const config = @import("../config.zig");
const layout = @import("../layout.zig");
const relele = @import("../relele.zig");
const misc = @import("../math/misc.zig");
const Geometry = @import("../geometry.zig").Geometry;

const small: f64 = 1.0e-80; // C: SMALL

/// PUFFY viscosity parameters (define.h:96-102). Held on Sim.Options so the
/// coefficient limiter has ALPHARADVISC / MAXRADVISCVEL without a global.
pub const Params = struct {
    /// C: ALPHARADVISC; ν = alpha·mfp.
    alpha: f64 = 0.1,
    /// C: MAXRADVISCVEL. The characteristic-velocity damping threshold.
    maxvel: f64 = 0.1,
};

/// Result of the shear assembly: the lab-frame shear tensor σ_ij (all 16
/// components) and the expansion θ = u^μ_{;μ}.
pub const ShearOut = struct {
    s: [4][4]f64,
    div: f64,
};

/// Pure lab-frame shear algebra (rad.c:4090-4180 tail). Given the covariant
/// velocity gradient du_i,j = ∂_j u_i and its contravariant counterpart du2
/// (only the diagonal is read, for the expansion), the frame 4-velocity
/// (ucon/ucov), the metric `gg`, and the Christoffel block `kr`
/// (kr[k·16 + i·4 + j] = Γ^k_ij), build σ_μν and the expansion θ. No grid or
/// sim access; feed it an analytic gradient and check the kinematic
/// invariants (σ symmetric, σ_μν u^ν = 0) directly. The FD stencil that
/// assembles the gradients from neighbour cells is sim/rijvisc.zig's
/// calcShearLab.
pub fn shearFromGradients(
    du: *const [4][4]f64,
    du2: *const [4][4]f64,
    ucon: [4]f64,
    ucov: [4]f64,
    gg: *const [4][5]f64,
    kr: *const [64]f64,
) ShearOut {
    // covariant derivatives du_i;j and (diagonal) du^i;i
    var dcu: [4][4]f64 = @splat(@splat(0));
    var dcudiag: [4]f64 = @splat(0);
    for (0..4) |i| {
        for (0..4) |j| {
            var krsum: f64 = 0;
            for (0..4) |k| krsum += kr[k * 16 + i * 4 + j] * ucov[k];
            dcu[i][j] = du[i][j] - krsum;
        }
        var krsum: f64 = 0;
        for (0..4) |k| krsum += kr[i * 16 + i * 4 + k] * ucon[k];
        dcudiag[i] = du2[i][i] + krsum;
    }

    // expansion θ = u^μ_{;μ}
    var theta: f64 = 0;
    for (0..4) |i| theta += dcudiag[i];

    // projection tensors P_ij and P^i_j
    var p11: [4][4]f64 = undefined;
    var p21: [4][4]f64 = undefined;
    for (0..4) |i| {
        for (0..4) |j| {
            p11[i][j] = gg[i][j] + ucov[i] * ucov[j];
            p21[i][j] = relele.kron(i, j) + ucon[i] * ucov[j];
        }
    }

    // spatial shear σ_ij (i,j = 1..3)
    var s: [4][4]f64 = @splat(@splat(0));
    var i: usize = 1;
    while (i < 4) : (i += 1) {
        var j: usize = 1;
        while (j < 4) : (j += 1) {
            var sum1: f64 = 0;
            var sum2: f64 = 0;
            for (0..4) |k| {
                sum1 += dcu[i][k] * p21[k][j];
                sum2 += dcu[j][k] * p21[k][i];
            }
            s[i][j] = 0.5 * (sum1 + sum2) - 1.0 / 3.0 * theta * p11[i][j];
        }
    }

    // time components from u^μ σ_μν = 0
    i = 1;
    while (i < 4) : (i += 1) {
        const v = -1.0 / ucon[0] * (ucon[1] * s[1][i] + ucon[2] * s[2][i] + ucon[3] * s[3][i]);
        s[i][0] = v;
        s[0][i] = v;
    }
    s[0][0] = -1.0 / ucon[0] * (ucon[1] * s[1][0] + ucon[2] * s[2][0] + ucon[3] * s[3][0]);

    return .{ .s = s, .div = theta };
}

/// Inputs of viscCoeff, named so the same-typed scalars (in particular the
/// two radii) cannot be transposed at a call site.
pub const ViscCoeffIn = struct {
    /// total opacity κ + κ_es (the mfp is 1/χ)
    chi: f64,
    /// smallest active proper cell size
    mindx: f64,
    /// BL radius of the cell
    r_bl: f64,
    /// horizon radius
    rhor: f64,
    /// C: ALPHARADVISC
    alpha: f64,
    /// the step's dt (C: global_dt, set before the RK stages)
    global_dt: f64,
};

/// C: calc_rad_visccoeff (rad.c:4508) with RADVISCMFPSPH + RADVISCNUDAMP;
/// the pure limiter chain.
pub inline fn viscCoeff(in: ViscCoeffIn) f64 {
    var mfp = 1.0 / in.chi;

    // RADVISCMFPSPH: mfp capped by the spherical (BL) radius, killed inside
    // rmin = 1.2·r_horizon (no RADVISCMFPSPHRMIN / MAX for PUFFY).
    const rmin = 1.2 * in.rhor;
    const mfplim = in.r_bl;
    if (mfp > mfplim or in.chi < small) mfp = mfplim;
    if (mfp < 0 or !std.math.isFinite(mfp)) mfp = 0;
    mfp *= misc.stepFunction(in.r_bl - rmin, 0.2 * rmin);
    if (in.r_bl <= 1.0 * in.rhor) mfp = 0;

    var nu = in.alpha * mfp;

    // RADVISCNUDAMP: cap ν at the maximal stable diffusion coefficient.
    const nulimit = in.mindx * in.mindx / 2.0 / in.global_dt / 2.0;
    if (nu > nulimit) nu = nulimit;

    return nu;
}

/// C: calc_Rij_visc (rad.c:4670); R^ij_visc = −2 ν Ê σ^ij (σ^ij both
/// indices raised), returned lowered to R^i_j (indices_2221).
pub inline fn rijVisc(nu: f64, erad: f64, shear: *const [4][4]f64, geom: *const Geometry) [4][4]f64 {
    var rvisc: [4][4]f64 = undefined;
    for (0..4) |i| {
        for (0..4) |j| {
            rvisc[i][j] = -2.0 * nu * erad * shear[i][j];
        }
    }
    return relele.lowerSecond(rvisc, geom); // R^ij → R^i_j
}

/// C: the RADVISCMAXVELDAMP block of f_flux_prime_rad_total (rad.c:3799-3845).
/// Adds the face-averaged, velocity-damped viscous flux to the M1 rad rows of
/// `ff`. `uu` are the conserveds of the face state, `rijvisc` the
/// face-averaged R^i_j (undamped), `dim` == ifacedim, `maxvel` C's
/// MAXRADVISCVEL. `y_active`/`z_active` gate the ii == 2/3 components of the
/// velocity probe; C never skips x, so there is deliberately no x flag.
pub fn addViscFlux(
    comptime cfg: config.Config,
    ff: *[layout.VarLayout(cfg).count]f64,
    uu: *const [layout.VarLayout(cfg).count]f64,
    rijvisc: *const [4][4]f64,
    geom: *const Geometry,
    dim: usize,
    y_active: bool,
    z_active: bool,
    maxvel: f64,
) void {
    const L = layout.VarLayout(cfg);
    const ee0 = comptime L.index(.ee);
    const gdetu = geom.gdet;
    const gg = &geom.gg;

    // characteristic viscous velocity per dimension (rad.c:3808-3819). Note
    // vel[id] is *overwritten* per ii, ending on the last non-skipped ii — a
    // C quirk transcribed verbatim.
    var vel = [3]f64{ 0, 0, 0 };
    for (0..3) |id| {
        var ii: usize = 1;
        while (ii < 4) : (ii += 1) {
            if (ii == 2 and !y_active) continue;
            if (ii == 3 and !z_active) continue;
            if (@abs(uu[ee0 + ii]) < 1.0e-10 * @abs(uu[ee0])) continue;
            vel[id] = rijvisc[id + 1][ii] / (uu[ee0 + ii] / gdetu) * @sqrt(gg[id + 1][id + 1]);
        }
    }
    const maxv = @abs(vel[dim]); // face flux → the cap tests |vel[ifacedim]|

    var dampfac: f64 = 1.0;
    if (maxv > maxvel) dampfac = maxvel / maxv;

    for (0..4) |nu| {
        ff[ee0 + nu] += gdetu * dampfac * rijvisc[dim + 1][nu];
    }
}

/// Minkowski geometry for the values-in kernel tests: g = diag(−1,1,1,1),
/// √−g = 1 (gg[3][4]), no derivative extras.
fn minkGeometry() Geometry {
    var g = Geometry{
        .coords = .mink,
        .xxvec = .{ 0, 0, 0, 0 },
        .gg = @splat(@splat(0)),
        .GG = @splat(@splat(0)),
        .gdet = 1.0,
        .alpha = 1.0,
        .gttpert = 0.0,
    };
    g.gg[0][0] = -1.0;
    g.GG[0][0] = -1.0;
    for (1..4) |i| {
        g.gg[i][i] = 1.0;
        g.GG[i][i] = 1.0;
    }
    g.gg[3][4] = 1.0; // √−g
    return g;
}

test "radvisc: viscCoeff limiter chain — free, mfp-capped, horizon kill, ν cap" {
    const expectEqual = std.testing.expectEqual;
    const rhor: f64 = 2.0;
    const rmin = 1.2 * rhor;

    // free regime: ν = α·(1/χ)·step, no cap bites (expected mirrors the
    // kernel's expression shape, so equality is exact)
    const step_far = misc.stepFunction(100.0 - rmin, 0.2 * rmin);
    const free = viscCoeff(.{ .chi = 10.0, .mindx = 1.0, .r_bl = 100.0, .rhor = rhor, .alpha = 0.1, .global_dt = 1e-30 });
    try expectEqual(0.1 * ((1.0 / 10.0) * step_far), free);

    // χ < SMALL ⇒ mfp = r_bl (the RADVISCMFPSPH limit)
    const capped = viscCoeff(.{ .chi = 1e-100, .mindx = 1.0, .r_bl = 100.0, .rhor = rhor, .alpha = 0.1, .global_dt = 1e-30 });
    try expectEqual(0.1 * (100.0 * step_far), capped);

    // inside the horizon the coefficient is exactly zero
    const killed = viscCoeff(.{ .chi = 10.0, .mindx = 1.0, .r_bl = rhor, .rhor = rhor, .alpha = 0.1, .global_dt = 1e-30 });
    try expectEqual(@as(f64, 0.0), killed);

    // RADVISCNUDAMP: huge dt ⇒ ν clamps to mindx²/(4·dt)
    const damped = viscCoeff(.{ .chi = 10.0, .mindx = 0.5, .r_bl = 100.0, .rhor = rhor, .alpha = 0.1, .global_dt = 1e10 });
    try expectEqual(0.5 * 0.5 / 2.0 / 1e10 / 2.0, damped);
}

test "radvisc: rijVisc = −2νÊσ^ij lowered on the second index" {
    const geom = minkGeometry();
    var shear: [4][4]f64 = @splat(@splat(0));
    shear[0][0] = 1.0;
    shear[1][0] = 0.25;
    shear[1][2] = 0.5;
    shear[2][1] = 0.5;

    // ν = 2, Ê = 3 ⇒ R^ij = −12 σ^ij; Minkowski lowering negates column 0.
    const r = rijVisc(2.0, 3.0, &shear, &geom);
    try std.testing.expectEqual(@as(f64, 12.0), r[0][0]);
    try std.testing.expectEqual(@as(f64, 3.0), r[1][0]);
    try std.testing.expectEqual(@as(f64, -6.0), r[1][2]);
    try std.testing.expectEqual(@as(f64, -6.0), r[2][1]);
    try std.testing.expectEqual(@as(f64, 0.0), r[3][3]);
}

test "radvisc: addViscFlux — undamped add, cap saturation, z-flag gating" {
    const cfg = config.puffy;
    const L = layout.VarLayout(cfg);
    const ee0 = comptime L.index(.ee);
    const geom = minkGeometry();

    var uu: [L.count]f64 = @splat(0);
    uu[ee0] = 1.0;
    uu[ee0 + 1] = 0.1;

    // below the cap: vel = 1e-3/0.1 = 1e-2 ≤ maxvel ⇒ dampfac = 1, exact add
    var rv: [4][4]f64 = @splat(@splat(0));
    rv[1][0] = 2.0;
    rv[1][1] = 1.0e-3;
    var ff: [L.count]f64 = @splat(0);
    addViscFlux(cfg, &ff, &uu, &rv, &geom, 0, true, false, 0.1);
    try std.testing.expectEqual(@as(f64, 2.0), ff[ee0]);
    try std.testing.expectEqual(@as(f64, 1.0e-3), ff[ee0 + 1]);

    // above the cap the damped flux saturates: 10× the tensor, same flux
    var rva: [4][4]f64 = @splat(@splat(0));
    rva[1][1] = 1.0; // vel = 10 > maxvel
    var ffa: [L.count]f64 = @splat(0);
    addViscFlux(cfg, &ffa, &uu, &rva, &geom, 0, true, false, 0.1);
    var rvb: [4][4]f64 = @splat(@splat(0));
    rvb[1][1] = 10.0;
    var ffb: [L.count]f64 = @splat(0);
    addViscFlux(cfg, &ffb, &uu, &rvb, &geom, 0, true, false, 0.1);
    try std.testing.expect(ffa[ee0 + 1] > 0);
    try std.testing.expectApproxEqRel(ffa[ee0 + 1], ffb[ee0 + 1], 1e-12);

    // z_active = false gates ii == 3 out of the velocity probe even when the
    // z conserved is huge — but the flux add itself still covers all columns
    var uuz = uu;
    uuz[ee0 + 3] = 1.0e30;
    var rvz: [4][4]f64 = @splat(@splat(0));
    rvz[1][1] = 1.0e-3;
    rvz[1][3] = 5.0; // would dominate vel if ii == 3 were probed
    var ffz: [L.count]f64 = @splat(0);
    addViscFlux(cfg, &ffz, &uuz, &rvz, &geom, 0, true, false, 0.1);
    try std.testing.expectEqual(@as(f64, 1.0e-3), ffz[ee0 + 1]); // undamped
    try std.testing.expectEqual(@as(f64, 5.0), ffz[ee0 + 3]);
}
