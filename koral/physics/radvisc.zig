//! Radiative shear viscosity (PUFFY's RADVISCOSITY == SHEARVISCOSITY),
//! transcribed from KORAL rad.c:
//!
//!   calc_shear_lab           rad.c:3952  — lab-frame shear tensor σ_ij from
//!                                          the radiation-rest-frame velocity
//!                                          field (FX/FY/FZ, VELPRIMRAD) with
//!                                          Christoffel corrections.
//!   calc_rad_visccoeff       rad.c:4508  — ν = ALPHARADVISC·mfp, the
//!                                          RADVISCMFPSPH mfp limiter and the
//!                                          RADVISCNUDAMP diffusion cap.
//!   calc_rad_shearviscosity  rad.c:3912  — σ^ij (indices_1122) + ν.
//!   calc_Rij_visc            rad.c:4670  — R^ij_visc = −2 ν Ê σ^ij → R^i_j.
//!   calc_Rij_visc_total      rad.c:4628  — fill the per-cell store over the
//!                                          domain + one non-corner ghost ring.
//!   f_flux_prime_rad_total   rad.c:3724  — the face average of R^i_j_visc,
//!                                          the RADVISCMAXVELDAMP velocity cap,
//!                                          and the addition to the M1 flux.
//!
//! PUFFY switches (define.h:96-102): RADVISCMFPSPH (no RMIN/MAX overrides →
//! rmin = 1.2·r_horizon), RADVISCNUDAMP, RADVISCMAXVELDAMP, ALPHARADVISC 0.1,
//! MAXRADVISCVEL 0.1. ACCELRADVISCOSITY is OFF (recompute every step, no
//! radvisclasttime caching). Derivatives are always centered (derdir≡0):
//! calc_Rij_visc_total and the face path both pass {0,0,0}.

const std = @import("std");
const config = @import("../config.zig");
const layout = @import("../layout.zig");
const relele = @import("../relele.zig");
const radiation = @import("radiation.zig");
const radforce = @import("radforce.zig");
const p2u_mod = @import("../p2u.zig");
const coco = @import("../metric/coco.zig");
const metric = @import("../metric/metric.zig");
const misc = @import("../math/misc.zig");
const threading = @import("../sim/threading.zig");
const Geometry = @import("../geometry.zig").Geometry;

const small: f64 = 1.0e-80; // C: SMALL

/// PUFFY viscosity parameters (define.h:96-102). Held on Sim.Options so the
/// coefficient limiter has ALPHARADVISC / MAXRADVISCVEL without a global.
pub const Params = struct {
    /// C: ALPHARADVISC — ν = alpha·mfp.
    alpha: f64 = 0.1,
    /// C: MAXRADVISCVEL — the characteristic-velocity damping threshold.
    maxvel: f64 = 0.1,
};

/// Result of calc_shear_lab: the lab-frame shear tensor σ_ij (all 16
/// components) and the expansion θ = u^μ_{;μ}.
pub const ShearOut = struct {
    s: [4][4]f64,
    div: f64,
};

/// C: calc_shear_lab (rad.c:3952), the RAD branch (VELPRIMRAD, FX..FZ). The
/// gas branch (VELPRIM, VX..VZ) is available via `istart`/`whichvel` but
/// PUFFY only uses RAD. Neighbour prims come from sim.p, neighbour metrics
/// and coordinates from the metric cache; Christoffels from the center cell.
pub fn calcShearLab(
    comptime SimT: type,
    sim: *const SimT,
    ix: i64,
    iy: i64,
    iz: i64,
    pp0: *const [SimT.nv]f64,
    comptime istart: usize,
    comptime whichvel: relele.VelType,
) relele.Error!ShearOut {
    const NV = SimT.nv;

    const geom = sim.cache.fillGeometry(ix, iy, iz);
    const gg = &geom.gg;

    // du_i,j (covariant velocity) and du^i,j (contravariant, only used on the
    // diagonal for the expansion). Time column forced to zero (d/dt = 0).
    var du: [4][4]f64 = @splat(@splat(0));
    var du2: [4][4]f64 = @splat(@splat(0));

    // center four-velocity
    const uc = try relele.convVelsBoth(
        .{ 0, pp0[istart], pp0[istart + 1], pp0[istart + 2] },
        whichvel,
        gg,
        &geom.GG,
    );
    const ucon = uc.con;
    const ucov = uc.cov;

    const nd = [3]i64{ sim.nDim(0), sim.nDim(1), sim.nDim(2) };
    const nx = nd[0];
    const ny = nd[1];
    const nz = nd[2];

    var idim: usize = 1;
    while (idim < 4) : (idim += 1) {
        const dim = idim - 1;

        // A collapsed dimension has bit-identical prims and metric on both
        // sides (axisymmetric metric, no z-dependence) → derivative ≡ 0.
        // C computes exactly 0 there; we skip the (nonexistent) z-ghosts.
        if (nd[dim] <= 1) continue;

        var cm = [3]i64{ ix, iy, iz };
        var cp = [3]i64{ ix, iy, iz };
        cm[dim] -= 1;
        cp[dim] += 1;

        var ppm1: [NV]f64 = undefined;
        var ppp1: [NV]f64 = undefined;
        sim.p.load(cm[0], cm[1], cm[2], &ppm1);
        sim.p.load(cp[0], cp[1], cp[2], &ppp1);

        const gm = sim.cache.fillGeometry(cm[0], cm[1], cm[2]);
        const gp = sim.cache.fillGeometry(cp[0], cp[1], cp[2]);

        const um = try relele.convVelsBoth(
            .{ 0, ppm1[istart], ppm1[istart + 1], ppm1[istart + 2] },
            whichvel,
            &gm.gg,
            &gm.GG,
        );
        const up = try relele.convVelsBoth(
            .{ 0, ppp1[istart], ppp1[istart + 1], ppp1[istart + 2] },
            whichvel,
            &gp.gg,
            &gp.GG,
        );

        const xm = gm.xxvec[idim];
        const xc = geom.xxvec[idim];
        const xp = gp.xxvec[idim];

        for (0..4) |i| {
            const dc = (up.cov[i] - um.cov[i]) / (xp - xm);
            const dr = (up.cov[i] - ucov[i]) / (xp - xc);
            const dl = (ucov[i] - um.cov[i]) / (xc - xm);
            const dc2 = (up.con[i] - um.con[i]) / (xp - xm);
            const dr2 = (up.con[i] - ucon[i]) / (xp - xc);
            const dl2 = (ucon[i] - um.con[i]) / (xc - xm);

            // corner avoidance (rad.c:4110-4137): a cell adjacent to a grid
            // corner uses a one-sided derivative in the other dimensions so
            // it never reads the (unfilled) diagonal corner. iz≡0≡NZ-1 in 2D.
            if ((ix < 0 and iy == 0 and iz == 0 and idim != 1) or
                (iy < 0 and ix == 0 and iz == 0 and idim != 2) or
                (iz < 0 and ix == 0 and iy == 0 and idim != 3))
            {
                du[i][idim] = dr;
                du2[i][idim] = dr2;
            } else if ((ix < 0 and iy == ny - 1 and iz == nz - 1 and idim != 1) or
                (iy < 0 and ix == nx - 1 and iz == nz - 1 and idim != 2) or
                (iz < 0 and ix == nx - 1 and iy == ny - 1 and idim != 3))
            {
                du[i][idim] = dl;
                du2[i][idim] = dl2;
            } else if ((ix >= nx and iy == 0 and iz == 0 and idim != 1) or
                (iy >= ny and ix == 0 and iz == 0 and idim != 2) or
                (iz >= nz and ix == 0 and iy == 0 and idim != 3))
            {
                du[i][idim] = dr;
                du2[i][idim] = dr2;
            } else if ((ix >= nx and iy == ny - 1 and iz == nz - 1 and idim != 1) or
                (iy >= ny and ix == nx - 1 and iz == nz - 1 and idim != 2) or
                (iz >= nz and ix == nx - 1 and iy == ny - 1 and idim != 3))
            {
                du[i][idim] = dl;
                du2[i][idim] = dl2;
            } else {
                // derdir ≡ 0 (centered) everywhere PUFFY uses this
                du[i][idim] = dc;
                du2[i][idim] = dc2;
            }
        }
    }

    // covariant derivatives du_i;j and (diagonal) du^i;i
    var dcu: [4][4]f64 = @splat(@splat(0));
    var dcudiag: [4]f64 = @splat(0);
    for (0..4) |i| {
        for (0..4) |j| {
            var krsum: f64 = 0;
            for (0..4) |k| krsum += sim.cache.kr(k, i, j, ix, iy, iz) * ucov[k];
            dcu[i][j] = du[i][j] - krsum;
        }
        var krsum: f64 = 0;
        for (0..4) |k| krsum += sim.cache.kr(i, i, k, ix, iy, iz) * ucon[k];
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

/// C: calc_rad_visccoeff (rad.c:4508) with RADVISCMFPSPH + RADVISCNUDAMP.
/// `global_dt` is the step's dt (C: global_dt, set before the RK stages).
pub fn calcRadViscCoeff(
    comptime SimT: type,
    sim: *const SimT,
    ix: i64,
    iy: i64,
    iz: i64,
    pp: *const [SimT.nv]f64,
    geom: *const Geometry,
    global_dt: f64,
) relele.Error!f64 {
    const cfg = SimT.Cfg;
    const opac = &(sim.opt.opac orelse return 0);

    const chi = try radforce.calcChiSlim(cfg, pp.*, geom, sim.opt.gam, opac);
    var mfp = 1.0 / chi;

    const g = &sim.grid;
    const gg = &geom.gg;
    const dx = [3]f64{
        g.cellSize(ix, 0) * @sqrt(gg[1][1]),
        g.cellSize(iy, 1) * @sqrt(gg[2][2]),
        g.cellSize(iz, 2) * @sqrt(gg[3][3]),
    };
    const ny = sim.nDim(1);
    const nz = sim.nDim(2);
    const mindx = if (ny == 1 and nz == 1)
        dx[0]
    else if (nz == 1)
        @min(dx[0], dx[1])
    else if (ny == 1)
        @min(dx[0], dx[2])
    else
        @min(dx[0], @min(dx[1], dx[2]));

    // RADVISCMFPSPH: mfp capped by the spherical (BL) radius, killed inside
    // rmin = 1.2·r_horizon (no RADVISCMFPSPHRMIN / RADVISCMFPSPHMAX for PUFFY).
    const rhor = metric.rHorizonBL(sim.opt.mp.a);
    const rmin = 1.2 * rhor;
    const xxbl = coco.cocoN(geom.xxvec, cfg.coords, .bl, sim.opt.mp);
    const mfplim = xxbl[1];
    if (mfp > mfplim or chi < small) mfp = mfplim;
    if (mfp < 0 or !std.math.isFinite(mfp)) mfp = 0;
    mfp *= misc.stepFunction(xxbl[1] - rmin, 0.2 * rmin);
    if (xxbl[1] <= 1.0 * rhor) mfp = 0;

    var nu = sim.opt.radvisc.alpha * mfp;

    // RADVISCNUDAMP: cap ν at the maximal stable diffusion coefficient.
    const nulimit = mindx * mindx / 2.0 / global_dt / 2.0;
    if (nu > nulimit) nu = nulimit;

    return nu;
}

/// C: calc_rad_shearviscosity (rad.c:3912) — σ^ij (both indices raised) and ν.
pub fn calcRadShearViscosity(
    comptime SimT: type,
    sim: *const SimT,
    ix: i64,
    iy: i64,
    iz: i64,
    pp: *const [SimT.nv]f64,
    geom: *const Geometry,
    global_dt: f64,
) relele.Error!struct { shear: [4][4]f64, nu: f64 } {
    const L = SimT.Layout;
    const sh = try calcShearLab(SimT, sim, ix, iy, iz, pp, comptime L.index(.fx), .velr);
    const shear = relele.indices1122(sh.s, &geom.GG); // σ_ij → σ^ij
    const nu = try calcRadViscCoeff(SimT, sim, ix, iy, iz, pp, geom, global_dt);
    return .{ .shear = shear, .nu = nu };
}

/// C: calc_Rij_visc (rad.c:4670) — R^ij_visc = −2 ν Ê_rest σ^ij, returned as
/// R^i_j (indices_2221). ACCELRADVISCOSITY off ⇒ always recomputed.
pub fn calcRijVisc(
    comptime SimT: type,
    sim: *const SimT,
    ix: i64,
    iy: i64,
    iz: i64,
    pp: *const [SimT.nv]f64,
    geom: *const Geometry,
    global_dt: f64,
) relele.Error![4][4]f64 {
    const L = SimT.Layout;
    const rv = try calcRadShearViscosity(SimT, sim, ix, iy, iz, pp, geom, global_dt);
    const erad = pp[L.index(.ee)];
    var rvisc: [4][4]f64 = undefined;
    for (0..4) |i| {
        for (0..4) |j| {
            rvisc[i][j] = -2.0 * rv.nu * erad * rv.shear[i][j];
        }
    }
    return relele.indices2221(rvisc, &geom.gg); // R^ij → R^i_j
}

/// C: calc_Rij_visc_total (rad.c:4628) — populate sim.rijvisc (R^i_j) over the
/// domain plus a one-cell ghost ring, skipping corners (if_outsidegc). Called
/// once per step (problem.c:127) with the step's global_dt. P1: band-parallel
/// over iy rows (each cell writes only its own rijvisc block; the ±1-stencil
/// reads of p are frozen during the pass).
pub fn calcRijViscTotal(comptime SimT: type, sim: *SimT, global_dt: f64) (relele.Error || error{OutOfMemory})!void {
    threading.parallelZero(sim.team, sim.rijvisc.data);

    const ny = sim.nyi();
    const ylim: i64 = if (ny > 1) 1 else 0;

    const Ctx = struct { sim: *SimT, dt: f64 };
    var ctx = Ctx{ .sim = sim, .dt = global_dt };
    const W = struct {
        fn w(c: *Ctx, iy0: i64, iy1: i64, res: *threading.ChunkResult) void {
            rijViscRows(SimT, c.sim, c.dt, iy0, iy1) catch |e| {
                res.err = e;
            };
        }
    };
    const res = threading.parallelRange(Ctx, &ctx, sim.team, -ylim, ny + ylim, W.w);
    if (res.err) |e| return e;
}

/// The per-cell R^i_j body for iy ∈ [iy0, iy1) (all iz, ix incl. the ring).
fn rijViscRows(comptime SimT: type, sim: *SimT, global_dt: f64, iy0: i64, iy1: i64) relele.Error!void {
    const nx = sim.nxi();
    const ny = sim.nyi();
    const nz = sim.nzi();
    const lim: i64 = 1;

    const xlim: i64 = if (nx > 1) lim else 0;
    const zlim: i64 = if (nz > 1) lim else 0;

    var iz: i64 = -zlim;
    while (iz < nz + zlim) : (iz += 1) {
        var iy: i64 = iy0;
        while (iy < iy1) : (iy += 1) {
            var ix: i64 = -xlim;
            while (ix < nx + xlim) : (ix += 1) {
                if (ifOutsideGc(nx, ny, nz, ix, iy, iz)) continue; // avoid corners
                var pp: [SimT.nv]f64 = undefined;
                sim.p.load(ix, iy, iz, &pp);
                const geom = sim.cache.fillGeometry(ix, iy, iz);
                const rvisc = try calcRijVisc(SimT, sim, ix, iy, iz, &pp, &geom, global_dt);
                var t: [16]f64 = undefined;
                for (0..4) |i| {
                    for (0..4) |j| t[i * 4 + j] = rvisc[i][j];
                }
                sim.rijvisc.store(ix, iy, iz, &t);
            }
        }
    }
}

/// C: if_outsidegc (finite.c) — true (skip) for corner ghost cells, where two
/// or more of (ix,iy,iz) are simultaneously outside the domain [0,n).
fn ifOutsideGc(nx: i64, ny: i64, nz: i64, ix: i64, iy: i64, iz: i64) bool {
    var outside: u2 = 0;
    if (ix < 0 or ix >= nx) outside += 1;
    if (iy < 0 or iy >= ny) outside += 1;
    if (iz < 0 or iz >= nz) outside += 1;
    return outside >= 2;
}

/// C: the RADVISCMAXVELDAMP block of f_flux_prime_rad_total (rad.c:3799-3845).
/// Adds the face-averaged, velocity-damped viscous flux to the M1 rad rows of
/// `ff`. `rijvisc` is the face-averaged R^i_j (undamped). `dim` == ifacedim.
pub fn addRadViscFlux(
    comptime SimT: type,
    sim: *const SimT,
    ff: *[SimT.nv]f64,
    pp: *const [SimT.nv]f64,
    geom: *const Geometry,
    dim: usize,
    rijvisc: *const [4][4]f64,
) relele.Error!void {
    const cfg = SimT.Cfg;
    const L = SimT.Layout;
    const ee0 = comptime L.index(.ee);
    const gdetu = geom.gdet;
    const gg = &geom.gg;

    const uu = try p2u_mod.p2u(cfg, pp.*, geom, sim.opt.gam);

    const ny = sim.nDim(1);
    const nz = sim.nDim(2);

    // characteristic viscous velocity per dimension (rad.c:3808-3819). Note
    // vel[idim] is *overwritten* per i, ending on the last non-skipped i — a
    // C quirk transcribed verbatim.
    var vel = [3]f64{ 0, 0, 0 };
    for (0..3) |id| {
        var ii: usize = 1;
        while (ii < 4) : (ii += 1) {
            if (ii == 2 and ny == 1) continue;
            if (ii == 3 and nz == 1) continue;
            if (@abs(uu[ee0 + ii]) < 1.0e-10 * @abs(uu[ee0])) continue;
            vel[id] = rijvisc[id + 1][ii] / (uu[ee0 + ii] / gdetu) * @sqrt(gg[id + 1][id + 1]);
        }
    }
    const maxvel = @abs(vel[dim]); // face flux → maxvel = |vel[ifacedim]|

    var dampfac: f64 = 1.0;
    if (maxvel > sim.opt.radvisc.maxvel) dampfac = sim.opt.radvisc.maxvel / maxvel;

    for (0..4) |nu| {
        ff[ee0 + nu] += gdetu * dampfac * rijvisc[dim + 1][nu];
    }
}
