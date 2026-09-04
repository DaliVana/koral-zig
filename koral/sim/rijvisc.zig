//! Sim-coupled radiative shear viscosity, generic over CoreT (the house
//! `fn f(comptime CoreT: type, core: *CoreT)` pattern, like sim/bc.zig):
//!
//!   calcShearLab: FD gather of the velocity gradients over the ±1
//!                           stencil (prims from core.p, metrics from the
//!                           cache), fed to the pure shear algebra
//!                           (C: calc_shear_lab, rad.c:3952).
//!   calcRadViscCoeff: gathers χ (calcChiSlim), the smallest proper
//!                           cell size and the cached BL radius for the pure
//!                           ν limiter (C: calc_rad_visccoeff, rad.c:4508).
//!   calcRadShearViscosity: σ^ij (indices_1122) + ν (C: rad.c:3912).
//!   calcRijVisc: per-cell R^i_j via the pure kernel
//!                           (C: calc_Rij_visc, rad.c:4670).
//!   calcRijViscTotal: the once-per-step threaded domain pass filling
//!                           core.visc.?.rijvisc (C: calc_Rij_visc_total, rad.c:4628).
//!   faceAvg: face-averaged R^i_j read from core.visc.?.rijvisc
//!                           (C: f_flux_prime_rad_total's ifacedim>−1 branch,
//!                           rad.c:3782).
//!   addRadViscFlux: p2u of the face state + the pure velocity-damped
//!                           flux addition (C: the RADVISCMAXVELDAMP block of
//!                           f_flux_prime_rad_total, rad.c:3799).
//!
//! The per-cell tensor algebra, limiter chains, and PUFFY switch notes live
//! in physics/radvisc.zig. ACCELRADVISCOSITY is OFF (recompute
//! every step, no radvisclasttime caching); derivatives are always centered
//! (C's derdir option. See the branch comment in calcShearLab).

const relele = @import("../relele.zig");
const radvisc = @import("../physics/radvisc.zig");
const radforce = @import("../physics/radforce.zig");
const p2u_mod = @import("../p2u.zig");
const metric = @import("../metric/metric.zig");
const threading = @import("../threading.zig");
const Geometry = @import("../geometry.zig").Geometry;
const std = @import("std");
const field_mod = @import("../field.zig");
const Grid = @import("../grid.zig").Grid;

/// The viscous-stress scratch, owned by a Sim only when `opt.radviscosity`
/// is on (redesign step 3: pass-owned scratch). Lives at `core.visc`.
pub const State = struct {
    /// C: RADVISCOSITY parameters (α, the velocity damping cap).
    params: radvisc.Params,
    /// C: Rijviscglobal — the per-cell viscous R^i_j (flattened i*4+j),
    /// filled once per step over the domain + one non-corner ghost ring.
    rijvisc: field_mod.Field(16),

    pub fn init(allocator: std.mem.Allocator, g: Grid, team: ?*threading.Team, params: radvisc.Params) !State {
        const rijvisc = try field_mod.Field(16).initUninitialized(allocator, g);
        threading.parallelZero(team, rijvisc.data);
        return .{ .params = params, .rijvisc = rijvisc };
    }

    pub fn deinit(self: *State) void {
        self.rijvisc.deinit();
        self.* = undefined;
    }
};

/// threading.Error's shape (the union parallelRangeErr workers return); the
/// same local alias sim/bc.zig uses.
const Error = relele.Error || error{OutOfMemory};

/// C: calc_shear_lab (rad.c:3952), the RAD branch (VELPRIMRAD, FX..FZ). The
/// gas branch (VELPRIM, VX..VZ) is available via `istart`/`whichvel` but
/// PUFFY only uses RAD. Center and neighbour prims both come from core.p,
/// neighbour metrics and coordinates from the metric cache; Christoffels from
/// the center cell. The assembled gradients go to the pure
/// radvisc.shearFromGradients kernel.
pub fn calcShearLab(
    comptime CoreT: type,
    core: *const CoreT,
    ix: i64,
    iy: i64,
    iz: i64,
    comptime istart: usize,
    comptime whichvel: relele.VelType,
) relele.Error!radvisc.ShearOut {
    const NV = CoreT.nv;

    // Center prims from the live state — loaded here (not passed in) so they
    // cannot diverge from the ±1 neighbours the FD stencil reads from core.p.
    var pp0: [NV]f64 = undefined;
    core.p.load(ix, iy, iz, &pp0);

    const geom = core.cache.fillGeometry(ix, iy, iz);
    const gg = &geom.gg;

    // du_i,j (covariant velocity) and du^i,j (contravariant, only used on the
    // diagonal for the expansion). Time column forced to zero (d/dt = 0).
    var du: [4][4]f64 = @splat(@splat(0));
    var du2: [4][4]f64 = @splat(@splat(0));

    // center four-velocity
    const uc = try relele.convertBoth(
        .{ 0, pp0[istart], pp0[istart + 1], pp0[istart + 2] },
        whichvel,
        &geom,
    );
    const ucon = uc.con;
    const ucov = uc.cov;

    const nd = [3]i64{ core.nDim(0), core.nDim(1), core.nDim(2) };
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
        core.p.load(cm[0], cm[1], cm[2], &ppm1);
        core.p.load(cp[0], cp[1], cp[2], &ppp1);

        const gm = core.cache.fillGeometry(cm[0], cm[1], cm[2]);
        const gp = core.cache.fillGeometry(cp[0], cp[1], cp[2]);

        const um = try relele.convertBoth(
            .{ 0, ppm1[istart], ppm1[istart + 1], ppm1[istart + 2] },
            whichvel,
            &gm,
        );
        const up = try relele.convertBoth(
            .{ 0, ppp1[istart], ppp1[istart + 1], ppp1[istart + 2] },
            whichvel,
            &gp,
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

    // Hand the assembled velocity gradients to the pure shear algebra. The
    // gather above (neighbour prims + metrics via the FD stencil) is all the
    // core/grid coupling; everything below is local tensor algebra.
    const kr_blk = core.cache.krBlock(ix, iy, iz);
    return radvisc.shearFromGradients(&du, &du2, ucon, ucov, gg, kr_blk);
}

/// C: calc_rad_visccoeff (rad.c:4508). Gathers χ (calcChiSlim), the
/// smallest active proper cell size, and the cached BL radius, then applies
/// the pure limiter chain (radvisc.viscCoeff). `global_dt` is the step's dt
/// (C: global_dt, set before the RK stages).
pub fn calcRadViscCoeff(
    comptime CoreT: type,
    core: *const CoreT,
    visc: *const State,
    ix: i64,
    iy: i64,
    iz: i64,
    pp: *const [CoreT.nv]f64,
    geom: *const Geometry,
    global_dt: f64,
) relele.Error!f64 {
    const cfg = CoreT.Cfg;
    // Sim.init rejects radviscosity with a null opac (there is no meaningful
    // ν without opacities — cf. the precondition check), so on the production
    // path this capture always binds. The `else return 0` stays as a defensive
    // guard for any direct call; the optional-pointer capture avoids copying
    // the whole ~300-byte Params to a stack temporary per cell per step (the
    // idiom core.zig uses at the wavespeed τ-limiter).
    const opac = if (core.phys.opac) |*o| o else return 0;

    const chi = try radforce.calcChiSlim(cfg, pp.*, geom, core.phys.gam, opac);

    const g = &core.grid;
    const gg = &geom.gg;
    const dx = [3]f64{
        g.cellSize(ix, 0) * @sqrt(gg[1][1]),
        g.cellSize(iy, 1) * @sqrt(gg[2][2]),
        g.cellSize(iz, 2) * @sqrt(gg[3][3]),
    };
    const ny = core.nDim(1);
    const nz = core.nDim(2);
    const mindx = if (ny == 1 and nz == 1)
        dx[0]
    else if (nz == 1)
        @min(dx[0], dx[1])
    else if (ny == 1)
        @min(dx[0], dx[2])
    else
        @min(dx[0], @min(dx[1], dx[2]));

    // BL radius from the cached BL geometry — bit-identical to
    // coco.cocoN(geom.xxvec, coords, .bl, mp)[1].
    return radvisc.viscCoeff(.{
        .chi = chi,
        .mindx = mindx,
        .r_bl = core.cache.blGeom(ix, iy, iz).xxvec[1],
        .rhor = metric.rHorizonBL(core.phys.mp.a),
        .alpha = visc.params.alpha,
        .global_dt = global_dt,
    });
}

/// C: calc_rad_shearviscosity (rad.c:3912); σ^ij (both indices raised) and ν.
pub fn calcRadShearViscosity(
    comptime CoreT: type,
    core: *const CoreT,
    visc: *const State,
    ix: i64,
    iy: i64,
    iz: i64,
    pp: *const [CoreT.nv]f64,
    geom: *const Geometry,
    global_dt: f64,
) relele.Error!struct { shear: [4][4]f64, nu: f64 } {
    const L = CoreT.Layout;
    const sh = try calcShearLab(CoreT, core, ix, iy, iz, comptime L.index(.fx), .velr);
    const shear = relele.raiseBoth(sh.s, geom); // σ_ij → σ^ij
    const nu = try calcRadViscCoeff(CoreT, core, visc, ix, iy, iz, pp, geom, global_dt);
    return .{ .shear = shear, .nu = nu };
}

/// C: calc_Rij_visc (rad.c:4670); σ^ij + ν gathered here, the R^i_j tensor
/// assembled by the pure kernel. ACCELRADVISCOSITY off ⇒ always recomputed.
pub fn calcRijVisc(
    comptime CoreT: type,
    core: *const CoreT,
    visc: *const State,
    ix: i64,
    iy: i64,
    iz: i64,
    pp: *const [CoreT.nv]f64,
    geom: *const Geometry,
    global_dt: f64,
) relele.Error![4][4]f64 {
    const L = CoreT.Layout;
    const rv = try calcRadShearViscosity(CoreT, core, visc, ix, iy, iz, pp, geom, global_dt);
    return radvisc.rijVisc(rv.nu, pp[L.index(.ee)], &rv.shear, geom);
}

/// C: calc_Rij_visc_total (rad.c:4628). Populate core.visc.?.rijvisc (R^i_j) over the
/// domain plus a one-cell ghost ring, skipping corners (core.isCorner). Called
/// once per step (problem.c:127) with the step's global_dt. P1: band-parallel
/// over iy rows (each cell writes only its own rijvisc block; the ±1-stencil
/// reads of p are frozen during the pass).
pub fn calcRijViscTotal(comptime CoreT: type, core: *CoreT, visc: *State, global_dt: f64) Error!void {
    threading.parallelZero(core.team, visc.rijvisc.data);

    const ny = core.nyi();
    const ylim: i64 = if (ny > 1) 1 else 0;

    const Ctx = struct { core: *CoreT, visc: *State, dt: f64 };
    var ctx = Ctx{ .core = core, .visc = visc, .dt = global_dt };
    const W = struct {
        fn w(c: *Ctx, iy0: i64, iy1: i64) Error!void {
            try rijViscRows(CoreT, c.core, c.visc, c.dt, iy0, iy1);
        }
    };
    try threading.parallelRangeErr(Ctx, &ctx, core.team, -ylim, ny + ylim, W.w);
}

/// The per-cell R^i_j body for iy ∈ [iy0, iy1) (all iz, ix incl. the ring).
fn rijViscRows(comptime CoreT: type, core: *CoreT, visc: *State, global_dt: f64, iy0: i64, iy1: i64) relele.Error!void {
    const nx = core.nxi();
    const nz = core.nzi();
    const lim: i64 = 1;

    const xlim: i64 = if (nx > 1) lim else 0;
    const zlim: i64 = if (nz > 1) lim else 0;

    var iz: i64 = -zlim;
    while (iz < nz + zlim) : (iz += 1) {
        var iy: i64 = iy0;
        while (iy < iy1) : (iy += 1) {
            var ix: i64 = -xlim;
            while (ix < nx + xlim) : (ix += 1) {
                if (core.isCorner(ix, iy, iz)) continue; // C: if_outsidegc
                var pp: [CoreT.nv]f64 = undefined;
                core.p.load(ix, iy, iz, &pp);
                const geom = core.cache.fillGeometry(ix, iy, iz);
                const rvisc = try calcRijVisc(CoreT, core, visc, ix, iy, iz, &pp, &geom, global_dt);
                // [4][4]f64 is row-major contiguous → bit-identical flatten.
                const t: [16]f64 = @bitCast(rvisc);
                visc.rijvisc.store(ix, iy, iz, &t);
            }
        }
    }
}

/// C: f_flux_prime_rad_total's ifacedim>−1 branch (rad.c:3782-3786). The
/// face-averaged viscous R^i_j at the face (fx,fy,fz) in dimension `dim`
/// (between that cell and its dim−1 neighbour), read from visc.rijvisc.
pub fn faceAvg(visc: *const State, dim: usize, fx: i64, fy: i64, fz: i64) [4][4]f64 {
    var a: [16]f64 = undefined;
    var b: [16]f64 = undefined;
    visc.rijvisc.load(fx, fy, fz, &a);
    var c = [3]i64{ fx, fy, fz };
    c[dim] -= 1;
    visc.rijvisc.load(c[0], c[1], c[2], &b);
    var out: [4][4]f64 = undefined;
    for (0..4) |i| {
        for (0..4) |j| out[i][j] = 0.5 * (a[i * 4 + j] + b[i * 4 + j]);
    }
    return out;
}

/// C: the RADVISCMAXVELDAMP block of f_flux_prime_rad_total (rad.c:3799-3845)
/// p2u of the face state, then the pure velocity-damped flux addition
/// (radvisc.addViscFlux). `rijvisc` is the face-averaged R^i_j (undamped);
/// `dim` == ifacedim.
pub fn addRadViscFlux(
    comptime CoreT: type,
    core: *const CoreT,
    visc: *const State,
    ff: *[CoreT.nv]f64,
    pp: *const [CoreT.nv]f64,
    geom: *const Geometry,
    dim: usize,
    rijvisc: *const [4][4]f64,
) relele.Error!void {
    const cfg = CoreT.Cfg;
    const uu = try p2u_mod.p2u(cfg, pp.*, geom, core.phys.gam);
    radvisc.addViscFlux(
        cfg,
        ff,
        &uu,
        rijvisc,
        geom,
        dim,
        core.nDim(1) > 1,
        core.nDim(2) > 1,
        visc.params.maxvel,
    );
}
