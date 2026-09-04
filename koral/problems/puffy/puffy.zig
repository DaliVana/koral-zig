//! The PUFFY problem (C: PROBLEMS/PUFFY, PROBLEM 147): 2D axisymmetric
//! radiation-MHD limotorus around a 10 M☉ Schwarzschild hole in MKS2.
//!
//! This module transcribes the problem-side C files:
//!   tools.c: limotorus solver (lamBL/rmidlam bisections at 5ε, the
//!                 ln f quadrature; gsl_integration_qags epsrel 1e-8 is
//!                 replaced by math/quad.zig adaptive GK21 at rtol 1e-12,
//!                 so the C diff is bounded by C's own epsrel; pinned by
//!                 the epsrel-1e-12 oracle variant)
//!   prepinit.c: per-cell initial primitives (torus + atmospheres, LTE
//!                 pressure split P = bT + aT⁴, prad_ff2lab, BL→MKS2,
//!                 QUADLOOPS vector potential in the B slots)
//!   init.c: ENTR from calc_Sfromu + p2u
//!   postinit.c: global β normalization: fac = √(MAXBETA/maxβ),
//!                 BETANORMFULL (max over the whole domain)
//!   bc.c: XBCHI outflow with r-rescaling + no-inflow, XBCLO copy
//!                 (RMIN=1.85 < r_horizon=2 skips the inflow check),
//!                 YBC polar reflection (VY/B2/FY sign flip)
//! and the ko.c init sequence (ko.c:140-263): prepinit+init → set_bc →
//! calc_BfromA(p,1) → set_bc → postinit. No BC refresh after postinit;
//! ghost B keeps the pre-scaling values, exactly as in C.
//!
//! The generic pieces (redesign step 5, 2026-09-04) live in
//! problems/common/: limotorus.zig (tools.c), atmosphere.zig (the floor
//! atmospheres + the LTE pressure split), perturb.zig, magnetize.zig
//! (postinit.c β normalization), mri.zig (seed quality) and bcs.zig (the
//! bc.c fragments). This file keeps what is PUFFY's — Physics, the LT_*
//! constants, prepinit.c's per-cell composition, the init sequence and the
//! BC dispatch — and re-exports every moved name so callers are unchanged.

const std = @import("std");
const config = @import("../../config.zig");
const layout = @import("../../layout.zig");
const grid_mod = @import("../../grid.zig");
const geometry = @import("../../geometry.zig");
const relele = @import("../../relele.zig");
const frames = @import("../../frames.zig");
const p2u_mod = @import("../../p2u.zig");
const hydro = @import("../../physics/hydro.zig");
const mhd = @import("../../physics/bfield.zig");
const thermo = @import("../../physics/thermo.zig");
const radiation = @import("../../physics/radiation.zig");
const invert_rad = @import("../../solve/invert_rad.zig");
const invert = @import("../../solve/invert.zig");
const threading = @import("../../threading.zig");
const implicit = @import("../../solve/implicit.zig");
const opacities = @import("../../physics/opacities.zig");
const mesa = @import("../../physics/mesa.zig");
const radforce = @import("../../physics/radforce.zig");
const units_mod = @import("../../units.zig");
const metric = @import("../../metric/metric.zig");
const coco = @import("../../metric/coco.zig");
const precompute = @import("../../metric/precompute.zig");
const ct = @import("../../sim/ct.zig");
const sim_mod = @import("../../sim.zig");
const params_mod = @import("../../params.zig");
const magnetize = @import("../common/magnetize.zig");
const mri = @import("../common/mri.zig");
const bcs = @import("../common/bcs.zig");

const Grid = grid_mod.Grid;
const Geometry = geometry.Geometry;

// ---------------------------------------------------------------------------
// problem constants (PROBLEMS/PUFFY/define.h)

/// The fiducial (Schwarzschild) RMIN; PUFFY's define.h value, immutable. Used
/// as the reference depth for `rminForSpin`; also the value tests/goldens use.
pub const rmin_ref: f64 = 1.85;

/// UINTATMMIN / ERADATMMIN (define.h:221/225), computed once
/// (problems/common/atmosphere.zig).
pub const Atm = @import("../common/atmosphere.zig").Atm;

/// Inner boundary placed the same fraction inside the (Kerr) horizon as the
/// fiducial a = 0 setup: RMIN(a) = rmin_ref · r_h(a)/r_h(0), i.e. 0.925·r_h(a)
/// (r_h(0) = 2). This is the standard KORAL idiom; other problems define
/// `RMIN = 0.7…0.825·RH`, and it is bit-exactly rmin_ref at a = 0 (·2/2 is
/// exact), so a Schwarzschild run is unchanged. For a = 0.9375, r_h = 1.348 →
/// RMIN ≈ 1.247 (~3.5 cells inside the horizon at nx = 256), restoring the
/// clean, causally-disconnected plain-copy excision at high spin.
pub fn rminForSpin(a: f64) f64 {
    return rmin_ref * metric.rHorizonBL(a) / metric.rHorizonBL(0.0);
}

/// Runtime physics for one PUFFY run. Defaults are the validated koral_lite
/// PROBLEM 147 constants (tests and goldens use `defaults` and never go
/// through `fromParams`). `fromParams` is how a TOML preset; Sgr A*, AGN;
/// retargets the run without process-wide mutation. See
/// docs/PUFFY_AGN_DIVERGENCES.md for what a preset can and cannot match.
pub const Physics = struct {
    mass: f64 = 10.0,
    gam: f64 = 5.0 / 3.0,
    mp: metric.MetricParams = .{ .a = 0.0, .mksr0 = 0.1, .mksh0 = 0.9 },
    rmin: f64 = 1.85,
    rmax: f64 = 500.0,

    rhoatmmin: f64 = 1.0e-24,
    maxbeta: f64 = 1.0 / 20.0,
    perturb: f64 = 0.0,
    lt_kappa: f64 = 6.0e1,
    atm_tgas: f64 = 1.0e10,
    atm_trad_init: f64 = 3.0e5,
    atm_erad_factor: ?f64 = null,

    floor_params: invert.FloorParams = invert.FloorParams.puffy,
    rad_params: invert_rad.RadParams = invert_rad.RadParams.puffy,
    impl_params: implicit.ImplicitParams = implicit.ImplicitParams.puffy,
    composition: thermo.Composition = thermo.Composition.puffy,
    channels: opacities.Channels = opacities.Channels.puffy,
    scattering: bool = true,
    fluid_floor_inside_horizon: bool = false,
    alpharadvisc: f64 = 0.1,
    maxradviscvel: f64 = 0.1,
    expectedhr: f64 = 0.3,

    tsteplim: f64 = 0.5,
    nthreads: usize = 1,
    pin_threads: bool = false,
    do_radimp_fixups: bool = false,
    reduceorderatbh: bool = false,
    reduceorderafterfixup: bool = false,
    dampradwavespeednearaxis: usize = 0,

    pub fn fromParams(p: *const params_mod.Params) Physics {
        var s = Physics{};
        s.mass = p.mass;
        s.mp.a = p.bhspin;
        if (p.mksr0) |v| s.mp.mksr0 = v;
        if (p.mksh0) |v| s.mp.mksh0 = v;
        s.rmin = if (p.rmin > 0.0) p.rmin else rminForSpin(s.mp.a);
        if (p.rmax > 0.0) s.rmax = p.rmax;
        if (p.lt_kappa) |v| s.lt_kappa = v;
        if (p.maxbeta) |v| s.maxbeta = v;
        if (p.perturb) |v| s.perturb = v;
        if (p.rhoatmmin) |v| s.rhoatmmin = v;
        if (p.atm_tgas) |v| s.atm_tgas = v;
        if (p.atm_trad_init) |v| s.atm_trad_init = v;
        if (p.atm_erad_factor) |v| s.atm_erad_factor = v;
        if (p.hfrac) |h| s.composition = .{ .hfrac = h, .hefrac = p.hefrac orelse 0.0 };
        if (p.bremsstrahlung) |b| s.channels.bremsstrahlung = b;
        if (p.kleinnishina) |b| s.channels.kleinnishina = b;
        if (p.synchrotron_bridge) |b| s.channels.synchrotron_bridge = b;
        if (p.scattering) |v| s.scattering = v;
        if (p.zamo_floor_frame) |z| s.floor_params.b2rhofloorframe = if (z) .zamoframe else .driftframe;
        if (p.isentropic_b2rhofloors) |b| s.floor_params.isentropic_b2rhofloors = b;
        if (p.b2uufloor) |b| s.floor_params.b2uufloor = b;
        if (p.fluid_floor_inside_horizon) |b| s.fluid_floor_inside_horizon = b;
        if (p.opdamp_maxlevels) |n| s.impl_params.opdamp_maxlevels = n;
        if (p.opdamp_factor) |v| s.impl_params.opdamp_factor = v;
        if (p.radimp_lag_opac) |b| s.impl_params.lag_opac = b;
        if (p.scale_jacobian) |b| s.impl_params.scale_jacobian = b;
        if (p.radimp_max_en_change_down) |v| s.impl_params.max_en_change_down = v;
        if (p.radimp_max_en_change_up) |v| s.impl_params.max_en_change_up = v;
        if (p.radimp_max_damping) |v| s.impl_params.max_damping = v;
        if (p.alpharadvisc) |v| s.alpharadvisc = v;
        if (p.maxradviscvel) |v| s.maxradviscvel = v;
        if (p.expectedhr) |v| s.expectedhr = v;
        if (p.rhofloor) |v| s.floor_params.rhofloor = v;
        if (p.uurhoratiomin) |v| s.floor_params.uurhoratiomin = v;
        if (p.uurhoratiomax) |v| s.floor_params.uurhoratiomax = v;
        if (p.b2rhoratiomax) |v| s.floor_params.b2rhoratiomax = v;
        if (p.b2uuratiomax) |v| s.floor_params.b2uuratiomax = v;
        if (p.gammamaxhd) |v| s.floor_params.gammamaxhd = v;
        if (p.gammamaxrad) |v| s.rad_params.gammamaxrad = v;
        if (p.eerhoratiomin) |v| s.rad_params.eerhoratiomin = v;
        if (p.eerhoratiomax) |v| s.rad_params.eerhoratiomax = v;
        if (p.eeuuratiomin) |v| s.rad_params.eeuuratiomin = v;
        if (p.eeuuratiomax) |v| s.rad_params.eeuuratiomax = v;
        if (p.radimpeps) |v| s.impl_params.eps = v;
        if (p.radimpmaxiter) |v| s.impl_params.maxiter = v;
        s.tsteplim = p.tsteplim;
        s.nthreads = p.nthreads;
        s.pin_threads = p.pin_threads;
        s.do_radimp_fixups = p.doradimpfixups orelse false;
        s.reduceorderatbh = p.reduceorderatbh orelse false;
        s.reduceorderafterfixup = p.reduceorderafterfixup orelse false;
        s.dampradwavespeednearaxis = p.dampradwavespeednearaxis orelse 0;
        return s;
    }

    pub fn makeGrid(self: *const Physics, nx: usize, ny: usize) Grid {
        return self.makeGridNz(nx, ny, 1);
    }

    pub fn makeGridNz(self: *const Physics, nx: usize, ny: usize, nz: usize) Grid {
        return Grid.init(.{
            .nx = nx,
            .ny = ny,
            .nz = nz,
            .ng = 3,
            .minx = @log(self.rmin - self.mp.mksr0),
            .maxx = @log(self.rmax - self.mp.mksr0),
            .miny = 0.001,
            .maxy = 1.0 - 0.001,
            .minz = -std.math.pi / 4.0,
            .maxz = std.math.pi / 4.0,
        });
    }

    pub fn consts(self: *const Physics) thermo.Consts {
        return thermo.Consts.init(units_mod.Units.init(self.mass), self.composition);
    }

    pub fn atmConsts(self: *const Physics, con: *const thermo.Consts) Atm {
        return .{
            .uintatmmin = thermo.uFromTrho(con, self.atm_tgas, self.rhoatmmin, self.gam),
            .eradatmmin = if (self.atm_erad_factor) |f|
                f * con.lteEfromT(self.atm_trad_init)
            else
                con.lteEfromT(self.atm_trad_init) / 10.0 * 6.62 / self.mass,
        };
    }

    /// Sim.Options for this physics. `comm` / `decomp` stay the caller's to fill.
    /// `Bc.calc` reads `sim.opt.mp`, so no `bc_ctx` is required.
    pub fn toOptions(self: *const Physics, comptime SimT: type) SimT.Options {
        var opac = radforce.Params.puffyMassChan(self.mass, self.composition, self.channels);
        if (!self.scattering) opac.kappaes = .none;
        var floors = self.floor_params;
        if (self.fluid_floor_inside_horizon)
            floors.horizon_x1 = @log(metric.rHorizonBL(self.mp.a) - self.mp.mksr0);
        return .{
            .coords = .mks2,
            .mp = self.mp,
            .gam = self.gam,
            .tsteplim = self.tsteplim,
            .floors = floors,
            .rad = self.rad_params,
            .opac = opac,
            .implicit = self.impl_params,
            .correct_polaraxis = true,
            .nccorrectpolar = 2,
            .radviscosity = true,
            .radvisc = .{ .alpha = self.alpharadvisc, .maxvel = self.maxradviscvel },
            .dynamo = true,
            .dynamo_params = .{ .expectedhr = self.expectedhr },
            .do_radimp_fixups = self.do_radimp_fixups,
            .reduceorderatbh = self.reduceorderatbh,
            .reduceorderafterfixup = self.reduceorderafterfixup,
            .dampradwavespeednearaxis = self.dampradwavespeednearaxis,
            .bc_x = .specific,
            .bc_y = .specific,
            .bc_z = .periodic,
            .specific_bc = &Bc(SimT).calc,
            .nthreads = self.nthreads,
            .pin_threads = self.pin_threads,
        };
    }
};

/// Validated PROBLEM 147 constants. Tests and goldens that never load a
/// params file read these; production builds a value via `fromParams`.
pub const defaults: Physics = .{};

pub const gam = defaults.gam;
pub const mass = defaults.mass;
pub const mp = defaults.mp;
pub const rmin = defaults.rmin;
pub const rmax = defaults.rmax;
pub const maxbeta = defaults.maxbeta;
pub const rhoatmmin = defaults.rhoatmmin;

/// Physics + optional heap MESA table. `setup` and the replay tools share this
/// so a params file produces one value, not a 15-line mutation ritual.
pub const Loaded = struct {
    physics: Physics,
    mesa: ?*mesa.MesaTable,

    pub fn deinit(self: *Loaded, allocator: std.mem.Allocator) void {
        if (self.mesa) |t| {
            t.deinit();
            allocator.destroy(t);
        }
        self.* = undefined;
    }
};

pub fn load(allocator: std.mem.Allocator, io: std.Io, p: *const params_mod.Params) !Loaded {
    var phys = Physics.fromParams(p);
    var mesa_ptr: ?*mesa.MesaTable = null;
    errdefer if (mesa_ptr) |t| {
        t.deinit();
        allocator.destroy(t);
    };
    if (p.mesa_table.len > 0) {
        const t = try allocator.create(mesa.MesaTable);
        t.* = mesa.MesaTable.load(allocator, io, p.mesa_table) catch |err| {
            allocator.destroy(t);
            return err;
        };
        mesa_ptr = t;
        phys.channels.mesa = t;
    }
    if (phys.rmax <= phys.rmin) return error.InvalidDomain;
    return .{ .physics = phys, .mesa = mesa_ptr };
}

pub fn Setup(comptime SimT: type) type {
    return struct {
        physics: Physics,
        grid_global: Grid,
        options: SimT.Options,
        mesa: ?*mesa.MesaTable,

        pub fn deinit(self: *@This(), allocator: std.mem.Allocator) void {
            if (self.mesa) |t| {
                t.deinit();
                allocator.destroy(t);
            }
            self.* = undefined;
        }
    };
}

/// One call for the driver and for tools that rebuild a Sim from a params file.
/// `options.comm` / `options.decomp` stay unset.
pub fn setup(comptime SimT: type, allocator: std.mem.Allocator, io: std.Io, p: *const params_mod.Params) !Setup(SimT) {
    const loaded = try load(allocator, io, p);
    return .{
        .physics = loaded.physics,
        .grid_global = loaded.physics.makeGridNz(p.nx, p.ny, p.nz),
        .options = loaded.physics.toOptions(SimT),
        .mesa = loaded.mesa,
    };
}

pub const lt = struct {
    pub const xi: f64 = 0.995; // LT_XI
    pub const r1: f64 = 20.0; // LT_R1
    pub const r2: f64 = 350.0; // LT_R2
    pub const gamma: f64 = 4.0 / 3.0; // LT_GAMMA
    pub const rin: f64 = 35.0; // LT_RIN
};

/// The production grid (define.h: TNX=384, TNY=360, TNZ=1, NG=3); nx/ny can
/// be reduced for cheaper tests; the coordinate extents stay PUFFY's. The
/// 2D axisymmetric slice (nz=1). Uses `defaults` (validated extents).
pub fn makeGrid(nx: usize, ny: usize) Grid {
    return defaults.makeGrid(nx, ny);
}

/// Grid with an explicit azimuthal resolution (define.h TNZ). nz=1 is the 2D
/// axisymmetric slice and reproduces `makeGrid` byte-for-byte; nz>1 subdivides
/// the fixed PHIWEDGE=π/2 wedge with periodic z (φ) boundaries. Only the
/// resolution is tunable; the extents stay `defaults`. Production uses
/// `Physics.makeGridNz` so a params override of RMIN/RMAX is honored.
pub fn makeGridNz(nx: usize, ny: usize, nz: usize) Grid {
    return defaults.makeGridNz(nx, ny, nz);
}

/// Sim.Options from a params file. Production prefers `setup` / `toOptions`;
/// this remains for callers that already have a `Params` and no MESA table.
pub fn simOptions(comptime SimT: type, p: *const params_mod.Params) SimT.Options {
    return Physics.fromParams(p).toOptions(SimT);
}

pub fn consts() thermo.Consts {
    return defaults.consts();
}

pub fn atmConsts(con: *const thermo.Consts) Atm {
    return defaults.atmConsts(con);
}

// ---------------------------------------------------------------------------
// limotorus (PROBLEMS/PUFFY/tools.c) — the solver lives in
// problems/common/limotorus.zig; PUFFY's LT_* constants are `lt` above.
// Everything is re-exported under its old name so callers are unchanged.

pub const limotorus = @import("../common/limotorus.zig");
pub const Gd = limotorus.Gd;
pub const computeGd = limotorus.computeGd;
pub const lK = limotorus.lK;
pub const l3d = limotorus.l3d;
pub const rtbis = limotorus.rtbis;
pub const LamF = limotorus.LamF;
pub const lamBL = limotorus.lamBL;
pub const omega3d = limotorus.omega3d;
pub const computeAgrav = limotorus.computeAgrav;
pub const RmidF = limotorus.RmidF;
pub const findRml = limotorus.findRml;
pub const LnfIntegrand = limotorus.LnfIntegrand;
pub const TorusConsts = limotorus.TorusConsts;
pub const DsVels = limotorus.DsVels;

/// PUFFY's limotorus constants (define.h LT_*) with this run's LT_KAPPA.
pub fn ltParams(kappa: f64) limotorus.Params {
    return .{ .xi = lt.xi, .r1 = lt.r1, .r2 = lt.r2, .gamma = lt.gamma, .rin = lt.rin, .kappa = kappa };
}

/// tools.c:239-253 for PUFFY's window (κ does not enter).
pub fn torusConsts(a: f64) TorusConsts {
    return limotorus.torusConsts(a, ltParams(defaults.lt_kappa));
}

/// tools.c:195 init_dsandvels_limotorus with the validated LT_KAPPA.
pub fn initDsandvels(r: f64, th: f64, a: f64, tc: *const TorusConsts) DsVels {
    return limotorus.initDsandvels(r, th, a, tc, ltParams(defaults.lt_kappa));
}

// ---------------------------------------------------------------------------
// atmospheres (relele.c:518 / :718) — problems/common/atmosphere.zig

pub const atmosphere = @import("../common/atmosphere.zig");
pub const setRadAtmosphere = atmosphere.setRadAtmosphere;
pub const tFromPtot = atmosphere.tFromPtot;

/// set_hdatmosphere with this problem's RHOATMMIN and metric.
pub fn setHdAtmosphere(
    comptime cfg: config.Config,
    pp: *[layout.VarLayout(cfg).count]f64,
    geom: *const Geometry,
    atm: *const Atm,
    phys: *const Physics,
) void {
    atmosphere.setHdAtmosphere(cfg, pp, geom, atm, phys.rhoatmmin, phys.mp);
}

// ---------------------------------------------------------------------------
// prepinit.c — per-cell initial primitives (vector potential in the B slots)

/// BL-coords geometry at the cell center (C: fill_geometry_arb KERRCOORDS),
/// supplying this problem's fixed metric params. The reduction lives once in
/// precompute.geometryBLat.
pub fn fillGeometryBL(g: *const Grid, coords: config.Coords, ix: i64, iy: i64, iz: i64) Geometry {
    return precompute.geometryBLat(g, coords, defaults.mp, ix, iy, iz);
}

/// The MRI-seeding noise (problems/common/perturb.zig).
pub const perturbXi = @import("../common/perturb.zig").perturbXi;

pub fn prepInitCell(
    comptime cfg: config.Config,
    geom: *const Geometry,
    geomBL: *const Geometry,
    tc: *const TorusConsts,
    con: *const thermo.Consts,
    atm: *const Atm,
) relele.Error![layout.VarLayout(cfg).count]f64 {
    return prepInitCellWith(cfg, geom, geomBL, tc, con, atm, &defaults);
}

pub fn prepInitCellWith(
    comptime cfg: config.Config,
    geom: *const Geometry,
    geomBL: *const Geometry,
    tc: *const TorusConsts,
    con: *const thermo.Consts,
    atm: *const Atm,
    phys: *const Physics,
) relele.Error![layout.VarLayout(cfg).count]f64 {
    const L = layout.VarLayout(cfg);
    const has_rad = comptime cfg.has(.radiation);
    const has_b = comptime L.hasVar(.b1);

    const r = geomBL.xxvec[1];
    const th = geomBL.xxvec[2];

    const dv = limotorus.initDsandvels(r, th, phys.mp.a, tc, ltParams(phys.lt_kappa));
    const rho = dv.rho;
    var uint = dv.uu;
    var ell = dv.ell;

    var pp = [_]f64{0} ** L.count;

    if (rho < 0.0) { // outside the donut
        setHdAtmosphere(cfg, &pp, geom, atm, phys);
        if (has_rad) setRadAtmosphere(cfg, &pp, geom, atm);
        pp[L.index(.entr)] = -1.0; // marker only; init overwrites
        return pp;
    }

    // inside the donut
    pp[L.index(.entr)] = 1.0;

    var ppback = [_]f64{0} ** L.count;
    setHdAtmosphere(cfg, &ppback, geom, atm, phys);
    if (has_rad) setRadAtmosphere(cfg, &ppback, geom, atm);

    uint = phys.lt_kappa * std.math.pow(f64, rho, lt.gamma) / (lt.gamma - 1.0);
    // optional MRI-seeding noise, torus interior only — applied BEFORE the
    // gas/radiation pressure split so the split stays LTE-consistent
    if (phys.perturb != 0.0) uint *= 1.0 + phys.perturb * perturbXi(geom.xxvec);
    ell *= -1.0;

    const GGBL = &geomBL.GG;
    const ulph = @sqrt(-1.0 / (GGBL[0][0] / ell / ell + 2.0 / ell * GGBL[0][3] + GGBL[3][3]));
    const ult = ulph / ell;
    const ucov = [4]f64{ ult, 0.0, 0.0, ulph };
    var ucon = relele.raiseVec(ucov, geomBL);
    ucon = try relele.convert(ucon, .vel4, .velr, geomBL, .trust_ut);

    pp[L.index(.rho)] = @max(rho, ppback[L.index(.rho)]);
    pp[L.index(.uu)] = @max(uint, ppback[L.index(.uu)]);
    pp[L.index(.vx)] = ucon[1];
    pp[L.index(.vy)] = ucon[2];
    pp[L.index(.vz)] = ucon[3];
    // B slots zeroed so the coordinate transformation below stays sane

    if (has_rad) {
        // distribute P = GAMMAM1·uint between gas and radiation by solving
        // P = bbb·T + aaa·T⁴ (prepinit.c:86-97, Mathematica closed form)
        const P = (phys.gam - 1.0) * uint;
        const aaa = con.four_sigmarad / 3.0; // C: 4.*SIGMA_RAD/3.
        const bbb = con.kb_over_mugas_mp * rho; // C: K_BOLTZ*rho/MU_GAS/M_PROTON
        const t4 = tFromPtot(P, aaa, bbb);

        const E = con.lteEfromT(t4);
        uint = thermo.uFromTrho(con, t4, rho, phys.gam);

        pp[L.index(.uu)] = @max(uint, ppback[L.index(.uu)]);
        pp[L.index(.ee)] = @max(E, ppback[L.index(.ee)]);
        pp[L.index(.fx)] = 0.0;
        pp[L.index(.fy)] = 0.0;
        pp[L.index(.fz)] = 0.0;

        // BL fluid-frame radiative primitives → BL lab
        pp = try radiation.pradFf2Lab(cfg, pp, geomBL, phys.rad_params);
    }

    // BL → MYCOORDS
    pp = try frames.transPallCoco(cfg, pp, geomBL, geom, phys.mp);

    if (has_b) {
        // MYCOORDS vector potential (only A_φ), stored in the B slots
        const base = pp[L.index(.rho)] * geomBL.xxvec[1] / 4.0e-22;
        var acov3 = @max(base * base - 0.02, 0.0) * @sqrt(1.0e-23);
        // QUADLOOPS
        acov3 *= @sin((std.math.pi / 2.0 - geomBL.xxvec[2]) / 0.1);
        pp[L.index(.b1)] = 0.0;
        pp[L.index(.b2)] = 0.0;
        pp[L.index(.b3)] = acov3;
    }

    return pp;
}

// ---------------------------------------------------------------------------
// the full init sequence (ko.c:140-263 with ifinit == 1)

/// prepinit + init over the domain (leaves the vector potential A_φ in the
/// B slots, ENTR = calc_Sfromu, p2u). Does NOT fill ghosts; the caller
/// runs set_bc next, matching ko.c's set_initial_profile → set_bc.
/// Band-parallel over iy: every cell writes only its own p/u slots and reads
/// only the (already-built, const) metric cache, so the initialized state is
/// bit-identical at any thread count. This is the dominant startup cost;
/// the limotorus solve per cell is heavy in pow/log/exp, and left serial it
/// ignored `nthreads` entirely.
pub fn prepInitDomain(comptime SimT: type, sim: *SimT) !void {
    return prepInitDomainWith(SimT, sim, &defaults);
}

pub fn prepInitDomainWith(comptime SimT: type, sim: *SimT, phys: *const Physics) !void {
    const Ctx = struct { sim: *SimT, phys: *const Physics };
    var ctx = Ctx{ .sim = sim, .phys = phys };
    const W = struct {
        fn rows(c: *Ctx, iy0: i64, iy1: i64, res: *threading.ChunkResult) void {
            prepInitRows(SimT, c.sim, c.phys, iy0, iy1) catch |e| {
                res.err = e;
            };
        }
    };
    const res = threading.parallelRange(Ctx, &ctx, sim.team, 0, @intCast(sim.grid.ny), W.rows);
    if (res.err) |e| return e;
}

fn prepInitRows(comptime SimT: type, sim: *SimT, phys: *const Physics, iy0: i64, iy1: i64) !void {
    const cfg = SimT.Cfg;
    const L = SimT.Layout;

    const con = phys.consts();
    const atm = phys.atmConsts(&con);
    const tc = torusConsts(phys.mp.a);

    const nx: i64 = @intCast(sim.grid.nx);
    const nz: i64 = @intCast(sim.grid.nz);

    var iz: i64 = 0;
    while (iz < nz) : (iz += 1) {
        var iy: i64 = iy0;
        while (iy < iy1) : (iy += 1) {
            var ix: i64 = 0;
            while (ix < nx) : (ix += 1) {
                const geom = sim.cache.fillGeometry(ix, iy, iz);
                const geomBL = precompute.geometryBLat(&sim.grid, cfg.coords, phys.mp, ix, iy, iz);
                var pp = try prepInitCellWith(cfg, &geom, &geomBL, &tc, &con, &atm, phys);
                pp[L.index(.entr)] = hydro.sFromU(pp[L.index(.rho)], pp[L.index(.uu)], sim.opt.gam);
                try sim.initCell(ix, iy, iz, pp);
            }
        }
    }
}

/// The full ko.c:140-263 init sequence: prepinit + init → set_bc →
/// calc_BfromA → set_bc → postinit. Returns the β-normalization factor
/// fac (postinit.c:78).
pub fn initAll(comptime SimT: type, sim: *SimT) !f64 {
    return initAllWith(SimT, sim, &defaults);
}

pub fn initAllWith(comptime SimT: type, sim: *SimT, phys: *const Physics) !f64 {
    try prepInitDomainWith(SimT, sim, phys);
    // MPI: each setBc needs exchanged z-ghosts first (its interior-face path
    // is p2u-of-exchanged-p; before the first exchange the ghost planes are
    // still the Field.init zeros). No-ops serially.
    sim.exchangeHalos();
    try sim.setBc(0.0, true);
    try ct.calcBfromA(SimT, sim, true);
    sim.exchangeHalos(); // domain B changed — refresh ghosts before the re-fill
    try sim.setBc(0.0, true);
    return try postinitWith(SimT, sim, phys);
}

/// postinit.c: global β normalization (problems/common/magnetize.zig).
/// Returns fac; ghost cells keep their unscaled B (C does not refresh BCs
/// after postinit).
pub fn postinit(comptime SimT: type, sim: *SimT) !f64 {
    return postinitWith(SimT, sim, &defaults);
}

pub fn postinitWith(comptime SimT: type, sim: *SimT, phys: *const Physics) !f64 {
    return magnetize.normalizeBeta(SimT, sim, phys.maxbeta);
}

// ---------------------------------------------------------------------------
// bc.c — problem-specific boundary conditions

// ---------------------------------------------------------------------------
// seed MRI-quality report (problems/common/mri.zig)

pub const SeedQ = mri.SeedQ;

pub fn seedQuality(comptime SimT: type, s: *SimT) !SeedQ {
    return seedQualityWith(SimT, s, &defaults);
}

pub fn seedQualityWith(comptime SimT: type, s: *SimT, phys: *const Physics) !SeedQ {
    return mri.seedQuality(SimT, s, .{ .mksr0 = phys.mp.mksr0, .rhoatmmin = phys.rhoatmmin, .gam = phys.gam });
}

// ---------------------------------------------------------------------------
// bc.c — the problem's boundary conditions, composed from the stock
// fragments in problems/common/bcs.zig: XBCHI outflow with r-rescaling +
// no-inflow, XBCLO plain copy (RMIN=1.85 < r_horizon=2 skips the inflow
// check), YBC polar reflection. TNZ == 1 has no z ghosts; nz > 1 is
// periodic in φ and never reaches this callback.

pub fn Bc(comptime SimT: type) type {
    return struct {
        const NV = SimT.nv;

        pub fn calc(
            ctx: ?*const anyopaque,
            sim: *const SimT,
            ix: i64,
            iy: i64,
            iz: i64,
            t: f64,
            ifinit: bool,
            face: sim_mod.BcFace,
        ) relele.Error![NV]f64 {
            _ = ctx; // Sim.opt.mp carries the problem metric; no extra context
            _ = t;
            _ = ifinit;
            return switch (face) {
                .xhi => try bcs.outflowRescaleXhi(SimT, sim, ix, iy, iz),
                .xlo => bcs.copyXlo(SimT, sim, iy, iz),
                .ylo, .yhi => bcs.polarReflect(SimT, sim, ix, iy, iz, face),
                .zlo, .zhi => unreachable,
            };
        }
    };
}
