//! mhdtorus: make the snapshot for the GRRT cross-check against ipole
//! (docs/RENDER.md, "Snapshot cross-check against ipole").
//!
//! PUFFY's torus is radiation-supported, sits at r = 35..450 and is far too
//! cold to emit synchrotron radiation (theta_e < 0.1 even without
//! radiation). The standard target of an EHT-style comparison is the
//! opposite regime: a hot, optically thin, magnetised flow close to the
//! hole. This tool evolves such a flow with the same code, without the
//! radiation module (hydro + MHD, Gamma = 4/3):
//!
//!  * the torus is the same limotorus solution (problems/common/
//!    limotorus.zig) with the thick, sub-Keplerian shape of Penna, Kulkarni
//!    & Narayan (2013): xi = 0.708, r1 = 42, r2 = 1000, inner edge at 10.
//!    Its enthalpy gives p / rho ~ 1e-2, i.e. T ~ 1e11 K;
//!  * the seed field is a single poloidal loop, A_phi = max(rho / rho_max
//!    - 0.2, 0), normalized to the toml's `maxbeta` like PUFFY's;
//!  * the atmosphere, perturbation, boundary conditions and polar-axis
//!    treatment are PUFFY's; the mean-field dynamo is off (plain ideal
//!    MHD).
//!
//! How long it can run: the evolution is stable to at least t = 1000 (the
//! longest tried). The funnel above the hole becomes magnetically dominated
//! near t = 400 and b^2 / rho then sits at the floor ratio, as it should.
//! Two things to know before taking a late snapshot:
//!  * this torus closes over the poles beyond r ~ 60, so a column of
//!    unmagnetised gas stands above the empty funnel at t = 0. It falls in
//!    along the axis and reaches the hole between t = 500 and 700, after
//!    which the funnel is no longer evacuated. That is the initial data,
//!    not the solver;
//!  * in axisymmetry the MRI saturates in channel flows: from t ~ 800 dense
//!    streams (rho ~ 10 rho_max) cross the horizon.
//! (Until 2026-11-20 the run blew up at t ~ 450: the polar-axis EMF zeroing
//! of C's flux-CT was missing, sim/polaraxis.zig zeroAxisEmfs.)
//!
//! The polytropic constant is chosen so that rho_max = 1 in code units.
//! Ideal MHD without radiation is invariant under (rho, u, B^2) ->
//! s (rho, u, B^2); `--scale s` applies that exact symmetry on output, which
//! is how the accretion rate of a scale-free GRMHD run is matched to a
//! source (ipole's M_unit). Geometry, velocities and temperatures are
//! untouched by it. The result is written as an ordinary 13-variable PUFFY
//! KDMP (radiation slots empty) so that kdmp2png and kdmp2ipole read it
//! unchanged.
//!
//! usage: mhdtorus <params.toml> <out.kdmp> [--tmax T] [--scale S]
//!   the toml gives the grid, mass, spin, maxbeta and perturbation
//!   (koral/problems/puffy/puffy_ipole.toml)

const std = @import("std");
const koral = @import("koral");

const puffy = koral.problems.puffy;
const limotorus = koral.problems.common.limotorus;
const magnetize = koral.problems.common.magnetize;
const precompute = koral.metric.precompute;
const relele = koral.relele;
const frames = koral.frames;
const hydro = koral.physics.hydro;
const dump = koral.io.dump;

/// Penna, Kulkarni & Narayan (2013) thick torus.
const torus_shape = struct {
    const xi: f64 = 0.708;
    const r1: f64 = 42.0;
    const r2: f64 = 1000.0;
    const rin: f64 = 10.0;
};

const cfg_mhd = koral.Config{
    .modules = &.{ .hydro, .mhd },
    .reconstruction = .ppm,
    .flux = .laxf,
    .timestepping = .rk2imex,
    .coords = .mks2,
};
const SimM = koral.Sim(cfg_mhd);
const LM = SimM.Layout;
const LP = koral.VarLayout(koral.config.puffy);

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    var args = std.process.Args.Iterator.init(init.minimal.args);
    defer args.deinit();
    _ = args.next();
    var ppath: ?[]const u8 = null;
    var opath: ?[]const u8 = null;
    var tmax: ?f64 = null;
    var scale: f64 = 1.0;
    while (args.next()) |a| {
        if (std.mem.eql(u8, a, "--tmax")) {
            tmax = try std.fmt.parseFloat(f64, args.next() orelse return usage());
        } else if (std.mem.eql(u8, a, "--scale")) {
            scale = try std.fmt.parseFloat(f64, args.next() orelse return usage());
        } else if (ppath == null) {
            ppath = a;
        } else if (opath == null) {
            opath = a;
        } else return usage();
    }
    if (ppath == null or opath == null) return usage();

    var p = try koral.Params.load(allocator, io, ppath.?);
    defer p.deinit(allocator);
    var loaded = try puffy.load(allocator, io, &p);
    defer loaded.deinit(allocator);
    var phys = loaded.physics;
    phys.gam = puffy.lt.gamma; // the torus' own polytropic index: exact equilibrium
    const t_end = tmax orelse p.tmax;

    const grid = phys.makeGridNz(p.nx, p.ny, p.nz);
    var sim = try SimM.init(allocator, grid, .{
        .phys = .{ .coords = .mks2, .mp = phys.mp, .gam = phys.gam, .floors = phys.floor_params },
        .num = .{ .tsteplim = phys.tsteplim, .polaraxis = .{ .ncells = 2 } },
        .bc = .{
            .x = .{ .specific = .{ .f = &puffy.Bc(SimM).calc } },
            .y = .{ .specific = .{ .f = &puffy.Bc(SimM).calc } },
            .z = .periodic,
        },
        .parallel = .{ .nthreads = phys.nthreads },
    });
    defer sim.deinit();

    // ---- initial state: limotorus + atmosphere, A_phi in the B slots ----
    const gam = phys.gam;
    var ltp = limotorus.Params{ .xi = torus_shape.xi, .r1 = torus_shape.r1, .r2 = torus_shape.r2, .gamma = gam, .rin = torus_shape.rin, .kappa = 1.0 };
    const tc = limotorus.torusConsts(phys.mp.a, ltp);
    // rho ~ kappa^(-1/(Gamma-1)): pick kappa so that rho_max = 1
    var rho_max: f64 = 0;
    {
        var iy: i64 = 0;
        while (iy < sim.nyi()) : (iy += 1) {
            var ix: i64 = 0;
            while (ix < sim.nxi()) : (ix += 1) {
                const gbl = precompute.geometryBLat(&grid, .mks2, phys.mp, ix, iy, 0);
                rho_max = @max(rho_max, limotorus.initDsandvels(gbl.xxvec[1], gbl.xxvec[2], phys.mp.a, &tc, ltp).rho);
            }
        }
    }
    if (!(rho_max > 0)) return error.NoTorus;
    ltp.kappa = std.math.pow(f64, rho_max, gam - 1.0);
    phys.rhoatmmin = 1.0e-6; // of rho_max
    const con = phys.consts();
    const atm = phys.atmConsts(&con);

    var iz: i64 = 0;
    while (iz < sim.nzi()) : (iz += 1) {
        var iy: i64 = 0;
        while (iy < sim.nyi()) : (iy += 1) {
            var ix: i64 = 0;
            while (ix < sim.nxi()) : (ix += 1) {
                const geom = sim.core.cache.fillGeometry(ix, iy, iz);
                const gbl = precompute.geometryBLat(&grid, .mks2, phys.mp, ix, iy, iz);
                var pp: [LM.count]f64 = @splat(0);
                puffy.setHdAtmosphere(cfg_mhd, &pp, &geom, &atm, &phys);
                const dv = limotorus.initDsandvels(gbl.xxvec[1], gbl.xxvec[2], phys.mp.a, &tc, ltp);
                if (dv.rho > pp[LM.index(.rho)]) {
                    const back = pp;
                    var uint = dv.uu;
                    if (phys.perturb != 0.0) uint *= 1.0 + phys.perturb * puffy.perturbXi(geom.xxvec);
                    // rotation: u_phi / u_t = -l  (tools.c / prepinit.c)
                    const ell = -dv.ell;
                    const GG = &gbl.GG;
                    const ulph = @sqrt(-1.0 / (GG[0][0] / ell / ell + 2.0 / ell * GG[0][3] + GG[3][3]));
                    const ucov = [4]f64{ ulph / ell, 0.0, 0.0, ulph };
                    var ucon = relele.raiseVec(ucov, &gbl);
                    ucon = try relele.convert(ucon, .vel4, .velr, &gbl, .trust_ut);
                    pp = @splat(0);
                    pp[LM.index(.rho)] = dv.rho;
                    pp[LM.index(.uu)] = @max(uint, back[LM.index(.uu)]);
                    pp[LM.index(.vx)] = ucon[1];
                    pp[LM.index(.vy)] = ucon[2];
                    pp[LM.index(.vz)] = ucon[3];
                    pp = try frames.transPallCoco(cfg_mhd, pp, &gbl, &geom, phys.mp);
                    // single poloidal loop
                    pp[LM.index(.b3)] = @max(dv.rho - 0.2, 0.0);
                }
                pp[LM.index(.entr)] = hydro.sFromU(pp[LM.index(.rho)], pp[LM.index(.uu)], gam);
                try sim.initCell(ix, iy, iz, pp);
            }
        }
    }
    try sim.setBc(0.0, true);
    try sim.calcBfromA(true);
    try sim.setBc(0.0, true);
    const fac = try magnetize.normalizeBeta(SimM.CoreT, &sim.core, phys.maxbeta);
    try sim.finishInit();
    std.debug.print("mhdtorus: {d}x{d}x{d} cells, a = {d}, Gamma = {d:.4}, kappa = {e:.4}, beta-normalization {e:.3}, perturbation {d}; evolving to t = {d}\n", .{ p.nx, p.ny, p.nz, phys.mp.a, gam, ltp.kappa, fac, phys.perturb, t_end });

    var next_report: f64 = 0;
    while (sim.t < t_end) {
        try sim.step(null);
        if (sim.t >= next_report) {
            // sanity: the torus peak stays O(1) and b^2 / rho stays under the floor ratio
            var rmax: f64 = 0;
            var smax: f64 = 0;
            var jy: i64 = 0;
            while (jy < sim.nyi()) : (jy += 1) {
                var jx: i64 = 0;
                while (jx < sim.nxi()) : (jx += 1) {
                    const rho = sim.core.p.get(LM.index(.rho), jx, jy, 0);
                    rmax = @max(rmax, rho);
                    const geom = sim.core.cache.fillGeometry(jx, jy, 0);
                    var pp: [LM.count]f64 = undefined;
                    sim.core.p.load(jx, jy, 0, &pp);
                    const u = relele.uconUcovFromPrims(.{ pp[LM.index(.vx)], pp[LM.index(.vy)], pp[LM.index(.vz)] }, &geom) catch continue;
                    const bf = koral.physics.mhd.bconBcovBsqFrom4vel(.{ pp[LM.index(.b1)], pp[LM.index(.b2)], pp[LM.index(.b3)] }, u.con, u.cov, &geom);
                    smax = @max(smax, bf.bsq / rho);
                }
            }
            std.debug.print("mhdtorus: t = {d:.1}, step {d}, dt = {e:.3}, max rho {e:.3}, max b^2/rho {e:.3}\n", .{ sim.t, sim.nstep, sim.cflDt(), rmax, smax });
            next_report += t_end / 10.0;
        }
    }

    // 13-variable PUFFY layout, radiation slots empty, (rho, u, B^2) scaled
    const n = p.nx * p.ny * p.nz;
    const bytes = try allocator.alloc(u8, dump.header_size + n * LP.count * 8);
    defer allocator.free(bytes);
    var w = dump.writeDumpHeader(bytes, .{
        .nx = @intCast(p.nx),
        .ny = @intCast(p.ny),
        .nz = @intCast(p.nz),
        .nv = @intCast(LP.count),
        .t = sim.t,
        .nstep = sim.nstep,
        .out_idx = 0,
    });
    const sb = @sqrt(scale);
    iz = 0;
    while (iz < sim.nzi()) : (iz += 1) {
        var iy: i64 = 0;
        while (iy < sim.nyi()) : (iy += 1) {
            var ix: i64 = 0;
            while (ix < sim.nxi()) : (ix += 1) {
                var pp: [LP.count]f64 = @splat(0);
                const rho = scale * sim.core.p.get(LM.index(.rho), ix, iy, iz);
                const uu = scale * sim.core.p.get(LM.index(.uu), ix, iy, iz);
                pp[LP.index(.rho)] = rho;
                pp[LP.index(.uu)] = uu;
                pp[LP.index(.entr)] = hydro.sFromU(rho, uu, phys.gam);
                inline for (.{ .vx, .vy, .vz }) |v| pp[LP.index(v)] = sim.core.p.get(LM.index(v), ix, iy, iz);
                inline for (.{ .b1, .b2, .b3 }) |v| pp[LP.index(v)] = sb * sim.core.p.get(LM.index(v), ix, iy, iz);
                pp[LP.index(.ee)] = 1.0e-300; // no radiation field
                for (pp) |v| {
                    std.mem.writeInt(u64, bytes[w..][0..8], @bitCast(v), .little);
                    w += 8;
                }
            }
        }
    }
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = opath.?, .data = bytes[0..w] });
    std.debug.print("mhdtorus: wrote {s} at t = {d:.2} ({d} steps), density scale {e:.3}\n", .{ opath.?, sim.t, sim.nstep, scale });
}

fn usage() !void {
    std.debug.print("usage: mhdtorus <params.toml> <out.kdmp> [--tmax T] [--scale S]\n", .{});
    return error.BadArgs;
}
