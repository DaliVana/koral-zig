//! Silo (`.silo`) export for VisIt — a port of the PUFFY-relevant subset of
//! KORAL's `silo.c`. Writes a `DB_PDB` non-collinear quad mesh (`mesh1`) plus
//! zone-centered fields, linking LLNL Silo directly (VisIt's bundled
//! `libsiloh5`) so the files are byte-compatible with what VisIt 3.5 reads.
//!
//! Field names / centering mirror `silo.c` so existing VisIt sessions and
//! expressions carry over — including PUFFY's two silo.c additions over the
//! base: the `PRINTKAPPASTOSILO` opacity block (`kappa_gas_abs`, `kappa_rad_abs`,
//! `kappa_gas_num`, `kappa_rad_num`, `kappa_gas_ross`, `kappa_rad_ross`,
//! `tot_emissivity`, `kappa_es`) and the `PRINT_FIXUPS_TO_SILO` flags
//! (`fixups_u2pmhd`, `fixups_u2prad`, `fixups_radimp`, `fixups`). The mesh nodes
//! are the cell *boundaries* transformed
//! to Cartesian (via the OUTCOORDS = BL spherical coordinates), and — as in
//! PUFFY's `SILO2D_XZPLANE` — a 2D (nz==1) run is laid out in the meridional
//! (x,z) plane with the in-plane vector component set to the physical z-part.
//!
//! Gated by the `-Dsilo` build flag (`build_options.silo`). When it is off,
//! `enabled == false` and `write()` is a comptime no-op whose body is never
//! analyzed — so no Silo symbols are referenced and `zig build test` plus any
//! Silo-less machine keep building. Only the `-Dsilo` problem executables link
//! `libsiloh5`.

const std = @import("std");
const build_options = @import("build_options");
const geometry = @import("../geometry.zig");
const coco = @import("../metric/coco.zig");
const precompute = @import("../metric/precompute.zig");
const relele = @import("../relele.zig");
const radiation = @import("../physics/radiation.zig");
const radforce = @import("../physics/radforce.zig");
const frames = @import("../frames.zig");
const bfield = @import("../physics/bfield.zig");
const thermo = @import("../physics/thermo.zig");
const ct = @import("../magn/ct.zig");

const Geometry = geometry.Geometry;

/// True when the executable was built with `-Dsilo` (and thus links Silo).
pub const enabled = build_options.silo;

pub const Options = struct {
    /// 2D (nz==1) runs render in the meridional (x,z) plane (PUFFY's
    /// SILO2D_XZPLANE); false lays them out in (x,y).
    xz_plane: bool = true,
    /// Multiply all node coordinates (e.g. GU→parsec). 1 = code units.
    coordscale: f64 = 1.0,
};

// ---------------------------------------------------------------------------
// LLNL Silo C API (v4.x) — hand-declared so no silo.h include is needed at
// build time. Constant values and prototypes are from VisIt 3.5's bundled
// silo.h; the ABI is plain C pointers/ints. These declarations emit no symbols
// unless referenced, which only happens inside the `if (comptime enabled)`
// branch below.
const DBfile = opaque {};
const DBoptlist = opaque {};

const DB_PDB: c_int = 2;
const DB_CLOBBER: c_int = 0;
const DB_LOCAL: c_int = 0;
const DB_DOUBLE: c_int = 20;
const DB_NONCOLLINEAR: c_int = 131;
const DB_ZONECENT: c_int = 111;
const DBOPT_CYCLE: c_int = 263;
const DBOPT_TIME: c_int = 275; // float*
const DBOPT_DTIME: c_int = 280; // double*

// `DBCreate` is a version-checking macro around `DBCreateReal` (same
// signature); the plain symbol is not exported, so we call Real directly.
extern fn DBCreateReal(name: [*:0]const u8, mode: c_int, target: c_int, finfo: ?[*:0]const u8, ftype: c_int) ?*DBfile;
extern fn DBClose(f: *DBfile) c_int;
extern fn DBMakeOptlist(maxopts: c_int) ?*DBoptlist;
extern fn DBFreeOptlist(ol: *DBoptlist) c_int;
extern fn DBAddOption(ol: *DBoptlist, opt: c_int, val: *const anyopaque) c_int;
extern fn DBPutQuadmesh(f: *DBfile, name: [*:0]const u8, coordnames: [*]const [*:0]const u8, coords: [*]const [*]const f64, dims: [*]const c_int, ndims: c_int, datatype: c_int, coordtype: c_int, ol: ?*DBoptlist) c_int;
extern fn DBPutQuadvar1(f: *DBfile, name: [*:0]const u8, meshname: [*:0]const u8, v: *const anyopaque, dims: [*]const c_int, ndims: c_int, mixvar: ?*const anyopaque, mixlen: c_int, datatype: c_int, centering: c_int, ol: ?*DBoptlist) c_int;
extern fn DBPutQuadvar(f: *DBfile, name: [*:0]const u8, meshname: [*:0]const u8, nvars: c_int, varnames: [*]const [*:0]const u8, vars: [*]const *const anyopaque, dims: [*]const c_int, ndims: c_int, mixvars: ?*const anyopaque, mixlen: c_int, datatype: c_int, centering: c_int, ol: ?*DBoptlist) c_int;

/// Project a spherical-basis vector `(vr, vth, vph)` (θ,φ components already
/// scaled to orthonormal) at direction `(r,θ,φ)` onto Cartesian axes — the
/// rotation `silo.c` applies before writing velocity/B/flux vectors.
fn sphToCart(th: f64, ph: f64, vr: f64, vth: f64, vph: f64) [3]f64 {
    const st = @sin(th);
    const ct_ = @cos(th);
    const sp = @sin(ph);
    const cp = @cos(ph);
    return .{
        st * cp * vr + ct_ * cp * vth - sp * vph,
        st * sp * vr + ct_ * sp * vth + cp * vph,
        ct_ * vr - st * vth,
    };
}

/// Write a `.silo` snapshot of `sim` to `path` (must be NUL-terminated). No-op
/// unless the executable was built with `-Dsilo`. `time`/`cycle` become the
/// mesh's DTOUT time / cycle metadata for VisIt's time slider.
pub fn write(
    comptime SimT: type,
    sim: *const SimT,
    allocator: std.mem.Allocator,
    path: [:0]const u8,
    time: f64,
    cycle: i32,
    opts: Options,
) !void {
    if (comptime enabled) {
        return writeImpl(SimT, sim, allocator, path, time, cycle, opts);
    }
}

fn writeImpl(
    comptime SimT: type,
    sim: *const SimT,
    allocator: std.mem.Allocator,
    path: [:0]const u8,
    time: f64,
    cycle: i32,
    opts: Options,
) !void {
    const cfg = SimT.Cfg;
    const L = SimT.Layout;
    const coords = cfg.coords;
    const mp = sim.opt.mp;
    const gam = sim.opt.gam;
    const spherical = coords != .mink;
    const has_b = comptime L.hasVar(.b1);
    const has_rad = comptime L.hasVar(.ee);

    const nx = sim.nxi();
    const ny = sim.nyi();
    const nz = sim.nzi();

    // Mesh dimensionality (silo.c: 1D / 2D / 2D-switched / 3D).
    const ndim: c_int = if (ny == 1 and nz == 1) 1 else if (nz == 1 or ny == 1) 2 else 3;
    const xz = opts.xz_plane and nz == 1 and ny > 1;

    // Node counts: one more than the zones per non-degenerate dimension.
    const nnx: usize = @intCast(nx + 1);
    const nny: usize = if (ny == 1) 1 else @intCast(ny + 1);
    const nnz: usize = if (nz == 1) 1 else @intCast(nz + 1);
    const nnode = nnx * nny * nnz;
    const ncell: usize = @intCast(nx * ny * nz);

    // thermo constants (for temp / trad) come from the opacity params; absent
    // → those fields stay 0.
    var consts_ptr: ?*const thermo.Consts = null;
    var par_ptr: ?*const radforce.Params = null;
    if (sim.opt.opac) |*op| {
        consts_ptr = &op.consts;
        par_ptr = op;
    }

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const a = arena.allocator();

    // ---- node coordinates (cell boundaries → Cartesian) -------------------
    const nodex = try a.alloc(f64, nnode);
    const nodey = try a.alloc(f64, nnode);
    const nodez = try a.alloc(f64, nnode);
    {
        var imz: usize = 0;
        while (imz < nnz) : (imz += 1) {
            var imy: usize = 0;
            while (imy < nny) : (imy += 1) {
                var imx: usize = 0;
                while (imx < nnx) : (imx += 1) {
                    const ix: i64 = @intCast(imx);
                    const iy: i64 = @intCast(imy);
                    const iz: i64 = @intCast(imz);
                    // silo.c node coordinate: boundary in each evolved
                    // dimension, cell center in a collapsed one.
                    const x1 = sim.grid.xl(ix);
                    const x2 = if (ny == 1) sim.grid.yc(0) else sim.grid.yl(iy);
                    const x3 = if (nz == 1) sim.grid.zc(0) else sim.grid.zl(iz);
                    var cx: f64 = undefined;
                    var cy: f64 = undefined;
                    var cz: f64 = undefined;
                    if (spherical) {
                        const bl = coco.cocoN(.{ 0, x1, x2, x3 }, coords, .bl, mp);
                        const r = bl[1];
                        const th = bl[2];
                        const ph = bl[3];
                        cx = r * @sin(th) * @cos(ph);
                        cy = r * @sin(th) * @sin(ph);
                        cz = r * @cos(th);
                    } else {
                        cx = x1;
                        cy = x2;
                        cz = x3;
                    }
                    const ni = imz * (nny * nnx) + imy * nnx + imx;
                    nodex[ni] = cx * opts.coordscale;
                    nodey[ni] = cy * opts.coordscale;
                    nodez[ni] = cz * opts.coordscale;
                }
            }
        }
    }

    // ---- zone fields ------------------------------------------------------
    const rho = try a.alloc(f64, ncell);
    const uint = try a.alloc(f64, ncell);
    const entr = try a.alloc(f64, ncell);
    const temp = try a.alloc(f64, ncell);
    const gammag = try a.alloc(f64, ncell);
    const uc0 = try a.alloc(f64, ncell);
    const uc1 = try a.alloc(f64, ncell);
    const uc2 = try a.alloc(f64, ncell);
    const uc3 = try a.alloc(f64, ncell);
    const lorentz = try a.alloc(f64, ncell);
    const vx = try a.alloc(f64, ncell);
    const vy = try a.alloc(f64, ncell);
    const vz = try a.alloc(f64, ncell);

    // MAGNFIELD
    const bsq = try a.alloc(f64, ncell);
    const b1c = try a.alloc(f64, ncell);
    const b2c = try a.alloc(f64, ncell);
    const b3c = try a.alloc(f64, ncell);
    const beta = try a.alloc(f64, ncell);
    const betainv = try a.alloc(f64, ncell);
    const sigma = try a.alloc(f64, ncell);
    const divb = try a.alloc(f64, ncell);
    const bx = try a.alloc(f64, ncell);
    const by = try a.alloc(f64, ncell);
    const bz = try a.alloc(f64, ncell);

    // RADIATION
    const ehat = try a.alloc(f64, ncell);
    const erad = try a.alloc(f64, ncell);
    const trad = try a.alloc(f64, ncell);
    const fx = try a.alloc(f64, ncell);
    const fy = try a.alloc(f64, ncell);
    const fz = try a.alloc(f64, ncell);

    // PRINTKAPPASTOSILO — the opacity channels (puffy silo.c:1116).
    const kgasabs = try a.alloc(f64, ncell);
    const kradabs = try a.alloc(f64, ncell);
    const kgasnum = try a.alloc(f64, ncell);
    const kradnum = try a.alloc(f64, ncell);
    const kgasross = try a.alloc(f64, ncell);
    const kradross = try a.alloc(f64, ncell);
    const totemiss = try a.alloc(f64, ncell);
    const kes = try a.alloc(f64, ncell);

    // PRINT_FIXUPS_TO_SILO — per-cell fixup flags (puffy silo.c:581).
    const fixups_mhd = try a.alloc(f64, ncell);
    const fixups_rad = try a.alloc(f64, ncell);
    const fixups_imp = try a.alloc(f64, ncell);
    const fixups_all = try a.alloc(f64, ncell);

    var iz: i64 = 0;
    while (iz < nz) : (iz += 1) {
        var iy: i64 = 0;
        while (iy < ny) : (iy += 1) {
            var ix: i64 = 0;
            while (ix < nx) : (ix += 1) {
                const zi: usize = @intCast(iz * (ny * nx) + iy * nx + ix);

                var pp: [SimT.nv]f64 = undefined;
                sim.p.load(ix, iy, iz, &pp);
                const geom = sim.cache.fillGeometry(ix, iy, iz);
                const geomBL = precompute.geometryBLat(&sim.grid, coords, mp, ix, iy, iz);
                const r = if (spherical) geomBL.xxvec[1] else 0;
                const th = if (spherical) geomBL.xxvec[2] else 0;
                const ph = if (spherical) geomBL.xxvec[3] else 0;

                // primitives → OUTCOORDS (BL)
                pp = try frames.transPallCoco(cfg, pp, &geom, &geomBL, mp);

                rho[zi] = pp[L.index(.rho)];
                uint[zi] = pp[L.index(.uu)];
                entr[zi] = pp[L.index(.entr)];
                gammag[zi] = gam;
                const pgas = (gam - 1.0) * uint[zi];
                temp[zi] = if (consts_ptr) |c| thermo.tFromUrho(c, uint[zi], rho[zi], gam) else 0;

                // 4-velocity in BL
                const vcon = [4]f64{ 0, pp[L.index(.vx)], pp[L.index(.vy)], pp[L.index(.vz)] };
                const ucon = try relele.convVels(vcon, .velr, .vel4, &geomBL.gg, &geomBL.GG);
                const ucov = relele.indices21(ucon, &geomBL.gg);
                uc0[zi] = @abs(ucon[0]);
                uc1[zi] = ucon[1];
                uc2[zi] = ucon[2];
                uc3[zi] = ucon[3];
                lorentz[zi] = @abs(ucon[0]) / @sqrt(@abs(geomBL.GG[0][0]));

                // Cartesian velocity (silo.c: v_θ*=r, v_φ*=r sinθ, then rotate)
                if (spherical) {
                    const cv = sphToCart(th, ph, ucon[1], ucon[2] * r, ucon[3] * r * @sin(th));
                    vx[zi] = cv[0];
                    vy[zi] = cv[1];
                    vz[zi] = cv[2];
                } else {
                    vx[zi] = ucon[1];
                    vy[zi] = ucon[2];
                    vz[zi] = ucon[3];
                }

                if (has_b) {
                    const bb = bfield.bconBcovBsqFrom4vel(
                        .{ pp[L.index(.b1)], pp[L.index(.b2)], pp[L.index(.b3)] },
                        ucon,
                        ucov,
                        &geomBL.gg,
                    );
                    bsq[zi] = bb.bsq;
                    b1c[zi] = pp[L.index(.b1)];
                    b2c[zi] = pp[L.index(.b2)];
                    b3c[zi] = pp[L.index(.b3)];
                    const premag = 0.5 * bb.bsq;
                    beta[zi] = pgas / premag;
                    betainv[zi] = premag / pgas;
                    sigma[zi] = bb.bsq / rho[zi];
                    // divB is left-biased: undefined on the low faces (silo.c).
                    divb[zi] = if (ix == 0 or (ny > 1 and iy == 0) or (nz > 1 and iz == 0))
                        0
                    else
                        ct.calcDivB(SimT, sim, ix, iy, iz);

                    if (spherical) {
                        const cb = sphToCart(th, ph, bb.bcon[1], bb.bcon[2] * r, bb.bcon[3] * r * @sin(th));
                        bx[zi] = cb[0];
                        by[zi] = cb[1];
                        bz[zi] = cb[2];
                    } else {
                        bx[zi] = bb.bcon[1];
                        by[zi] = bb.bcon[2];
                        bz[zi] = bb.bcon[3];
                    }
                }

                if (has_rad) {
                    const rtt = (try radiation.calcFfRtt(cfg, pp, &geomBL)).rtt;
                    ehat[zi] = -rtt;
                    const rij = try radiation.calcRij(cfg, pp, &geomBL); // R^{μν} (BL)
                    erad[zi] = rij[0][0];
                    trad[zi] = if (consts_ptr) |c| c.lteTfromE(ehat[zi]) else 0;
                    // Cartesian radiative flux (silo.c: F_θ*=√g_θθ, F_φ*=√g_φφ)
                    if (spherical) {
                        const cf = sphToCart(
                            th,
                            ph,
                            rij[1][0],
                            rij[2][0] * @sqrt(geomBL.gg[2][2]),
                            rij[3][0] * @sqrt(geomBL.gg[3][3]),
                        );
                        fx[zi] = cf[0];
                        fy[zi] = cf[1];
                        fz[zi] = cf[2];
                    } else {
                        fx[zi] = rij[1][0];
                        fy[zi] = rij[2][0];
                        fz[zi] = rij[3][0];
                    }

                    // PRINTKAPPASTOSILO: opacity channels from calc_kappa's
                    // filled opac struct + the standalone calc_kappaes
                    // (Trad = Te). Zero when no opacity model is wired.
                    if (par_ptr) |par| {
                        const st = try radforce.fillRadState(cfg, pp, &geomBL, gam, par);
                        kgasabs[zi] = st.opac.gas_abs;
                        kradabs[zi] = st.opac.rad_abs;
                        kgasnum[zi] = st.opac.gas_num;
                        kradnum[zi] = st.opac.rad_num;
                        kgasross[zi] = st.opac.gas_ross;
                        kradross[zi] = st.opac.rad_ross;
                        totemiss[zi] = st.opac.tot_emissivity;
                        kes[zi] = radforce.calcKappaes(cfg, pp, gam, par);
                    } else {
                        kgasabs[zi] = 0;
                        kradabs[zi] = 0;
                        kgasnum[zi] = 0;
                        kradnum[zi] = 0;
                        kgasross[zi] = 0;
                        kradross[zi] = 0;
                        totemiss[zi] = 0;
                        kes[zi] = 0;
                    }
                }

                // PRINT_FIXUPS_TO_SILO: the three fixup flags + a count of how
                // many fired at this cell. (C's `fixups++` increments the
                // *pointer* — a bug; the intent is the per-cell count.)
                const fmhd = sim.getFlag(.hd_fixup, ix, iy, iz);
                const frad = sim.getFlag(.rad_fixup, ix, iy, iz);
                const fimp = sim.getFlag(.radimp_fixup, ix, iy, iz);
                fixups_mhd[zi] = @floatFromInt(fmhd);
                fixups_rad[zi] = @floatFromInt(frad);
                fixups_imp[zi] = @floatFromInt(fimp);
                var nfix: f64 = 0;
                if (fmhd != 0) nfix += 1;
                if (frad != 0) nfix += 1;
                if (fimp != 0) nfix += 1;
                fixups_all[zi] = nfix;
            }
        }
    }

    // ---- write the Silo file ----------------------------------------------
    const file = DBCreateReal(path.ptr, DB_CLOBBER, DB_LOCAL, null, DB_PDB) orelse
        return error.SiloCreateFailed;
    defer _ = DBClose(file);

    // time / cycle metadata for VisIt's time slider
    var dtime: f64 = time;
    var ftime: f32 = @floatCast(time);
    var ncycle: c_int = cycle;
    const optlist = DBMakeOptlist(3) orelse return error.SiloCreateFailed;
    defer _ = DBFreeOptlist(optlist);
    _ = DBAddOption(optlist, DBOPT_DTIME, &dtime);
    _ = DBAddOption(optlist, DBOPT_TIME, &ftime);
    _ = DBAddOption(optlist, DBOPT_CYCLE, &ncycle);

    // node + zone dimension arrays (silo.c switches x/y for the ny==1 2D case;
    // PUFFY is ny>1 so the natural order holds)
    var ndims_node = [3]c_int{ @intCast(nnx), @intCast(nny), @intCast(nnz) };
    var zdims = [3]c_int{ @intCast(nx), @intCast(ny), @intCast(nz) };
    if (nz == 1 and ny == 1) {
        // 1D
    } else if (nz == 1) {
        // 2D (x, y|z): dims already {nx, ny}
    } else if (ny == 1) {
        ndims_node = .{ @intCast(nnx), @intCast(nnz), 1 };
        zdims = .{ @intCast(nx), @intCast(nz), 1 };
    }

    const coordnames = [3][*:0]const u8{ "X", "Y", "Z" };
    // For a 2D meridional (XZ) run the mesh's 2nd axis is z (silo.c:
    // coordinates[1] = nodez).
    const y_or_z: [*]const f64 = if (xz) nodez.ptr else nodey.ptr;
    const coord_ptrs = [3][*]const f64{
        nodex.ptr,
        if (ny == 1) nodez.ptr else y_or_z,
        nodez.ptr,
    };
    if (DBPutQuadmesh(file, "mesh1", &coordnames, &coord_ptrs, &ndims_node, ndim, DB_DOUBLE, DB_NONCOLLINEAR, optlist) != 0)
        return error.SiloWriteFailed;

    const putScalar = struct {
        fn f(fl: *DBfile, name: [*:0]const u8, data: []const f64, d: *const [3]c_int, nd: c_int, ol: *DBoptlist) !void {
            if (DBPutQuadvar1(fl, name, "mesh1", data.ptr, d, nd, null, 0, DB_DOUBLE, DB_ZONECENT, ol) != 0)
                return error.SiloWriteFailed;
        }
    }.f;

    try putScalar(file, "rho", rho, &zdims, ndim, optlist);
    try putScalar(file, "uint", uint, &zdims, ndim, optlist);
    try putScalar(file, "entr", entr, &zdims, ndim, optlist);
    try putScalar(file, "temp", temp, &zdims, ndim, optlist);
    try putScalar(file, "gammagas", gammag, &zdims, ndim, optlist);
    try putScalar(file, "u0", uc0, &zdims, ndim, optlist);
    try putScalar(file, "u1", uc1, &zdims, ndim, optlist);
    try putScalar(file, "u2", uc2, &zdims, ndim, optlist);
    try putScalar(file, "u3", uc3, &zdims, ndim, optlist);
    try putScalar(file, "lorentz", lorentz, &zdims, ndim, optlist);

    if (has_b) {
        try putScalar(file, "bsq", bsq, &zdims, ndim, optlist);
        try putScalar(file, "B1", b1c, &zdims, ndim, optlist);
        try putScalar(file, "B2", b2c, &zdims, ndim, optlist);
        try putScalar(file, "B3", b3c, &zdims, ndim, optlist);
        try putScalar(file, "beta", beta, &zdims, ndim, optlist);
        try putScalar(file, "betainv", betainv, &zdims, ndim, optlist);
        try putScalar(file, "sigma", sigma, &zdims, ndim, optlist);
        try putScalar(file, "divB", divb, &zdims, ndim, optlist);
    }
    if (has_rad) {
        try putScalar(file, "ehat", ehat, &zdims, ndim, optlist);
        try putScalar(file, "erad", erad, &zdims, ndim, optlist);
        try putScalar(file, "trad", trad, &zdims, ndim, optlist);

        // PRINTKAPPASTOSILO
        try putScalar(file, "kappa_gas_abs", kgasabs, &zdims, ndim, optlist);
        try putScalar(file, "kappa_rad_abs", kradabs, &zdims, ndim, optlist);
        try putScalar(file, "kappa_gas_num", kgasnum, &zdims, ndim, optlist);
        try putScalar(file, "kappa_rad_num", kradnum, &zdims, ndim, optlist);
        try putScalar(file, "kappa_gas_ross", kgasross, &zdims, ndim, optlist);
        try putScalar(file, "kappa_rad_ross", kradross, &zdims, ndim, optlist);
        try putScalar(file, "tot_emissivity", totemiss, &zdims, ndim, optlist);
        try putScalar(file, "kappa_es", kes, &zdims, ndim, optlist);
    }

    // PRINT_FIXUPS_TO_SILO (independent of RADIATION in puffy silo.c)
    try putScalar(file, "fixups_u2pmhd", fixups_mhd, &zdims, ndim, optlist);
    try putScalar(file, "fixups_u2prad", fixups_rad, &zdims, ndim, optlist);
    try putScalar(file, "fixups_radimp", fixups_imp, &zdims, ndim, optlist);
    try putScalar(file, "fixups", fixups_all, &zdims, ndim, optlist);

    const putVec = struct {
        fn f(fl: *DBfile, name: [*:0]const u8, n0: [*:0]const u8, n1: [*:0]const u8, n2: [*:0]const u8, c0: []const f64, c1: []const f64, c2: []const f64, d: *const [3]c_int, nd: c_int, ol: *DBoptlist) !void {
            const vnames = [3][*:0]const u8{ n0, n1, n2 };
            const vptrs = [3]*const anyopaque{ c0.ptr, c1.ptr, c2.ptr };
            if (DBPutQuadvar(fl, name, "mesh1", 3, &vnames, &vptrs, d, nd, null, 0, DB_DOUBLE, DB_ZONECENT, ol) != 0)
                return error.SiloWriteFailed;
        }
    }.f;

    // silo.c SILO2D_XZPLANE: the in-plane 2nd component is the physical z-part
    // so VisIt vector glyphs lie in the meridional plane.
    const v_mid = if (xz) vz else vy;
    try putVec(file, "velocity", "vel1", "vel2", "vel3", vx, v_mid, vz, &zdims, ndim, optlist);
    if (has_b) {
        const b_mid = if (xz) bz else by;
        try putVec(file, "magn_field", "B1", "B2", "B3", bx, b_mid, bz, &zdims, ndim, optlist);
    }
    if (has_rad) {
        const f_mid = if (xz) fz else fy;
        try putVec(file, "rad_flux", "F1", "F2", "F3", fx, f_mid, fz, &zdims, ndim, optlist);
    }
}

test "sphToCart projects a radial vector onto the equatorial x-axis" {
    // r̂ at (θ=π/2, φ=0) points along +x.
    const c = sphToCart(std.math.pi / 2.0, 0.0, 1.0, 0.0, 0.0);
    try std.testing.expectApproxEqAbs(@as(f64, 1), c[0], 1e-15);
    try std.testing.expectApproxEqAbs(@as(f64, 0), c[1], 1e-15);
    try std.testing.expectApproxEqAbs(@as(f64, 0), c[2], 1e-15);
    // θ̂ at (θ=π/2, φ=0) points along −z.
    const t = sphToCart(std.math.pi / 2.0, 0.0, 0.0, 1.0, 0.0);
    try std.testing.expectApproxEqAbs(@as(f64, -1), t[2], 1e-15);
}
