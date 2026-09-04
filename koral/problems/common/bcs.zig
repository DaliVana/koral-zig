//! Stock boundary-condition fragments from KORAL's per-problem bc.c files,
//! generic over CoreT. A problem's `SpecificBc` callback is a switch on the
//! face that returns one of these (or its own arm):
//!
//!   outflowRescaleXhi  bc.c:23-121   XBCHI: copy the last cell, rescale
//!                                    ρ,u,B,v by (r_bound/r_ghost)^n in BL,
//!                                    then kill inflow (u^r ≥ 0) for gas
//!                                    and radiation.
//!   copyXlo            bc.c:126      XBCLO: plain copy of cell 0 (the inner
//!                                    edge is inside the horizon, so C's
//!                                    inflow check bc.c:133 never runs;
//!                                    COPY_XBC problems use it on both faces).
//!   polarReflect       bc.c:150-190  YBC: reflect across the axis with the
//!                                    VY / B2 / FY sign flip.
//!
//! Moved verbatim out of problems/puffy/puffy.zig (redesign step 5,
//! 2026-09-04); expression shapes are unchanged.

const relele = @import("../../relele.zig");
const frames = @import("../../frames.zig");
const precompute = @import("../../metric/precompute.zig");
const BcFace = @import("../../sim/bc.zig").BcFace;

/// XBCHI outflow with r-rescaling and no-inflow (bc.c:23-121). Reads the
/// last radial domain cell of the same (iy, iz) column.
pub fn outflowRescaleXhi(comptime CoreT: type, core: *const CoreT, ix: i64, iy: i64, iz: i64) relele.Error![CoreT.nv]f64 {
    const cfg = CoreT.Cfg;
    const L = CoreT.Layout;
    const NV = CoreT.nv;
    const has_rad = comptime cfg.has(.radiation);
    const has_b = comptime L.hasVar(.b1);
    const nx: i64 = @intCast(core.grid.nx);
    var pp: [NV]f64 = undefined;

    core.p.load(nx - 1, iy, iz, &pp);

    const geom = core.cache.fillGeometry(ix, iy, iz);
    const geomBL = precompute.geometryBLat(&core.grid, cfg.coords, core.phys.mp, ix, iy, iz);

    // MHD prims to BL (ghost-cell geometries, as in C)
    pp = try frames.transPmhdCoco(cfg, pp, &geom, &geomBL, core.phys.mp);

    const geombdBL = precompute.geometryBLat(&core.grid, cfg.coords, core.phys.mp, nx - 1, iy, iz);
    const rghost = geomBL.xxvec[1];
    const rbound = geombdBL.xxvec[1];
    const scale1 = rbound * rbound / rghost / rghost;
    const scale2 = rbound / rghost;

    pp[L.index(.rho)] *= scale1;
    pp[L.index(.uu)] *= scale1;
    if (comptime has_b) {
        pp[L.index(.b1)] *= scale1;
        pp[L.index(.b2)] *= scale2;
        pp[L.index(.b3)] *= scale2;
    }
    pp[L.index(.vy)] *= scale1;
    pp[L.index(.vz)] *= scale1;
    if (comptime has_rad) {
        // note: rad prims scaled while the MHD block sits in
        // BL — they never left MYCOORDS (C does the same)
        pp[L.index(.ee)] *= scale1;
        pp[L.index(.fy)] *= scale1;
        pp[L.index(.fz)] *= scale1;
    }

    pp = try frames.transPmhdCoco(cfg, pp, &geomBL, &geom, core.phys.mp);

    // no-inflow: gas
    var ucon = [4]f64{ 0.0, pp[L.index(.vx)], pp[L.index(.vy)], pp[L.index(.vz)] };
    ucon = try relele.convert(ucon, .velr, .vel4, &geom, .recompute_ut);
    ucon = frames.trans2Coco(geom.xxvec, ucon, cfg.coords, .bl, core.phys.mp);
    if (ucon[1] < 0.0) {
        ucon[1] = 0.0;
        ucon = frames.trans2Coco(geomBL.xxvec, ucon, .bl, cfg.coords, core.phys.mp);
        ucon = try relele.convert(ucon, .vel4, .velr, &geom, .recompute_ut);
        pp[L.index(.vx)] = ucon[1];
        pp[L.index(.vy)] = ucon[2];
        pp[L.index(.vz)] = ucon[3];
    }
    if (comptime has_rad) {
        var urf = [4]f64{ 0.0, pp[L.index(.fx)], pp[L.index(.fy)], pp[L.index(.fz)] };
        urf = try relele.convert(urf, .velr, .vel4, &geom, .recompute_ut);
        urf = frames.trans2Coco(geom.xxvec, urf, cfg.coords, .bl, core.phys.mp);
        if (urf[1] < 0.0) {
            urf[1] = 0.0;
            urf = frames.trans2Coco(geomBL.xxvec, urf, .bl, cfg.coords, core.phys.mp);
            urf = try relele.convert(urf, .vel4, .velr, &geom, .recompute_ut);
            pp[L.index(.fx)] = urf[1];
            pp[L.index(.fy)] = urf[2];
            pp[L.index(.fz)] = urf[3];
        }
    }
    return pp;
}

/// XBCLO plain copy of the first radial domain cell (bc.c:126).
pub fn copyXlo(comptime CoreT: type, core: *const CoreT, iy: i64, iz: i64) [CoreT.nv]f64 {
    var pp: [CoreT.nv]f64 = undefined;
    core.p.load(0, iy, iz, &pp);
    return pp;
}

/// YBC polar reflection (bc.c:150-190): mirror the θ index across the axis
/// and flip the θ components of v, B and F. `face` must be .ylo or .yhi.
///
/// "Axis" here is the first grid face (PUFFY: MINY = 0.001), never θ = 0
/// itself, where g^φφ diverges. Physics anchor: the 2D Bondi gates in
/// tests/ks_evolution_tests.zig run through this reflection and the polar
/// band on a KS grid (stationary, θ-uniform to roundoff).
pub fn polarReflect(comptime CoreT: type, core: *const CoreT, ix: i64, iy: i64, iz: i64, face: BcFace) [CoreT.nv]f64 {
    const cfg = CoreT.Cfg;
    const L = CoreT.Layout;
    const has_rad = comptime cfg.has(.radiation);
    const has_b = comptime L.hasVar(.b1);
    const ny: i64 = @intCast(core.grid.ny);
    var pp: [CoreT.nv]f64 = undefined;
    const iiy: i64 = switch (face) {
        .ylo => -iy - 1, // upper axis
        .yhi => ny - (iy - ny) - 1, // lower axis
        else => unreachable,
    };
    core.p.load(ix, iiy, iz, &pp);
    pp[L.index(.vy)] = -pp[L.index(.vy)];
    if (comptime has_b) pp[L.index(.b2)] = -pp[L.index(.b2)];
    if (comptime has_rad) pp[L.index(.fy)] = -pp[L.index(.fy)];
    return pp;
}
