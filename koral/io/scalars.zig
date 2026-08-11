//! Diagnostic scalar reductions (C: postproc.c calc_scalars + its helpers) —
//! the physically central accretion/luminosity/geometry diagnostics for a
//! BH-disk run. These are the quantities written to `scalars.dat` and used to
//! judge a run's health (Ṁ tracking, luminosity, disk thickness, dynamo-held
//! magnetization).
//!
//! Only the reductions PUFFY's `calc_scalars` actually selects are ported:
//!   * totalMass       — Σ ρ √−g dV over the domain      (calc_totalmass)
//!   * mdot(shell)      — Σ √−g ρ uʳ dθ dφ through a radial shell (calc_mdot,
//!                        type 0 net; MYCOORDS, dφ→2π for TNZ==1)
//!   * lum(shell)       — radiative + total energy flux through a shell in the
//!                        OUTCOORDS (BL) frame                (calc_lum type 1)
//!   * scaleHeightAt    — density-weighted RMS |π/2−θ| at a radius
//!                        (calc_scaleheight → dynamo.calcScaleHeight)
//!   * maxPmagPtot      — max pmag/ptot over the domain (the BETANORMFULL
//!                        quantity, so "dynamo sustains pmag/ptot" is readable)
//!
//! MYCOORDS reductions (mass, mdot) use the sim's cached metric; the BL-frame
//! luminosity rebuilds the OUTCOORDS geometry on the fly via coco (C: OUTCOORDS
//! == BLCOORDS for every BHDISK problem). Everything is in code (GU) units —
//! the executable converts to CGS/Eddington only at print time, matching C's
//! fileop.c which keeps scalars[] in GU and scales in the fprintf.

const std = @import("std");
const config = @import("../config.zig");
const layout = @import("../layout.zig");
const geometry = @import("../geometry.zig");
const coco = @import("../metric/coco.zig");
const precompute = @import("../metric/precompute.zig");
const relele = @import("../relele.zig");
const hydro = @import("../physics/hydro.zig");
const radiation = @import("../physics/radiation.zig");
const frames = @import("../frames.zig");
const mhd = @import("../physics/bfield.zig");
const dynamo = @import("../sim/dynamo.zig");

const Geometry = geometry.Geometry;
const pi = std.math.pi;

/// BL (OUTCOORDS) geometry at a cell center, pulling the coordinate system /
/// metric params from the sim options so the reductions stay problem-agnostic.
/// The reduction itself lives once in precompute.geometryBLat.
fn blGeom(comptime SimT: type, sim: *const SimT, ix: i64, iy: i64, iz: i64) Geometry {
    return precompute.geometryBLat(&sim.grid, SimT.Cfg.coords, sim.opt.mp, ix, iy, iz);
}

/// r in OUTCOORDS at a radial index (θ,φ-independent for BL/MKS). MINK has no
/// BH frame, so the Cartesian x-coordinate stands in as the radius proxy.
pub fn radiusBL(comptime SimT: type, sim: *const SimT, ix: i64) f64 {
    if (comptime SimT.Cfg.coords == .mink) return sim.grid.xc(ix);
    const xx = [4]f64{ 0.0, sim.grid.xc(ix), sim.grid.yc(0), sim.grid.zc(0) };
    return coco.cocoN(xx, SimT.Cfg.coords, .bl, sim.opt.mp)[1];
}

/// First domain radial index whose BL radius exceeds `radius` (C's
/// `for(ix..) if(xxBL[1]>radius) break;`). Clamped to the last domain cell if
/// the whole grid is inside `radius` (C: calc_lum resets ix=NX-2; calc_mdot
/// would read NX — we clamp to nx-1, the outermost shell, which is what the
/// luminosity radius 5000 > RMAX means physically: the outer boundary).
pub fn radialShellIndex(comptime SimT: type, sim: *const SimT, radius: f64) i64 {
    const nx = sim.nxi();
    var ix: i64 = 0;
    while (ix < nx) : (ix += 1) {
        if (radiusBL(SimT, sim, ix) > radius) return ix;
    }
    return nx - 1;
}

/// C: calc_totalmass (postproc.c:1723) — Σ ρ dx dy dz √−g over the domain,
/// MYCOORDS. NOTE: unlike calc_mdot / calc_lum (which override dz → 2π for
/// TNZ==1), the *snapshot* branch of calc_totalmass uses the raw cell dz
/// (= the φ-wedge width, PHIWEDGE, for an axisymmetric slice) with only a
/// `dz *= 2π/PHIWEDGE` scaling when TNZ>1 (postproc.c:1767-1769) — a C
/// inconsistency, transcribed as-is. So for PUFFY 2D (TNZ==1) the wedge is
/// NOT expanded to 2π here; a 3D wedge run IS (so the reported mass is the
/// full-torus mass, not the π/2 slice).
pub fn totalMass(comptime SimT: type, sim: *const SimT) f64 {
    const L = SimT.Layout;
    const rho_i = comptime L.index(.rho);

    // 2π/PHIWEDGE wedge→full-torus factor for TNZ>1 (1.0 in 2D, where the
    // single z-cell already spans the whole wedge and C does not expand it).
    const phifac: f64 = if (sim.nzi() > 1) 2.0 * pi / (sim.grid.maxz - sim.grid.minz) else 1.0;

    var mass: f64 = 0;
    var iz: i64 = 0;
    while (iz < sim.nzi()) : (iz += 1) {
        var iy: i64 = 0;
        while (iy < sim.nyi()) : (iy += 1) {
            var ix: i64 = 0;
            while (ix < sim.nxi()) : (ix += 1) {
                const geom = sim.cache.fillGeometry(ix, iy, iz);
                const dx = sim.grid.cellSize(ix, 0);
                const dy = sim.grid.cellSize(iy, 1);
                const dz = sim.grid.cellSize(iz, 2) * phifac;
                mass += sim.p.get(rho_i, ix, iy, iz) * dx * dy * dz * geom.gdet;
            }
        }
    }
    return mass;
}

/// C: calc_mdot (postproc.c:2737), type 0 (net) through the radial shell at
/// index `ix`: Σ_{iy,iz} √−g ρ uʳ dθ dφ (MYCOORDS; dφ → 2π for TNZ==1). The
/// caller negates for the conventional Ṁ>0 accretion sign (scalars[1]=−mdot).
pub fn mdot(comptime SimT: type, sim: *const SimT, ix: i64) relele.Error!f64 {
    const L = SimT.Layout;
    const rho_i = comptime L.index(.rho);
    const twod = sim.nzi() == 1;

    var acc: f64 = 0;
    var iz: i64 = 0;
    while (iz < sim.nzi()) : (iz += 1) {
        var iy: i64 = 0;
        while (iy < sim.nyi()) : (iy += 1) {
            var pp: [SimT.nv]f64 = undefined;
            sim.p.load(ix, iy, iz, &pp);
            const geom = sim.cache.fillGeometry(ix, iy, iz);
            const vcon = [4]f64{ 0, pp[L.index(.vx)], pp[L.index(.vy)], pp[L.index(.vz)] };
            const ucon = try relele.convert(vcon, .velr, .vel4, &geom, .recompute_ut);
            const rhouconr = pp[rho_i] * ucon[1];
            const dy = sim.grid.cellSize(iy, 1);
            // C calc_mdot: dφ → 2π for TNZ==1, raw cell dφ × (2π/PHIWEDGE) for
            // TNZ>1 (postproc.c:2786-2807) — both give the full-torus flux.
            const dz = if (twod) 2.0 * pi else sim.grid.cellSize(iz, 2) * (2.0 * pi / (sim.grid.maxz - sim.grid.minz));
            acc += geom.gdet * rhouconr * dy * dz;
        }
    }
    return acc;
}

pub const Lum = struct { radlum: f64, totallum: f64 };

/// C: calc_lum (postproc.c:1905), type 1 (positive Rʳ_t everywhere), the 2D
/// snapshot branch, through the radial shell at index `ix`. Prims are boosted
/// to OUTCOORDS (BL) and the radiative + total (ρuʳ+Tʳ_t+Rʳ_t) energy fluxes
/// are summed with the BL cell-θ width and dφ → 2π (TNZ==1).
pub fn lum(comptime SimT: type, sim: *const SimT, ix: i64) relele.Error!Lum {
    const cfg = SimT.Cfg;
    const L = SimT.Layout;
    const twod = sim.nzi() == 1;
    const has_rad = comptime L.hasVar(.ee);

    var radlum: f64 = 0;
    var totallum: f64 = 0;
    var iz: i64 = 0;
    while (iz < sim.nzi()) : (iz += 1) {
        var iy: i64 = 0;
        while (iy < sim.nyi()) : (iy += 1) {
            var pp: [SimT.nv]f64 = undefined;
            sim.p.load(ix, iy, iz, &pp);
            const geom = sim.cache.fillGeometry(ix, iy, iz);
            const geomBL = blGeom(SimT, sim, ix, iy, iz);

            // to BL (both MHD and radiation blocks; C: trans_pall_coco)
            pp = try frames.transPallCoco(cfg, pp, &geom, &geomBL, sim.opt.mp);

            // BL cell θ-width (C: get_cellsize_out non-precompute path — the
            // difference of OUTCOORDS θ across the y-faces at this x-center)
            const dthBL = blThetaWidth(SimT, sim, ix, iy, iz);
            // C calc_lum: dφ_BL → 2π for TNZ==1, BL cell dφ × (2π/PHIWEDGE)
            // for TNZ>1 — the full-torus luminosity, not the π/2 wedge's.
            const dphBL = if (twod) 2.0 * pi else blPhiWidth(SimT, sim, ix, iy, iz) * (2.0 * pi / (sim.grid.maxz - sim.grid.minz));

            const vcon = [4]f64{ 0, pp[L.index(.vx)], pp[L.index(.vy)], pp[L.index(.vz)] };
            const ucon = try relele.convert(vcon, .velr, .vel4, &geomBL, .recompute_ut);
            const rhour = pp[L.index(.rho)] * ucon[1];

            const tij_up = try hydro.calcTij(cfg, pp, &geomBL, sim.opt.gam);
            const tij = relele.lowerSecond(tij_up, &geomBL);
            const trt = tij[1][0];

            var rrt: f64 = 0;
            if (has_rad) {
                const rij_up = try radiation.calcRij(cfg, pp, &geomBL);
                const rij = relele.lowerSecond(rij_up, &geomBL);
                rrt = -rij[1][0]; // type 1
                if (rrt < 0) rrt = 0;
                radlum += geomBL.gdet * rrt * dthBL * dphBL;
            } else {
                radlum += geomBL.gdet * (pp[L.index(.uu)] * ucon[1]) * dthBL * dphBL;
            }
            totallum += geomBL.gdet * (rhour + trt + rrt) * dthBL * dphBL;
        }
    }
    return .{ .radlum = radlum, .totallum = totallum };
}

/// OUTCOORDS θ across the two y-faces of cell (ix,iy) at its x-center
/// (C: get_cellsize_out dx[1], non-precompute).
fn blThetaWidth(comptime SimT: type, sim: *const SimT, ix: i64, iy: i64, iz: i64) f64 {
    const xc = sim.grid.xc(ix);
    const zc = sim.grid.zc(iz);
    const a = coco.cocoN(.{ 0, xc, sim.grid.yl(iy), zc }, SimT.Cfg.coords, .bl, sim.opt.mp)[2];
    const b = coco.cocoN(.{ 0, xc, sim.grid.yl(iy + 1), zc }, SimT.Cfg.coords, .bl, sim.opt.mp)[2];
    return @abs(b - a);
}

/// OUTCOORDS φ across the two z-faces of cell (ix,iy,iz) (unused for TNZ==1).
fn blPhiWidth(comptime SimT: type, sim: *const SimT, ix: i64, iy: i64, iz: i64) f64 {
    const xc = sim.grid.xc(ix);
    const yc = sim.grid.yc(iy);
    const a = coco.cocoN(.{ 0, xc, yc, sim.grid.zl(iz) }, SimT.Cfg.coords, .bl, sim.opt.mp)[3];
    const b = coco.cocoN(.{ 0, xc, yc, sim.grid.zl(iz + 1) }, SimT.Cfg.coords, .bl, sim.opt.mp)[3];
    return @abs(b - a);
}

/// C: calc_scaleheight (postproc.c:2645) — the density-weighted RMS |π/2−θ|
/// at the radial shell nearest `radius`. Fills sim.scaleth first (C's
/// scaleth_otg is populated by calc_avgs_throughout during the step).
pub fn scaleHeightAt(comptime SimT: type, sim: *const SimT, radius: f64) f64 {
    // Read-only: query the one radial column directly instead of filling the
    // whole sim.scaleth (that fill is dynamo-owned state; a diagnostic must
    // not mutate the Sim). Bit-identical — same per-ix body as calcScaleHeight.
    const ix = radialShellIndex(SimT, sim, radius);
    return dynamo.scaleHeightAtIx(SimT, sim, ix);
}

pub const GlobalScalars = struct {
    mass: f64,
    /// net flux through the shell, NOT sign-flipped (the caller negates for
    /// the Ṁ>0 accretion convention, exactly as with mdot()).
    mdot: f64,
    radlum: f64,
    totallum: f64,
    scaleheight: f64,
    max_pmag_ptot: f64,
};

/// All the calc_scalars reductions with the MPI folds applied (plan §8.3):
/// each rank sums its slab, then ONE Allreduce(SUM) of [mass, mdot, radlum,
/// totallum, Σρ√g, Σρ√gΔθ²] ++ `extra_sums` (folded in place — the driver's
/// diagnostic counters ride along, so an output cadence costs exactly two
/// collectives) and one Allreduce(MAX) for pmag/ptot. Serially / at 1 rank
/// the folds are identity, so the values are BITWISE the serial ones (the
/// local loops and the scale-height finalize are byte-for-byte the serial
/// path). r and θ are never decomposed, so the shell indices agree across
/// ranks and every rank returns the same globals.
pub fn globalScalars(
    comptime SimT: type,
    sim: *const SimT,
    ix_mdot: i64,
    ix_lum: i64,
    r_scale: f64,
    extra_sums: []f64,
) relele.Error!GlobalScalars {
    const mass = totalMass(SimT, sim);
    const md = try mdot(SimT, sim, ix_mdot);
    const l = try lum(SimT, sim, ix_lum);
    const ix_s = radialShellIndex(SimT, sim, r_scale);
    const parts = dynamo.scaleHeightPartsAtIx(SimT, sim, ix_s);
    var maxb = [1]f64{try maxPmagPtot(SimT, sim)};

    var sums = [6]f64{ mass, md, l.radlum, l.totallum, parts.sig, parts.sth };
    if (sim.opt.comm) |c| {
        // Two fixed collectives, same order on every rank (the extras are
        // carried inside the first so the count never varies by caller).
        var buf: [16]f64 = undefined;
        std.debug.assert(sums.len + extra_sums.len <= buf.len);
        @memcpy(buf[0..sums.len], &sums);
        @memcpy(buf[sums.len..][0..extra_sums.len], extra_sums);
        c.allreduceSum(buf[0 .. sums.len + extra_sums.len]);
        @memcpy(&sums, buf[0..sums.len]);
        @memcpy(extra_sums, buf[sums.len..][0..extra_sums.len]);
        c.allreduceMax(maxb[0..]);
    }
    return .{
        .mass = sums[0],
        .mdot = sums[1],
        .radlum = sums[2],
        .totallum = sums[3],
        .scaleheight = dynamo.scaleHeightFinalize(ix_s, sums[4], sums[5]),
        .max_pmag_ptot = maxb[0],
    };
}

/// max pmag/ptot over the domain — the BETANORMFULL quantity. Reports
/// whether the (initially 1/20) pmag/ptot ratio is being held; 0 if no B field.
pub fn maxPmagPtot(comptime SimT: type, sim: *const SimT) relele.Error!f64 {
    const cfg = SimT.Cfg;
    const L = SimT.Layout;
    if (comptime !L.hasVar(.b1)) return 0;

    var maxb: f64 = 0;
    var iz: i64 = 0;
    while (iz < sim.nzi()) : (iz += 1) {
        var iy: i64 = 0;
        while (iy < sim.nyi()) : (iy += 1) {
            var ix: i64 = 0;
            while (ix < sim.nxi()) : (ix += 1) {
                var pp: [SimT.nv]f64 = undefined;
                sim.p.load(ix, iy, iz, &pp);
                const geom = sim.cache.fillGeometry(ix, iy, iz);
                const ug = try relele.uconUcovFromPrims(
                    .{ pp[L.index(.vx)], pp[L.index(.vy)], pp[L.index(.vz)] },
                    &geom,
                );
                const bb = mhd.bconBcovBsqFrom4vel(
                    .{ pp[L.index(.b1)], pp[L.index(.b2)], pp[L.index(.b3)] },
                    ug.con,
                    ug.cov,
                    &geom,
                );
                const pmag = bb.bsq / 2.0;
                var ptot = (sim.opt.gam - 1.0) * pp[L.index(.uu)];
                if (comptime cfg.has(.radiation)) {
                    const rt = try radiation.calcFfRtt(cfg, pp, &geom);
                    ptot += (-rt.rtt) / 3.0;
                }
                if (pmag / ptot > maxb) maxb = pmag / ptot;
            }
        }
    }
    return maxb;
}
