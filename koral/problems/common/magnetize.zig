//! Global β normalization of the seed field (C: postinit.c, BETANORMFULL):
//! fac = √(MAXBETA / max_cells(p_mag/p_tot)), then B *= fac on the domain
//! with a p2u refresh. Ghost cells keep their unscaled B (C does not
//! refresh BCs after postinit). Generic over SimT; every torus problem's
//! postinit is this call with its own MAXBETA.
//!
//! The max is band-parallel over iy through ChunkResult's max-merged slot,
//! so combining bands is order-insensitive and bit-identical to the serial
//! scan; it is then folded across ranks (MPI plan §1.1-5: without the fold
//! each rank normalizes B differently and the field is discontinuous from
//! step 0; max is exact, so the folded value equals a serial run's).
//!
//! JETCOORDS' MAXBETA_SEPARATE (a per-rank ratio folded by MAX) is a
//! second variant of this pass; it is not implemented here yet and must
//! be transcribed from C when that problem lands, not guessed.
//!
//! Moved verbatim out of problems/puffy/puffy.zig (redesign step 5,
//! 2026-09-04); expression shapes are unchanged.

const relele = @import("../../relele.zig");
const mhd = @import("../../physics/bfield.zig");
const radiation = @import("../../physics/radiation.zig");
const p2u_mod = @import("../../p2u.zig");
const threading = @import("../../threading.zig");

/// Returns fac (postinit.c:78). No-op (fac = 1) without a magnetic field.
pub fn normalizeBeta(comptime SimT: type, sim: *SimT, maxbeta: f64) !f64 {
    const L = SimT.Layout;
    if (comptime !L.hasVar(.b1)) return 1.0;

    const ny: i64 = @intCast(sim.grid.ny);

    // Pass 1 — the BETANORMFULL max, band-parallel over iy. `tsd_max` is
    // ChunkResult's max-merged slot, so combining bands is order-insensitive
    // and therefore bit-identical to the serial scan; clamping at 0 below
    // reproduces the serial `maxb = 0.0` starting value exactly (the slot's
    // neutral element is -inf).
    const MaxW = struct {
        fn rows(s: *SimT, iy0: i64, iy1: i64, res: *threading.ChunkResult) void {
            betaMaxRows(SimT, s, iy0, iy1, res) catch |e| {
                res.err = e;
            };
        }
    };
    const mres = threading.parallelRange(SimT, sim, sim.team, 0, ny, MaxW.rows);
    if (mres.err) |e| return e;
    var maxb: f64 = @max(0.0, mres.tsd_max);

    // BETANORMFULL is a GLOBAL max (MPI plan §1.1-5): without the fold each
    // rank normalizes B differently and the field is discontinuous from
    // step 0. Identity serially; max is exact, so the folded value is
    // bit-identical to a serial run's.
    maxb = sim.globalMax(maxb);
    const fac = @sqrt(maxbeta / maxb);

    // Pass 2 — apply the factor. Per-cell writes only; band-parallel.
    const ScaleCtx = struct { sim: *SimT, fac: f64 };
    var sctx = ScaleCtx{ .sim = sim, .fac = fac };
    const ScaleW = struct {
        fn rows(c: *ScaleCtx, iy0: i64, iy1: i64, res: *threading.ChunkResult) void {
            betaScaleRows(SimT, c.sim, c.fac, iy0, iy1) catch |e| {
                res.err = e;
            };
        }
    };
    const sres = threading.parallelRange(ScaleCtx, &sctx, sim.team, 0, ny, ScaleW.rows);
    if (sres.err) |e| return e;

    return fac;
}

/// postinit pass 1 body for iy ∈ [iy0, iy1): the per-cell β = p_mag/p_tot,
/// accumulated into the chunk's max-merged slot.
fn betaMaxRows(comptime SimT: type, sim: *SimT, iy0: i64, iy1: i64, res: *threading.ChunkResult) !void {
    const cfg = SimT.Cfg;
    const L = SimT.Layout;
    const nx: i64 = @intCast(sim.grid.nx);
    const nz: i64 = @intCast(sim.grid.nz);

    var iz: i64 = 0;
    while (iz < nz) : (iz += 1) {
        var iy: i64 = iy0;
        while (iy < iy1) : (iy += 1) {
            var ix: i64 = 0;
            while (ix < nx) : (ix += 1) {
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

                const pgas = (sim.opt.gam - 1.0) * pp[L.index(.uu)];
                var ptot = pgas;
                if (comptime cfg.has(.radiation)) {
                    const rt = try radiation.calcFfRtt(cfg, pp, &geom);
                    const ehat = -rt.rtt;
                    ptot += ehat / 3.0;
                }

                // BETANORMFULL: max over the whole domain
                if (pmag / ptot > res.tsd_max) res.tsd_max = pmag / ptot;
            }
        }
    }
}

/// postinit pass 2 body for iy ∈ [iy0, iy1): scale B by `fac` and refresh
/// the conserveds.
fn betaScaleRows(comptime SimT: type, sim: *SimT, fac: f64, iy0: i64, iy1: i64) !void {
    const cfg = SimT.Cfg;
    const L = SimT.Layout;
    const nx: i64 = @intCast(sim.grid.nx);
    const nz: i64 = @intCast(sim.grid.nz);

    var iz: i64 = 0;
    while (iz < nz) : (iz += 1) {
        var iy: i64 = iy0;
        while (iy < iy1) : (iy += 1) {
            var ix: i64 = 0;
            while (ix < nx) : (ix += 1) {
                var pp: [SimT.nv]f64 = undefined;
                sim.p.load(ix, iy, iz, &pp);
                pp[L.index(.b1)] *= fac;
                pp[L.index(.b2)] *= fac;
                pp[L.index(.b3)] *= fac;
                const geom = sim.cache.fillGeometry(ix, iy, iz);
                const uu = try p2u_mod.p2u(cfg, pp, &geom, sim.opt.gam);
                sim.p.store(ix, iy, iz, &pp);
                sim.u.store(ix, iy, iz, &uu);
            }
        }
    }
}
