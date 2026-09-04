//! CORRECT_POLARAXIS: the polar-axis special treatment (PUFFY define.h:107-109,
//! NCCORRECTPOLAR = 2). The NCCORRECTPOLAR most-polar θ rows on each side are
//! not evolved: they are overwritten every step from the first row below the
//! band.
//!
//!   correct_polaraxis            finite.c:5525; the overwrite (this file)
//!   is_cell_corrected_polaraxis  finite.c:6132; the predicate the evolution
//!                                  passes gate on: u2p inverts B only
//!                                  (u2p.c:57/80/92, both floor checks
//!                                  skipped), cell_fixup never targets these
//!                                  cells (finite.c:5042), op_implicit skips
//!                                  them entirely (finite.c:1427)
//!
//! Those two are one contract, not two features: a row the predicate claims
//! MUST be a row the overwrite rewrites, or the evolution passes skip work that
//! nothing then supplies; p decouples from u, and since the B-only u2p branch
//! deliberately zeroes the fixup flags, nothing reports it. `Band` is the
//! single source of truth both halves derive from, and it carries BOTH
//! activation gates (the `correct_polaraxis` option and the coordinate system)
//! so they cannot drift apart. `Sim.init` also rejects the two configs
//! that would make the contract unsatisfiable: `ny ≤ 2·nccorrectpolar` (the
//! overwrite's sources would lie inside the overwritten band) and non-spherical
//! coords (preconditions (5) and (7)).
//!
//! `sim.zig` keeps the two entry points; `Sim.doCorrect` and
//! `Sim.isCellCorrectedPolaraxis`; as thin delegates, the same way it fronts
//! sim/bc.zig with `Sim.setBc`.
//!
//! Physics anchor: tests/ks_evolution_tests.zig drives the 2D Bondi/Michel
//! flow through the band and the stock polar reflection on a full-θ KS grid;
//! the drift converges at 2nd order and the profile stays θ-uniform to
//! roundoff. The grid must not put a face on the axis itself (g^φφ = 1/sin²θ
//! diverges there): PUFFY's MINY = 0.001 is that offset, and the band's
//! `thaxis` is the first face, not θ = 0.
//!
//! Not yet transcribed, but lands here rather than in sim/ct.zig when it does:
//! adjust_fluxcttoth_emfs (polar-axis EMF zeroing), which is CORRECT_POLARAXIS
//! machinery that happens to live in magn.c.

const threading = @import("../threading.zig");
const p2u_mod = @import("../p2u.zig");

const Error = @import("../relele.zig").Error || error{OutOfMemory};

/// The θ rows the axis treatment owns. Construct only via `band()`; a `Band`
/// existing means the treatment is active and both halves agree on its extent.
pub const Band = struct {
    /// C: NCCORRECTPOLAR
    nc: i64,
    /// domain θ extent (`Sim.nyi()`)
    ny: i64,

    /// C: is_cell_corrected_polaraxis (finite.c:6132). No HALFTHETA, and the
    /// φ-only MPI decomposition leaves θ undivided, so every rank owns both
    /// sides of the axis.
    pub fn owns(b: Band, iy: i64) bool {
        return iy < b.nc or iy > b.ny - b.nc - 1;
    }

    /// The row `iy` is overwritten from (C: finite.c:5525 loop bounds). Never
    /// itself owned; `Sim.init` precondition (5) rejects `ny ≤ 2·nc`, so the
    /// overwrite never reads a cell it also writes, at any ordering.
    pub fn sourceRow(b: Band, iy: i64) i64 {
        return if (iy < b.nc) b.nc else b.ny - 1 - b.nc;
    }

    /// θ-face index this row's side collapses onto: the `thaxis` of the scale
    /// factor (`Grid.yl(0)` for the upper axis, `Grid.yl(ny)` for the lower).
    pub fn axisFace(b: Band, iy: i64) i64 {
        return if (iy < b.nc) 0 else b.ny;
    }
};

/// The band this sim's axis treatment owns, or null when it is inactive.
/// `.mink` is the only non-spherical `Coords` (same test as io/silo.zig).
pub fn band(comptime CoreT: type, sim: *const CoreT) ?Band {
    if (comptime CoreT.Cfg.coords == .mink) return null;
    const pa = sim.num.polaraxis orelse return null;
    return .{ .nc = pa.ncells, .ny = sim.nyi() };
}

/// C: correct_polaraxis (finite.c:5525). Scalars and in-row velocities copied
/// from the source row, the θ-components scaled by |θ−θ_axis|/|θ_src−θ_axis|
/// (internal x2), then p2u at the *target* geometry rewrites all conserveds. B
/// is untouched: PUFFY defines no CORRECTMAGNFIELD.
///
/// Band-parallel over ix columns: each column reads only its own (unowned)
/// source rows and writes only its own owned rows, so banding is bit-identical
/// to serial. For the same reason the two sides can share one `ic` loop;
/// no write is ever read back, so their interleaving cannot change a value.
pub fn correct(comptime CoreT: type, sim: *CoreT) Error!void {
    const b = band(CoreT, sim) orelse return;
    var ctx = Ctx(CoreT){ .sim = sim, .band = b };
    return threading.parallelRangeErr(Ctx(CoreT), &ctx, sim.team, 0, sim.nxi(), colsFn(CoreT));
}

fn Ctx(comptime CoreT: type) type {
    return struct { sim: *CoreT, band: Band };
}

fn colsFn(comptime CoreT: type) fn (*Ctx(CoreT), i64, i64) Error!void {
    return struct {
        fn w(c: *Ctx(CoreT), ix0: i64, ix1: i64) Error!void {
            return colsBand(CoreT, c, ix0, ix1);
        }
    }.w;
}

fn colsBand(comptime CoreT: type, c: *Ctx(CoreT), ix0: i64, ix1: i64) Error!void {
    const b = c.band;
    var iz: i64 = 0;
    while (iz < c.sim.nzi()) : (iz += 1) {
        var ix: i64 = ix0;
        while (ix < ix1) : (ix += 1) {
            var ic: i64 = 0;
            while (ic < b.nc) : (ic += 1) {
                try overwriteCell(CoreT, c.sim, b, ix, ic, iz); // upper axis
                try overwriteCell(CoreT, c.sim, b, ix, b.ny - 1 - ic, iz); // lower
            }
        }
    }
}

fn overwriteCell(comptime CoreT: type, sim: *CoreT, b: Band, ix: i64, iy: i64, iz: i64) Error!void {
    const L = CoreT.Layout;
    const g = &sim.grid;

    const iysrc = b.sourceRow(iy);
    const thaxis = g.yl(b.axisFace(iy));
    const fac = @abs((g.yc(iy) - thaxis) / (g.yc(iysrc) - thaxis));

    var pp: [CoreT.nv]f64 = undefined;
    var pps: [CoreT.nv]f64 = undefined;
    sim.p.load(ix, iy, iz, &pp);
    sim.p.load(ix, iysrc, iz, &pps);

    pp[L.index(.rho)] = pps[L.index(.rho)];
    pp[L.index(.uu)] = pps[L.index(.uu)];
    pp[L.index(.entr)] = pps[L.index(.entr)];
    pp[L.index(.vx)] = pps[L.index(.vx)];
    pp[L.index(.vz)] = pps[L.index(.vz)];
    pp[L.index(.vy)] = fac * pps[L.index(.vy)];
    if (comptime L.hasVar(.ee)) {
        pp[L.index(.ee)] = pps[L.index(.ee)];
        pp[L.index(.fx)] = pps[L.index(.fx)];
        pp[L.index(.fz)] = pps[L.index(.fz)];
        pp[L.index(.fy)] = fac * pps[L.index(.fy)];
    }

    const geom = sim.cache.fillGeometry(ix, iy, iz);
    const uu = try p2u_mod.p2u(CoreT.Cfg, pp, &geom, sim.phys.gam);
    sim.p.store(ix, iy, iz, &pp);
    sim.u.store(ix, iy, iz, &uu);
}
