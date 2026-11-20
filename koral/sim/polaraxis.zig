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
//!   TRANSMITTING_YBC             the pole passes the flow to the far side
//!                                  instead of reflecting it; four pieces
//!                                  here and one in the comm layer:
//!                                  `flipAcrossPole` (signs of a state
//!                                  carried across), `Band.stencilRows`
//!                                  (finite.c:923-993), the E_r exemption in
//!                                  `zeroAxisEmfs` (magn.c:424), `correct_b`
//!                                  in the overwrite (CORRECTMAGNFIELD), and
//!                                  `Core.exchangePole` between ranks. The
//!                                  ghost fill itself is problems/common/
//!                                  bcs.zig `polarTransmit`.
//!
//! Those two are one contract, not two features: a row the predicate claims
//! MUST be a row the overwrite rewrites, or the evolution passes skip work that
//! nothing then supplies; p decouples from u, and since the B-only u2p branch
//! deliberately zeroes the fixup flags, nothing reports it. `Band` is the
//! single source of truth both halves derive from, and it carries BOTH
//! activation gates (the `num.polaraxis` option and the coordinate system)
//! so they cannot drift apart. `Sim.init` also rejects the two configs
//! that would make the contract unsatisfiable: `ny ≤ 2·ncells` (the
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
//!   adjust_fluxcttoth_emfs       magn.c:388; flux-CT's corner EMFs zeroed on
//!                                  the two polar faces (`zeroAxisEmfs`, called
//!                                  from sim/ct.zig). CORRECT_POLARAXIS
//!                                  machinery that happens to live in magn.c.
//!                                  Transcribed 2026-11-20; its absence was the
//!                                  funnel runaway of magnetised runs.

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
    /// C: TRANSMITTING_YBC (options.zig `PolarAxis.transmitting`)
    transmitting: bool = false,
    /// C: CORRECTMAGNFIELD
    correct_b: bool = false,

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

    /// The five theta rows the reconstruction at row `iy` reads, (iy-2 ..
    /// iy+2) unless the pole transmits (C: finite.c:923-993). Then the first
    /// row on each side, which the overwrite fills with a copy of its
    /// neighbour, is replaced in every stencil by the first ghost row, which
    /// holds the cell across the axis: the rows next to the pole reconstruct
    /// against the far side instead of against their own copy. Only the
    /// slot that would have read row 0 (row ny-1) changes; C writes the
    /// substitution for one corrected row whatever NCCORRECTPOLAR is, and so
    /// does this. Cell sizes are not substituted (C keeps get_size_x(iy±k)).
    pub fn stencilRows(b: Band, iy: i64) [5]i64 {
        var r = [5]i64{ iy - 2, iy - 1, iy, iy + 1, iy + 2 };
        if (!b.transmitting) return r;
        // upper axis: row 0 -> row -1
        if (iy == 0) r[2] = iy - 1;
        if (iy == 1) r[1] = iy - 2;
        if (iy == 2) r[0] = iy - 3;
        // lower axis: row ny-1 -> row ny
        if (iy == b.ny - 1) r[2] = iy + 1;
        if (iy == b.ny - 2) r[3] = iy + 2;
        if (iy == b.ny - 3) r[4] = iy + 3;
        return r;
    }
};

/// The band this sim's axis treatment owns, or null when it is inactive.
/// `.mink` is the only non-spherical `Coords` (same test as io/silo.zig).
pub fn band(comptime CoreT: type, sim: *const CoreT) ?Band {
    if (comptime CoreT.Cfg.coords == .mink) return null;
    const pa = sim.num.polaraxis orelse return null;
    return .{ .nc = pa.ncells, .ny = sim.nyi(), .transmitting = pa.transmitting, .correct_b = pa.correct_b };
}

/// C: adjust_fluxcttoth_emfs (magn.c:388), the CORRECT_POLARAXIS branch
/// without TRANSMITTING_YBC (no problem here transmits through the pole):
/// E_r and E_phi are zeroed on every corner of the two polar faces. `cts` is
/// flux-CT's scratch (sim/ct.zig `Scratch`); call between its EMF pass and
/// its flux rebuild.
///
/// The B^r flux through a polar face is the average of the two E_phi on its
/// corners, and the B^phi flux that of the two E_r. Zeroed, the first theta
/// row exchanges no magnetic flux with the axis, so the radial flux through
/// each spherical shell is conserved to roundoff, as it must be: the axis is
/// not a boundary (tests/polaraxis_tests.zig pins exactly that).
///
/// Left alone, nothing ties those EMFs to zero. The reflected ghost row is
/// the mirror image of the first row in the internal theta coordinate, but
/// the face it is mirrored about is not theta = 0 (MINY = 0.001), so on an
/// MKS2 grid the ghost sits at a different sin(theta) than its mirror cell
/// (0.096 against 0.135 for 120 cells, h0 = 0.9). The two x-face B^theta
/// fluxes that should cancel in the corner average then do not, and the
/// residual E_phi pumps radial flux through the axis wherever the funnel
/// carries a field. In PUFFY's first steps it does not, which is why the C
/// step goldens never saw the gap. In a magnetised funnel the leak seeds a
/// grid-scale checkerboard in B at the corner of the axis and the inner
/// boundary, B grows exponentially there, and the b^2 / rho floor follows it
/// up without bound (rho ~ 1e62 before NanInFlux in tools/mhdtorus.zig).
///
/// Serial: 2 (nx + 1) (nz + 1) corners.
pub fn zeroAxisEmfs(comptime CoreT: type, sim: *const CoreT, cts: anytype) void {
    const b = band(CoreT, sim) orelse return;
    const nx = sim.nxi();
    const nz = sim.nzi();
    var iz: i64 = 0;
    while (iz <= nz) : (iz += 1) {
        var ix: i64 = 0;
        while (ix <= nx) : (ix += 1) {
            for ([2]i64{ 0, b.ny }) |iy| {
                // a transmitting pole keeps E_r: flux may cross the axis in
                // phi, only E_phi is pinned (magn.c:424-451). Note that the
                // polar-face EMFs only ever feed B in the first row, which a
                // transmitting run overwrites every step (CORRECTMAGNFIELD
                // is forced on, Options.validate (7)): there the choice
                // cannot reach the evolved state, in C or here.
                if (!b.transmitting) cts.setEmf(1, ix, iy, iz, 0);
                cts.setEmf(3, ix, iy, iz, 0);
            }
        }
    }
}

/// The sign changes of a state carried across a transmitting pole: the
/// theta and phi components of v, B and the radiative flux (the why is at
/// problems/common/bcs.zig `polarTransmit`; C: mpi.c:1996). Shared by the
/// one-rank ghost fill and the cross-pole exchange unpack, so the two
/// cannot disagree.
pub fn flipAcrossPole(comptime CoreT: type, pp: *[CoreT.nv]f64) void {
    const L = CoreT.Layout;
    pp[L.index(.vy)] = -pp[L.index(.vy)];
    pp[L.index(.vz)] = -pp[L.index(.vz)];
    if (comptime L.hasVar(.b1)) {
        pp[L.index(.b2)] = -pp[L.index(.b2)];
        pp[L.index(.b3)] = -pp[L.index(.b3)];
    }
    if (comptime L.hasVar(.ee)) {
        pp[L.index(.fy)] = -pp[L.index(.fy)];
        pp[L.index(.fz)] = -pp[L.index(.fz)];
    }
}

/// C: correct_polaraxis (finite.c:5525). Scalars and in-row velocities copied
/// from the source row, the θ-components scaled by |θ−θ_axis|/|θ_src−θ_axis|
/// (internal x2), then p2u at the *target* geometry rewrites all conserveds. B
/// is untouched unless `correct_b` (C: CORRECTMAGNFIELD, which PUFFY does not
/// define and TRANSMITTING_YBC switches on): then B^r and B^phi are copied
/// and B^theta scaled too.
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
    if (comptime L.hasVar(.b1)) {
        // C: CORRECTMAGNFIELD (finite.c:5583) — "do overwrite magnetic
        // field, div B != 0 does not propagate in"
        if (b.correct_b) {
            pp[L.index(.b1)] = pps[L.index(.b1)];
            pp[L.index(.b3)] = pps[L.index(.b3)];
            pp[L.index(.b2)] = fac * pps[L.index(.b2)];
        }
    }
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
