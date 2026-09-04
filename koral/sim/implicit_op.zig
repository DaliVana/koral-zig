//! The implicit radiative source operator's per-cell pass (C: op_implicit,
//! finite.c:1400), generic over CoreT: solve_implicit_lab per cell,
//! RADIMPFIXUPFLAG, and the iteration/solve/failure tallies through
//! ChunkResult. `Sim.opImplicit` is the operator: the SKIPRADSOURCE gate,
//! the timer, the team dispatch, the counter merge and the radimp fixup.
//!
//! Moved verbatim out of sim.zig (redesign step 2, 2026-09-04).

const implicit = @import("../solve/implicit.zig");
const threading = @import("../threading.zig");

/// Worker context: the sub-step dt.
pub fn Ctx(comptime CoreT: type) type {
    return struct { core: *CoreT, dt: f64 };
}

/// The band worker `Sim.opImplicit` hands to parallelRange. Errors here
/// would be relele failures inside the solver, but solve_implicit_lab
/// returns a status (never throws), so this worker cannot error — it
/// only tallies counters into `res`.
pub fn rowsWorker(comptime CoreT: type) fn (*Ctx(CoreT), i64, i64, *threading.ChunkResult) void {
    return struct {
        fn w(ctx: *Ctx(CoreT), iy0: i64, iy1: i64, res: *threading.ChunkResult) void {
            const cfg = CoreT.Cfg;
            const NV = CoreT.nv;
            const self = ctx.core;
            const opac = &(self.phys.opac.?);
            const ImplT = implicit.Solver(cfg);

            var iz: i64 = 0;
            while (iz < self.nzi()) : (iz += 1) {
                var iy: i64 = iy0;
                while (iy < iy1) : (iy += 1) {
                    // C: finite.c:1427 — polar-corrected cells skip the
                    // implicit entirely (is_cell_active ≡ 1 and PUFFY defines
                    // no SKIPIMPLICIT_* hooks). Row-invariant: skip the row.
                    if (self.isCellCorrectedPolaraxis(iy)) continue;
                    var ix: i64 = 0;
                    while (ix < self.nxi()) : (ix += 1) {
                        var uu: [NV]f64 = undefined;
                        var pp: [NV]f64 = undefined;
                        self.u.load(ix, iy, iz, &uu);
                        self.p.load(ix, iy, iz, &pp);

                        const geom = self.cache.fillGeometry(ix, iy, iz);
                        const rr = ImplT.solveImplicitLab(&uu, &pp, &geom, ctx.dt, self.phys.gam, self.phys.rad, opac, &self.phys.implicit);

                        if (rr.ok) {
                            self.u.store(ix, iy, iz, &uu);
                            self.p.store(ix, iy, iz, &pp);
                            self.setFlag(.radimp_fixup, ix, iy, iz, 0);
                            res.n_iters += rr.iters;
                            res.n_solves += 1;
                        } else {
                            // C: unchanged u/p, flag for fixups
                            self.setFlag(.radimp_fixup, ix, iy, iz, -1);
                            res.n_fail += 1;
                        }
                    }
                }
            }
        }
    }.w;
}
