//! The per-cell conserved→primitive inversion sweep (C: calc_u2p,
//! finite.c:546), generic over CoreT. `Sim.calcU2p` is the operator: it runs
//! this pass on the team, then the fixups and a boundary refresh.
//!
//! Moved verbatim out of sim.zig (redesign step 2, 2026-09-04).

const relele = @import("../relele.zig");
const invert = @import("../solve/invert.zig");
const invert_rad = @import("../solve/invert_rad.zig");

const Error = relele.Error || error{OutOfMemory};

/// The band worker `Sim.calcU2p` hands to parallelRangeErr.
pub fn rowsFn(comptime CoreT: type) fn (*CoreT, i64, i64) Error!void {
    return struct {
        fn w(s: *CoreT, iy0: i64, iy1: i64) Error!void {
            return u2pRows(CoreT, s, iy0, iy1);
        }
    }.w;
}

/// The per-cell inversion body for iy ∈ [iy0, iy1) (all iz, all ix).
fn u2pRows(comptime CoreT: type, self: *CoreT, iy0: i64, iy1: i64) Error!void {
    const cfg = CoreT.Cfg;
    const L = CoreT.Layout;
    const NV = CoreT.nv;
    var iz: i64 = 0;
    while (iz < self.nzi()) : (iz += 1) {
        var iy: i64 = iy0;
        while (iy < iy1) : (iy += 1) {
            // Row-invariant: hoisted out of the ix loop.
            const polar = self.isCellCorrectedPolaraxis(iy);
            var ix: i64 = 0;
            while (ix < self.nxi()) : (ix += 1) {
                var uu: [NV]f64 = undefined;
                var pp: [NV]f64 = undefined;
                self.u.load(ix, iy, iz, &uu);
                self.p.load(ix, iy, iz, &pp);

                self.setFlag(.entropy, ix, iy, iz, 0);
                self.setFlag(.entropy2, ix, iy, iz, 0);

                const geom = self.cache.fillGeometry(ix, iy, iz);

                if (polar) {
                    // C: u2p_solver_Bonly (u2p.c:57) — invert only B
                    // (the rest is overwritten by do_correct); both
                    // floor checks skipped (u2p.c:80-92); corrected/
                    // fixups stay 0 so all flags read 0.
                    if (comptime L.hasVar(.b1)) {
                        const gdetu_inv = 1.0 / geom.gdet; // GDETIN == 1
                        pp[L.index(.b1)] = uu[L.index(.b1)] * gdetu_inv;
                        pp[L.index(.b2)] = uu[L.index(.b2)] * gdetu_inv;
                        pp[L.index(.b3)] = uu[L.index(.b3)] * gdetu_inv;
                    }
                    self.p.store(ix, iy, iz, &pp);
                    self.setFlag(.hd_fixup, ix, iy, iz, 0);
                    if (comptime L.hasVar(.ee)) {
                        self.setFlag(.rad_fixup, ix, iy, iz, 0);
                    }
                    continue;
                }

                const res = invert.u2pMhd(cfg, uu, &pp, &geom, self.phys.gam, self.phys.floors);
                if (res.entropy_used) self.setFlag(.entropy, ix, iy, iz, 1);

                // radiative closed-form inversion (u2p.c:386); runs
                // regardless of the MHD outcome, on the same uu
                var rad_fixup = false;
                if (comptime L.hasVar(.ee)) {
                    const rr = invert_rad.u2pRad(cfg, uu, &pp, &geom, self.phys.rad);
                    rad_fixup = rr.corrected;
                }

                _ = try invert.checkFloorsMhd(cfg, &pp, &geom, self.phys.gam, self.phys.floors);
                if (comptime L.hasVar(.ee)) {
                    _ = try invert_rad.checkFloorsRad(cfg, &pp, &geom, self.phys.rad);
                }

                self.p.store(ix, iy, iz, &pp);
                self.setFlag(.hd_fixup, ix, iy, iz, if (res.fixup) 1 else 0);
                if (comptime L.hasVar(.ee)) {
                    // C sets RADFIXUPFLAG to −1 (not 1) on request
                    self.setFlag(.rad_fixup, ix, iy, iz, if (rad_fixup) -1 else 0);
                }
            }
        }
    }
}
