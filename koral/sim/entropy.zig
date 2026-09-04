//! Entropy bookkeeping (u2p.c:2237 copy_entropycount, u2p.c:2257
//! update_entropy), generic over SimT. `Sim.updateEntropy` owns the timer
//! and the team dispatch of `rowsFn`.
//!
//! Moved verbatim out of sim.zig (redesign step 2, 2026-09-04).

const relele = @import("../relele.zig");
const hydro = @import("../physics/hydro.zig");
const p2u_mod = @import("../p2u.zig");

const Error = relele.Error || error{OutOfMemory};

/// C: copy_entropycount (u2p.c:2237).
pub fn copyEntropyCount(comptime SimT: type, self: *SimT) void {
    var iz: i64 = 0;
    while (iz < self.nzi()) : (iz += 1) {
        var iy: i64 = 0;
        while (iy < self.nyi()) : (iy += 1) {
            var ix: i64 = 0;
            while (ix < self.nxi()) : (ix += 1) {
                self.setFlag(.entropy3, ix, iy, iz, self.getFlag(.entropy, ix, iy, iz));
            }
        }
    }
}

/// The band worker `Sim.updateEntropy` hands to parallelRangeErr.
pub fn rowsFn(comptime SimT: type) fn (*SimT, i64, i64) Error!void {
    return struct {
        fn w(s: *SimT, iy0: i64, iy1: i64) Error!void {
            return entropyRows(SimT, s, iy0, iy1);
        }
    }.w;
}

/// C: update_entropy (u2p.c:2257). Recompute S(ρ,u) and refresh the
/// MHD conserveds. Cell-local.
fn entropyRows(comptime SimT: type, self: *SimT, iy0: i64, iy1: i64) Error!void {
    const cfg = SimT.Cfg;
    const L = SimT.Layout;
    const NV = SimT.nv;
    var iz: i64 = 0;
    while (iz < self.nzi()) : (iz += 1) {
        var iy: i64 = iy0;
        while (iy < iy1) : (iy += 1) {
            var ix: i64 = 0;
            while (ix < self.nxi()) : (ix += 1) {
                var pp: [NV]f64 = undefined;
                self.p.load(ix, iy, iz, &pp);
                pp[L.index(.entr)] = hydro.sFromU(pp[L.index(.rho)], pp[L.index(.uu)], self.opt.gam);
                self.p.set(L.index(.entr), ix, iy, iz, pp[L.index(.entr)]);

                var uu: [NV]f64 = undefined;
                self.u.load(ix, iy, iz, &uu);
                const geom = self.cache.fillGeometry(ix, iy, iz);
                try p2u_mod.p2uMhd(cfg, pp, &uu, &geom, self.opt.gam);
                self.u.store(ix, iy, iz, &uu);
            }
        }
    }
}
