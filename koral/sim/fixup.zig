//! Cell fixups (C: cell_fixup, finite.c:5012), generic over SimT: average
//! cells flagged by the inversion / implicit passes from their non-flagged
//! in-domain neighbours through the whole-grid u_bak/p_bak backups.
//! `Sim.cellFixup(which)` is the stable entry point (timers + the enable
//! switches); the staging and the averaging live here.
//!
//! Moved verbatim out of sim.zig (redesign step 2, 2026-09-04).

const relele = @import("../relele.zig");
const p2u_mod = @import("../p2u.zig");
const threading = @import("../threading.zig");
const storage = @import("storage.zig");

const Error = relele.Error || error{OutOfMemory};
const Flag = storage.Flag;

/// True if any domain cell carries flag `which`. Early-out scan for
/// cellFixup: reads one i32/cell (getFlag is inline) and returns on the
/// first hit. Serial: the scan is ~NV× cheaper than the copies it may
/// save, and short-circuits (P2 #3).
fn anyFlagSet(comptime SimT: type, self: *const SimT, comptime which: Flag) bool {
    const nx = self.nxi();
    const ny = self.nyi();
    const nz = self.nzi();
    var iz: i64 = 0;
    while (iz < nz) : (iz += 1) {
        var iy: i64 = 0;
        while (iy < ny) : (iy += 1) {
            var ix: i64 = 0;
            while (ix < nx) : (ix += 1) {
                if (self.getFlag(which, ix, iy, iz) != 0) return true;
            }
        }
    }
    return false;
}

/// C: cell_fixup (finite.c:5012), types FIXUP_U2PMHD / FIXUP_U2PRAD /
/// FIXUP_RADIMP. Averages flagged cells from their non-flagged in-domain
/// neighbors. U2PMHD never averages rho or the magnetic field (iv != RHO &&
/// iv < B1); U2PRAD averages only the radiative block (EE..FZ).
pub fn cellFixup(comptime SimT: type, self: *SimT, comptime which: Flag) Error!void {
    // Common case: no cell carries `which` this pass. Then fixupRows
    // writes nothing (every cell hits `getFlag == 0 → continue`), so the
    // four full-grid copies below (u↔u_bak, p↔p_bak — ~4×NV×cells×8 B,
    // ~350 MB/step at production size across the ~6 hd + radimp calls)
    // are pure churn. Scan the flag column first — one i32/cell, ~100×
    // less traffic than the copies — and skip the whole staging dance.
    // Bit-identical: with nothing flagged, u/p are unchanged either way
    // (P2 #3).
    if (!anyFlagSet(SimT, self, which)) return;

    threading.parallelCopy(self.team, self.u_bak.data, self.u.data);
    threading.parallelCopy(self.team, self.p_bak.data, self.p.data);

    // the averaging loop is parallel-safe as-is: flags are frozen
    // during the pass, reads touch only non-flagged neighbours of
    // the (frozen) p, writes only the flagged cells' _bak slots
    try threading.parallelRangeErr(SimT, self, self.team, 0, self.nyi(), rowsFn(SimT, which));

    threading.parallelCopy(self.team, self.u.data, self.u_bak.data);
    threading.parallelCopy(self.team, self.p.data, self.p_bak.data);
}

fn rowsFn(comptime SimT: type, comptime which: Flag) fn (*SimT, i64, i64) Error!void {
    return struct {
        fn w(s: *SimT, iy0: i64, iy1: i64) Error!void {
            return fixupRows(SimT, s, which, iy0, iy1);
        }
    }.w;
}

fn fixupRows(comptime SimT: type, self: *SimT, comptime which: Flag, iy0: i64, iy1: i64) Error!void {
    const cfg = SimT.Cfg;
    const L = SimT.Layout;
    const NV = SimT.nv;
    const b1_bound: usize = comptime if (L.hasVar(.b1)) L.index(.b1) else L.count;
    const nx = self.nxi();
    const ny = self.nyi();
    const nz = self.nzi();

    var iz: i64 = 0;
    while (iz < nz) : (iz += 1) {
        var iy: i64 = iy0;
        while (iy < iy1) : (iy += 1) {
            // C: "do not correct if overwritten later on" — row-
            // invariant, so skip the whole row rather than each cell.
            if (self.isCellCorrectedPolaraxis(iy)) continue;
            var ix: i64 = 0;
            while (ix < nx) : (ix += 1) {
                if (self.getFlag(which, ix, iy, iz) == 0) continue;

                var ppn: [6][NV]f64 = undefined;
                var in_n: usize = 0;
                // neighbor order matches C: x−, x+, y−, y+, z−, z+
                const nbrs = [6][3]i64{
                    .{ ix - 1, iy, iz }, .{ ix + 1, iy, iz },
                    .{ ix, iy - 1, iz }, .{ ix, iy + 1, iz },
                    .{ ix, iy, iz - 1 }, .{ ix, iy, iz + 1 },
                };
                for (nbrs) |nb| {
                    if (nb[0] < 0 or nb[0] >= nx or nb[1] < 0 or nb[1] >= ny or
                        nb[2] < 0 or nb[2] >= nz) continue;
                    if (self.getFlag(which, nb[0], nb[1], nb[2]) != 0) continue;
                    self.p.load(nb[0], nb[1], nb[2], &ppn[in_n]);
                    in_n += 1;
                }

                const enough = (nz == 1 and ny == 1 and in_n >= 1) or
                    (nz == 1 and in_n >= 2) or
                    (ny == 1 and in_n >= 2) or
                    in_n >= 3;
                if (!enough) continue; // C prints "didn't manage"

                var pp: [NV]f64 = undefined;
                self.p.load(ix, iy, iz, &pp);
                for (0..NV) |iv| {
                    // `which` is comptime — only the taken arm is analyzed
                    const fixthis = switch (which) {
                        .hd_fixup => iv != L.index(.rho) and iv < b1_bound,
                        .rad_fixup => iv >= L.index(.ee) and iv <= L.index(.fz),
                        // FIXUP_RADIMP: both fluids, never rho/B
                        .radimp_fixup => iv != L.index(.rho) and
                            (iv < b1_bound or (iv >= L.index(.ee) and iv <= L.index(.fz))),
                        else => unreachable,
                    };
                    if (fixthis) {
                        var s: f64 = 0;
                        for (0..in_n) |k| s += ppn[k][iv];
                        pp[iv] = s / @as(f64, @floatFromInt(in_n));
                    }
                }
                const geom = self.cache.fillGeometry(ix, iy, iz);
                const uu = try p2u_mod.p2u(cfg, pp, &geom, self.opt.gam);
                self.u_bak.store(ix, iy, iz, &uu);
                self.p_bak.store(ix, iy, iz, &pp);
            }
        }
    }
}
