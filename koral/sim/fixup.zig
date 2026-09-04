//! Cell fixups (C: cell_fixup, finite.c:5012), generic over CoreT: average
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
const std = @import("std");
const field_mod = @import("../field.zig");
const Grid = @import("../grid.zig").Grid;

const Error = relele.Error || error{OutOfMemory};
const Flag = storage.Flag;

/// The whole-grid u/p backups cell_fixup averages through (finite.c:5030),
/// owned by the fixup pass (redesign step 3). Lives at `sim.bak`.
pub fn Backups(comptime NV: usize) type {
    return struct {
        const Self = @This();
        pub const FieldT = field_mod.Field(NV);
        u: FieldT,
        p: FieldT,

        pub fn init(allocator: std.mem.Allocator, g: Grid, team: ?*threading.Team) !Self {
            var u = try FieldT.initUninitialized(allocator, g);
            errdefer u.deinit();
            threading.parallelZero(team, u.data);
            const p = try FieldT.initUninitialized(allocator, g);
            threading.parallelZero(team, p.data);
            return .{ .u = u, .p = p };
        }

        pub fn deinit(self: *Self) void {
            self.u.deinit();
            self.p.deinit();
            self.* = undefined;
        }
    };
}

/// True if any domain cell carries flag `which`. Early-out scan for
/// cellFixup: reads one i32/cell (getFlag is inline) and returns on the
/// first hit. Serial: the scan is ~NV× cheaper than the copies it may
/// save, and short-circuits (P2 #3).
fn anyFlagSet(comptime CoreT: type, self: *const CoreT, comptime which: Flag) bool {
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
pub fn cellFixup(comptime CoreT: type, self: *CoreT, bak: *Backups(CoreT.nv), comptime which: Flag) Error!void {
    // Common case: no cell carries `which` this pass. Then fixupRows
    // writes nothing (every cell hits `getFlag == 0 → continue`), so the
    // four full-grid copies below (u↔u_bak, p↔p_bak — ~4×NV×cells×8 B,
    // ~350 MB/step at production size across the ~6 hd + radimp calls)
    // are pure churn. Scan the flag column first — one i32/cell, ~100×
    // less traffic than the copies — and skip the whole staging dance.
    // Bit-identical: with nothing flagged, u/p are unchanged either way
    // (P2 #3).
    if (!anyFlagSet(CoreT, self, which)) return;

    threading.parallelCopy(self.team, bak.u.data, self.u.data);
    threading.parallelCopy(self.team, bak.p.data, self.p.data);

    // the averaging loop is parallel-safe as-is: flags are frozen
    // during the pass, reads touch only non-flagged neighbours of
    // the (frozen) p, writes only the flagged cells' _bak slots
    var ctx = Ctx(CoreT){ .core = self, .bak = bak };
    try threading.parallelRangeErr(Ctx(CoreT), &ctx, self.team, 0, self.nyi(), rowsFn(CoreT, which));

    threading.parallelCopy(self.team, self.u.data, bak.u.data);
    threading.parallelCopy(self.team, self.p.data, bak.p.data);
}

/// The pass's view: the Core plus its own backups.
fn Ctx(comptime CoreT: type) type {
    return struct { core: *CoreT, bak: *Backups(CoreT.nv) };
}

fn rowsFn(comptime CoreT: type, comptime which: Flag) fn (*Ctx(CoreT), i64, i64) Error!void {
    return struct {
        fn w(c: *Ctx(CoreT), iy0: i64, iy1: i64) Error!void {
            return fixupRows(CoreT, c, which, iy0, iy1);
        }
    }.w;
}

fn fixupRows(comptime CoreT: type, c: *Ctx(CoreT), comptime which: Flag, iy0: i64, iy1: i64) Error!void {
    const self = c.core;
    const bak = c.bak;
    const cfg = CoreT.Cfg;
    const L = CoreT.Layout;
    const NV = CoreT.nv;
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
                const uu = try p2u_mod.p2u(cfg, pp, &geom, self.phys.gam);
                bak.u.store(ix, iy, iz, &uu);
                bak.p.store(ix, iy, iz, &pp);
            }
        }
    }
}
