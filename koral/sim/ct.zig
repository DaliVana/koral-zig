//! Constrained transport and vector-potential machinery (C: magn.c).
//!
//!   flux_ct          magn.c:240: Tóth flux-CT; corner EMFs from the B
//!                                  rows of flbx/flby/flbz, then the B fluxes
//!                                  rebuilt from EMF averages.
//!   calc_BfromA      magn.c:462: cell-centered A → corner A (linear
//!                                  interpolation) → curl → B, optionally
//!                                  overwriting p/u.
//!   calc_BfromA_core magn.c:568: the curl. 2D (TNZ==1) uses the exact
//!                                  flux-ct-compatible corner differences.
//!   calc_divB        magn.c:693: corner-based divergence diagnostic.
//!
//! All functions are generic over Sim(cfg) (passed as the type + pointer) so
//! this module doesn't circularly import sim.zig at comptime.
//!
//! C quirks kept:
//!  * fl_x/fl_y/fl_z pin an index to 0 when that dimension is collapsed.
//!  * coefemf is 0.5 unless *both* transverse dimensions are active (0.25).
//!  * calc_divB divides the x-differences by 2·(x(ix+1)−x(ix)); the
//!    *forward* center distance; while differencing (ix)−(ix−1).
//!  * adjust_fluxcttoth_emfs (polar-axis EMF zeroing) is CORRECT_POLARAXIS
//!    machinery and lands with M11; none of the M6 problems define it.

const std = @import("std");
const relele = @import("../relele.zig");
const p2u_mod = @import("../p2u.zig");
const threading = @import("../threading.zig");
const field_mod = @import("../field.zig");
const Grid = @import("../grid.zig").Grid;

/// The constrained-transport scratch a Sim owns when the config has MHD
/// (redesign step 3: pass-owned scratch): the corner EMFs of flux_ct and
/// the 6-slot vector-potential work field of calc_BfromA (also borrowed by
/// the dynamo's curl). Lives at `sim.ct`.
pub const Scratch = struct {
    allocator: std.mem.Allocator,
    nx: usize,
    ny: usize,
    nz: usize,
    /// corner EMFs for flux-CT (comps 1..3 at slots 0..2), (nx+1)(ny+1)(nz+1)
    emf: [3][]f64,
    /// vector-potential/B scratch for calc_BfromA (C: pvecpot — corner A
    /// lives in slots B1..B3, the curl result in slots 1..3, coexisting).
    /// Here: slots 0..2 = corner A_i, slots 3..5 = B^i.
    vecpot: field_mod.Field(6),

    pub fn init(allocator: std.mem.Allocator, g: Grid, team: ?*threading.Team) !Scratch {
        const ncorn = (g.nx + 1) * (g.ny + 1) * (g.nz + 1);
        var emf: [3][]f64 = undefined;
        var n_alloc: usize = 0;
        errdefer for (emf[0..n_alloc]) |e| allocator.free(e);
        for (0..3) |c| {
            emf[c] = try allocator.alloc(f64, ncorn);
            n_alloc += 1;
            @memset(emf[c], 0);
        }
        // Zeroed THROUGH THE TEAM (P4b NUMA first-touch); zeros are zeros.
        const vecpot = try field_mod.Field(6).initUninitialized(allocator, g);
        threading.parallelZero(team, vecpot.data);
        return .{ .allocator = allocator, .nx = g.nx, .ny = g.ny, .nz = g.nz, .emf = emf, .vecpot = vecpot };
    }

    pub fn deinit(self: *Scratch) void {
        for (0..3) |c| self.allocator.free(self.emf[c]);
        self.vecpot.deinit();
        self.* = undefined;
    }

    pub fn emfIdx(self: *const Scratch, ix: i64, iy: i64, iz: i64) usize {
        const jx: usize = @intCast(ix);
        const jy: usize = @intCast(iy);
        const jz: usize = @intCast(iz);
        std.debug.assert(jx <= self.nx and jy <= self.ny and jz <= self.nz);
        return (jz * (self.ny + 1) + jy) * (self.nx + 1) + jx;
    }
    pub fn getEmf(self: *const Scratch, comp: usize, ix: i64, iy: i64, iz: i64) f64 {
        return self.emf[comp - 1][self.emfIdx(ix, iy, iz)];
    }
    pub fn setEmf(self: *Scratch, comp: usize, ix: i64, iy: i64, iz: i64, v: f64) void {
        self.emf[comp - 1][self.emfIdx(ix, iy, iz)] = v;
    }
};

fn flPin(n: i64, i: i64) i64 {
    return if (n == 1) 0 else i;
}

/// C: flux_ct (magn.c:240). Requires GDETIN == 1 (ours always is).
/// P1: both passes run band-parallel over iy corner rows; pass 1 writes
/// only its own corners' EMFs, pass 2 only its own faces' flb rows; the
/// region boundary between them is the required RAW barrier (pass 2 reads
/// pass 1's EMFs at iy and iy+1).
pub fn fluxCt(comptime SimT: type, sim: *SimT) void {
    const L = SimT.Layout;
    if (comptime !L.hasVar(.b1)) return;

    const ny = sim.nyi();
    _ = threading.parallelRange(SimT, sim, sim.team, 0, ny + 1, emfBandWorker(SimT));

    // adjust_fluxcttoth_emfs: CORRECT_POLARAXIS only (M11)

    _ = threading.parallelRange(SimT, sim, sim.team, 0, ny + 1, rebuildBandWorker(SimT));
}

fn emfBandWorker(comptime SimT: type) fn (*SimT, i64, i64, *threading.ChunkResult) void {
    return struct {
        fn w(sim: *SimT, iy0: i64, iy1: i64, res: *threading.ChunkResult) void {
            _ = res;
            emfBand(SimT, sim, iy0, iy1);
        }
    }.w;
}

/// flux_ct pass 1 for corner rows iy ∈ [iy0, iy1).
fn emfBand(comptime SimT: type, sim: *SimT, iy0: i64, iy1: i64) void {
    const L = SimT.Layout;
    const b1 = comptime L.index(.b1);
    const b2 = comptime L.index(.b2);
    const b3 = comptime L.index(.b3);

    const nx = sim.nxi();
    const ny = sim.nyi();
    const nz = sim.nzi();

    var coefemf: [4]f64 = undefined;
    coefemf[1] = if (ny > 1 and nz > 1) 0.25 else 0.5;
    coefemf[2] = if (nx > 1 and nz > 1) 0.25 else 0.5;
    coefemf[3] = if (nx > 1 and ny > 1) 0.25 else 0.5;

    // EMFs on all domain corners (loop_4: 0..N inclusive per dimension)
    var iz: i64 = 0;
    while (iz <= nz) : (iz += 1) {
        var iy: i64 = iy0;
        while (iy < iy1) : (iy += 1) {
            var ix: i64 = 0;
            while (ix <= nx) : (ix += 1) {
                // EMF1
                if (ny > 1 or nz > 1) {
                    var s: f64 = 0;
                    if (ny > 1) {
                        s += sim.faces.flb[1].get(b3, flPin(nx, ix), flPin(ny, iy), flPin(nz, iz)) +
                            sim.faces.flb[1].get(b3, flPin(nx, ix), flPin(ny, iy), flPin(nz, iz - 1));
                    }
                    if (nz > 1) {
                        s -= sim.faces.flb[2].get(b2, flPin(nx, ix), flPin(ny, iy), flPin(nz, iz)) +
                            sim.faces.flb[2].get(b2, flPin(nx, ix), flPin(ny, iy - 1), flPin(nz, iz));
                    }
                    sim.ct.setEmf(1, ix, iy, iz, coefemf[1] * s);
                } else {
                    sim.ct.setEmf(1, ix, iy, iz, 0);
                }
                // EMF2
                if (nx > 1 or nz > 1) {
                    var s: f64 = 0;
                    if (nz > 1) {
                        s += sim.faces.flb[2].get(b1, flPin(nx, ix), flPin(ny, iy), flPin(nz, iz)) +
                            sim.faces.flb[2].get(b1, flPin(nx, ix - 1), flPin(ny, iy), flPin(nz, iz));
                    }
                    if (nx > 1) {
                        s -= sim.faces.flb[0].get(b3, ix, flPin(ny, iy), flPin(nz, iz)) +
                            sim.faces.flb[0].get(b3, ix, flPin(ny, iy), flPin(nz, iz - 1));
                    }
                    sim.ct.setEmf(2, ix, iy, iz, coefemf[2] * s);
                } else {
                    sim.ct.setEmf(2, ix, iy, iz, 0);
                }
                // EMF3
                if (nx > 1 or ny > 1) {
                    var s: f64 = 0;
                    if (nx > 1) {
                        s += sim.faces.flb[0].get(b2, ix, flPin(ny, iy), flPin(nz, iz)) +
                            sim.faces.flb[0].get(b2, ix, flPin(ny, iy - 1), flPin(nz, iz));
                    }
                    if (ny > 1) {
                        s -= sim.faces.flb[1].get(b1, flPin(nx, ix), flPin(ny, iy), flPin(nz, iz)) +
                            sim.faces.flb[1].get(b1, flPin(nx, ix - 1), flPin(ny, iy), flPin(nz, iz));
                    }
                    sim.ct.setEmf(3, ix, iy, iz, coefemf[3] * s);
                } else {
                    sim.ct.setEmf(3, ix, iy, iz, 0);
                }
            }
        }
    }
}

fn rebuildBandWorker(comptime SimT: type) fn (*SimT, i64, i64, *threading.ChunkResult) void {
    return struct {
        fn w(sim: *SimT, iy0: i64, iy1: i64, res: *threading.ChunkResult) void {
            _ = res;
            rebuildBand(SimT, sim, iy0, iy1);
        }
    }.w;
}

/// flux_ct pass 2; rebuild the B rows of the face fluxes from the corner
/// EMFs, for corner rows iy ∈ [iy0, iy1).
fn rebuildBand(comptime SimT: type, sim: *SimT, iy0: i64, iy1: i64) void {
    const L = SimT.Layout;
    const b1 = comptime L.index(.b1);
    const b2 = comptime L.index(.b2);
    const b3 = comptime L.index(.b3);

    const nx = sim.nxi();
    const ny = sim.nyi();
    const nz = sim.nzi();

    var iz: i64 = 0;
    while (iz <= nz) : (iz += 1) {
        var iy: i64 = iy0;
        while (iy < iy1) : (iy += 1) {
            var ix: i64 = 0;
            while (ix <= nx) : (ix += 1) {
                if (nx > 1 and iy < ny and iz < nz) {
                    sim.faces.flb[0].set(b1, ix, iy, iz, 0);
                    sim.faces.flb[0].set(b2, ix, iy, iz, 0.5 * (sim.ct.getEmf(3, ix, iy, iz) + sim.ct.getEmf(3, ix, iy + 1, iz)));
                    sim.faces.flb[0].set(b3, ix, iy, iz, -0.5 * (sim.ct.getEmf(2, ix, iy, iz) + sim.ct.getEmf(2, ix, iy, iz + 1)));
                }
                if (ny > 1 and ix < nx and iz < nz) {
                    sim.faces.flb[1].set(b1, ix, iy, iz, -0.5 * (sim.ct.getEmf(3, ix, iy, iz) + sim.ct.getEmf(3, ix + 1, iy, iz)));
                    sim.faces.flb[1].set(b2, ix, iy, iz, 0);
                    sim.faces.flb[1].set(b3, ix, iy, iz, 0.5 * (sim.ct.getEmf(1, ix, iy, iz) + sim.ct.getEmf(1, ix, iy, iz + 1)));
                }
                if (nz > 1 and ix < nx and iy < ny) {
                    sim.faces.flb[2].set(b1, ix, iy, iz, 0.5 * (sim.ct.getEmf(2, ix, iy, iz) + sim.ct.getEmf(2, ix + 1, iy, iz)));
                    sim.faces.flb[2].set(b2, ix, iy, iz, -0.5 * (sim.ct.getEmf(1, ix, iy, iz) + sim.ct.getEmf(1, ix, iy + 1, iz)));
                    sim.faces.flb[2].set(b3, ix, iy, iz, 0);
                }
            }
        }
    }
}

/// C: calc_BfromA (magn.c:462) with the default linear-interpolation branch.
/// Reads cell-centered A_i from the B1..B3 slots of `sim.p`, averages to
/// corners, takes the curl, and (ifoverwrite) writes B into p/u everywhere;
/// ghost cells get the stale scratch (zeros) exactly like C, so call setBc
/// afterwards, as ko.c does.
pub fn calcBfromA(comptime SimT: type, sim: *SimT, ifoverwrite: bool) relele.Error!void {
    const L = SimT.Layout;
    if (comptime !L.hasVar(.b1)) return;
    const b1 = comptime L.index(.b1);

    const nx = sim.nxi();
    const ny = sim.nyi();
    const nz = sim.nzi();

    cornerAverageA(SimT, sim, &sim.ct.vecpot, &sim.p, b1);
    calcBfromACore(SimT, sim, &sim.ct.vecpot);

    if (ifoverwrite) {
        const ngx: i64 = @intCast(sim.grid.ngx);
        const ngy: i64 = @intCast(sim.grid.ngy);
        const ngz: i64 = @intCast(sim.grid.ngz);
        var iz: i64 = -ngz;
        while (iz < nz + ngz) : (iz += 1) {
            var iy: i64 = -ngy;
            while (iy < ny + ngy) : (iy += 1) {
                var ix: i64 = -ngx;
                while (ix < nx + ngx) : (ix += 1) {
                    var pp: [SimT.nv]f64 = undefined;
                    sim.p.load(ix, iy, iz, &pp);
                    pp[b1] = sim.ct.vecpot.get(3, ix, iy, iz);
                    pp[b1 + 1] = sim.ct.vecpot.get(4, ix, iy, iz);
                    pp[b1 + 2] = sim.ct.vecpot.get(5, ix, iy, iz);
                    const geom = sim.cache.fillGeometry(ix, iy, iz);
                    const uu = try p2u_mod.p2u(SimT.Cfg, pp, &geom, sim.opt.gam);
                    sim.p.store(ix, iy, iz, &pp);
                    sim.u.store(ix, iy, iz, &uu);
                }
            }
        }
    }
}

/// Corner-averaged vector potential (magn.c:499-515): A_i at each corner is
/// the average of the surrounding cell-centered A_i, read from `src` slots
/// [base, base+1, base+2] and written into `scratch` slots 0..2. `src` is any
/// Field on the same grid; sim.p (B slots) for calc_BfromA, the dynamo scratch
/// for mimic_dynamo; `scratch` is the 6-slot work Field (`&sim.ct.vecpot`) that
/// carries the corner A (0..2) then the curl B (3..5). Exposed via pub so the
/// dynamo can reuse the curl.
pub fn cornerAverageA(comptime SimT: type, sim: *SimT, scratch: anytype, src: anytype, base: usize) void {
    const nx = sim.nxi();
    const ny = sim.nyi();
    const nz = sim.nzi();

    var iz: i64 = 0;
    while (iz <= nz) : (iz += 1) {
        var iy: i64 = 0;
        while (iy <= ny) : (iy += 1) {
            var ix: i64 = 0;
            while (ix <= nx) : (ix += 1) {
                if (nz == 1 and iz > 0) continue;
                if (ny == 1 and iy > 0) continue;

                for (0..3) |iv| {
                    const c = base + iv;
                    var a_val: f64 = undefined;
                    if (ny == 1 and nz == 1) {
                        a_val = 1.0 / 2.0 * (src.get(c, ix, iy, iz) + src.get(c, ix - 1, iy, iz));
                    } else if (ny > 1 and nz == 1) {
                        a_val = 1.0 / 4.0 * (src.get(c, ix, iy, iz) + src.get(c, ix, iy - 1, iz) +
                            src.get(c, ix - 1, iy, iz) + src.get(c, ix - 1, iy - 1, iz));
                    } else if (nz > 1 and ny == 1) {
                        a_val = 1.0 / 4.0 * (src.get(c, ix, iy, iz) + src.get(c, ix, iy, iz - 1) +
                            src.get(c, ix - 1, iy, iz) + src.get(c, ix - 1, iy, iz - 1));
                    } else {
                        a_val = 1.0 / 8.0 * (src.get(c, ix, iy, iz) + src.get(c, ix, iy - 1, iz) +
                            src.get(c, ix - 1, iy, iz) + src.get(c, ix - 1, iy - 1, iz) +
                            src.get(c, ix, iy, iz - 1) + src.get(c, ix, iy - 1, iz - 1) +
                            src.get(c, ix - 1, iy, iz - 1) + src.get(c, ix - 1, iy - 1, iz - 1));
                    }
                    scratch.set(iv, ix, iy, iz, a_val);
                }
            }
        }
    }
}

/// Curl of a cell-centered vector potential held in `src` slots
/// [base..base+2]: corner-average then curl, leaving B^i in `scratch` slots
/// 3..5 (C: calc_BfromA(pinput, 0); the ifoverwrite==0 path the dynamo uses).
/// Returns `scratch` so the consumer visibly reads the result there; slots
/// 0..2 are clobbered (corner A), 3..5 hold the output B.
pub fn curlFromA(comptime SimT: type, sim: *SimT, scratch: anytype, src: anytype, base: usize) @TypeOf(scratch) {
    cornerAverageA(SimT, sim, scratch, src, base);
    calcBfromACore(SimT, sim, scratch);
    return scratch;
}

/// C: calc_BfromA_core (magn.c:568). Curl of the corner A over the domain,
/// stored back into the `scratch` slots (as C reuses pvecpot[1..3]).
fn calcBfromACore(comptime SimT: type, sim: *SimT, scratch: anytype) void {
    const nx = sim.nxi();
    const ny = sim.nyi();
    const nz = sim.nzi();
    const g = &sim.grid;

    std.debug.assert(!(ny == 1 and nz == 1)); // C: my_err("1D not implemented")

    var iz: i64 = 0;
    while (iz < nz) : (iz += 1) {
        var iy: i64 = 0;
        while (iy < ny) : (iy += 1) {
            var ix: i64 = 0;
            while (ix < nx) : (ix += 1) {
                var dA: [4][4]f64 = @splat(@splat(0));

                if (nz == 1) {
                    // flux-ct-compatible, axisymmetric (magn.c:597-618)
                    inline for (0..3) |c| {
                        dA[c + 1][2] = -(scratch.get(c, ix, iy, iz) - scratch.get(c, ix, iy + 1, iz) +
                            scratch.get(c, ix + 1, iy, iz) - scratch.get(c, ix + 1, iy + 1, iz)) /
                            (2.0 * g.cellSize(iy, 1));
                        dA[c + 1][1] = -(scratch.get(c, ix, iy, iz) + scratch.get(c, ix, iy + 1, iz) -
                            scratch.get(c, ix + 1, iy, iz) - scratch.get(c, ix + 1, iy + 1, iz)) /
                            (2.0 * g.cellSize(ix, 0));
                        dA[c + 1][3] = 0;
                    }
                } else {
                    // full 3D corner differences (magn.c:622-659)
                    inline for (0..3) |c| {
                        dA[c + 1][1] = (scratch.get(c, ix + 1, iy, iz) - scratch.get(c, ix, iy, iz) +
                            scratch.get(c, ix + 1, iy + 1, iz) - scratch.get(c, ix, iy + 1, iz) +
                            scratch.get(c, ix + 1, iy, iz + 1) - scratch.get(c, ix, iy, iz + 1) +
                            scratch.get(c, ix + 1, iy + 1, iz + 1) - scratch.get(c, ix, iy + 1, iz + 1)) /
                            (4.0 * g.cellSize(ix, 0));
                        dA[c + 1][2] = (scratch.get(c, ix, iy + 1, iz) - scratch.get(c, ix, iy, iz) +
                            scratch.get(c, ix + 1, iy + 1, iz) - scratch.get(c, ix + 1, iy, iz) +
                            scratch.get(c, ix, iy + 1, iz + 1) - scratch.get(c, ix, iy, iz + 1) +
                            scratch.get(c, ix + 1, iy + 1, iz + 1) - scratch.get(c, ix + 1, iy, iz + 1)) /
                            (4.0 * g.cellSize(iy, 1));
                        dA[c + 1][3] = (scratch.get(c, ix, iy, iz + 1) - scratch.get(c, ix, iy, iz) +
                            scratch.get(c, ix + 1, iy, iz + 1) - scratch.get(c, ix + 1, iy, iz) +
                            scratch.get(c, ix, iy + 1, iz + 1) - scratch.get(c, ix, iy + 1, iz) +
                            scratch.get(c, ix + 1, iy + 1, iz + 1) - scratch.get(c, ix + 1, iy + 1, iz)) /
                            (4.0 * g.cellSize(iz, 2));
                    }
                }

                const gdet = sim.cache.gdet(ix, iy, iz); // finding #1: no full Geometry build
                scratch.set(3, ix, iy, iz, (dA[2][3] - dA[3][2]) / gdet);
                scratch.set(4, ix, iy, iz, (dA[3][1] - dA[1][3]) / gdet);
                scratch.set(5, ix, iy, iz, (dA[1][2] - dA[2][1]) / gdet);
            }
        }
    }
}

/// C: calc_divB (magn.c:693). Corner-based ∇·(√−g B)/√−g at cell corners
/// using cell-centered values. Returns 0 outside the domain, like C.
pub fn calcDivB(comptime SimT: type, sim: *const SimT, ix: i64, iy: i64, iz: i64) f64 {
    const L = SimT.Layout;
    if (comptime !L.hasVar(.b1)) return 0;
    const b1 = comptime L.index(.b1);
    const b2 = comptime L.index(.b2);
    const b3 = comptime L.index(.b3);

    if (ix < 0 or ix >= sim.nxi() or iy < 0 or iy >= sim.nyi() or iz < 0 or iz >= sim.nzi()) return 0;

    const g = &sim.grid;
    const pB = struct {
        fn f(s: *const SimT, iv: usize, jx: i64, jy: i64, jz: i64) f64 {
            return s.cache.gdet(jx, jy, jz) * s.p.get(iv, jx, jy, jz); // finding #1
        }
    }.f;

    var divB: f64 = undefined;
    if (sim.nzi() == 1) {
        divB = (pB(sim, b1, ix, iy, iz) + pB(sim, b1, ix, iy - 1, iz) -
            pB(sim, b1, ix - 1, iy, iz) - pB(sim, b1, ix - 1, iy - 1, iz)) /
            (2.0 * (g.xc(ix + 1) - g.xc(ix))) +
            (pB(sim, b2, ix, iy, iz) + pB(sim, b2, ix - 1, iy, iz) -
                pB(sim, b2, ix, iy - 1, iz) - pB(sim, b2, ix - 1, iy - 1, iz)) /
                (2.0 * (g.yc(iy + 1) - g.yc(iy)));
    } else {
        divB = (pB(sim, b1, ix, iy, iz) + pB(sim, b1, ix, iy - 1, iz) -
            pB(sim, b1, ix - 1, iy, iz) - pB(sim, b1, ix - 1, iy - 1, iz) +
            pB(sim, b1, ix, iy, iz - 1) + pB(sim, b1, ix, iy - 1, iz - 1) -
            pB(sim, b1, ix - 1, iy, iz - 1) - pB(sim, b1, ix - 1, iy - 1, iz - 1)) /
            (4.0 * (g.xc(ix) - g.xc(ix - 1))) +
            (pB(sim, b2, ix, iy, iz) + pB(sim, b2, ix - 1, iy, iz) -
                pB(sim, b2, ix, iy - 1, iz) - pB(sim, b2, ix - 1, iy - 1, iz) +
                pB(sim, b2, ix, iy, iz - 1) + pB(sim, b2, ix - 1, iy, iz - 1) -
                pB(sim, b2, ix, iy - 1, iz - 1) - pB(sim, b2, ix - 1, iy - 1, iz - 1)) /
                (4.0 * (g.yc(iy) - g.yc(iy - 1))) +
            (pB(sim, b3, ix, iy, iz) + pB(sim, b3, ix - 1, iy, iz) -
                pB(sim, b3, ix, iy, iz - 1) - pB(sim, b3, ix - 1, iy, iz - 1) +
                pB(sim, b3, ix, iy - 1, iz) + pB(sim, b3, ix - 1, iy - 1, iz) -
                pB(sim, b3, ix, iy - 1, iz - 1) - pB(sim, b3, ix - 1, iy - 1, iz - 1)) /
                (4.0 * (g.zc(iz) - g.zc(iz - 1)));
    }

    return divB / sim.cache.fillGeometry(ix, iy, iz).gdet;
}
