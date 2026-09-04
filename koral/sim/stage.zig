//! RK stage arithmetic over whole-grid conserved buffers, generic over
//! SimT (problem.c:181/230/243/357): the full-field copy and the two
//! `a*x + b*y` combination shapes, kept exactly as C writes them so the
//! forced-dt step tests gate at 1e-13. Band-parallel over iy; pure
//! per-cell assignments, so bit-identical to serial.
//!
//! Moved verbatim out of sim.zig (redesign step 2, 2026-09-04). The
//! caller (Sim.step) owns the `.stage` timer.

const threading = @import("../threading.zig");

/// dst ← src over the whole padded field.
pub fn copyFull(comptime SimT: type, self: *SimT, dst: *SimT.FieldT, src: *const SimT.FieldT) void {
    threading.parallelCopy(self.team, dst.data, src.data);
}

/// Worker context for the two stage-arithmetic shapes: deriv uses
/// (dst, a, b, f1, f2); combine also uses c.
fn Ctx(comptime SimT: type) type {
    return struct {
        sim: *SimT,
        dst: *SimT.FieldT,
        a_f: *const SimT.FieldT,
        b_f: *const SimT.FieldT,
        c_f: *const SimT.FieldT = undefined,
        f1: f64,
        f2: f64,
    };
}

/// dst = (1/Δ)·a + (−1/Δ)·b over the domain (problem.c:181/230/…).
pub fn deriv(comptime SimT: type, self: *SimT, dst: *SimT.FieldT, a_f: *const SimT.FieldT, b_f: *const SimT.FieldT, delta: f64) void {
    var ctx = Ctx(SimT){ .sim = self, .dst = dst, .a_f = a_f, .b_f = b_f, .f1 = 1.0 / delta, .f2 = -1.0 / delta };
    _ = threading.parallelRange(Ctx(SimT), &ctx, self.team, 0, self.nyi(), derivWorker(SimT));
}

fn derivWorker(comptime SimT: type) fn (*Ctx(SimT), i64, i64, *threading.ChunkResult) void {
    return struct {
        fn w(ctx: *Ctx(SimT), iy0: i64, iy1: i64, res: *threading.ChunkResult) void {
            _ = res;
            const NV = SimT.nv;
            const self = ctx.sim;
            var iz: i64 = 0;
            while (iz < self.nzi()) : (iz += 1) {
                var iy: i64 = iy0;
                while (iy < iy1) : (iy += 1) {
                    var ix: i64 = 0;
                    while (ix < self.nxi()) : (ix += 1) {
                        // one offset per cell, index the stack buffers per iv (P2 #4)
                        var a_v: [NV]f64 = undefined;
                        var b_v: [NV]f64 = undefined;
                        ctx.a_f.load(ix, iy, iz, &a_v);
                        ctx.b_f.load(ix, iy, iz, &b_v);
                        var d_v: [NV]f64 = undefined;
                        for (0..NV) |iv| d_v[iv] = ctx.f1 * a_v[iv] + ctx.f2 * b_v[iv];
                        ctx.dst.store(ix, iy, iz, &d_v);
                    }
                }
            }
        }
    }.w;
}

/// dst = a + f1·b + f2·c over the domain (problem.c:243/357).
pub fn combine(comptime SimT: type, self: *SimT, dst: *SimT.FieldT, a_f: *const SimT.FieldT, f1: f64, b_f: *const SimT.FieldT, f2: f64, c_f: *const SimT.FieldT) void {
    var ctx = Ctx(SimT){ .sim = self, .dst = dst, .a_f = a_f, .b_f = b_f, .c_f = c_f, .f1 = f1, .f2 = f2 };
    _ = threading.parallelRange(Ctx(SimT), &ctx, self.team, 0, self.nyi(), combineWorker(SimT));
}

fn combineWorker(comptime SimT: type) fn (*Ctx(SimT), i64, i64, *threading.ChunkResult) void {
    return struct {
        fn w(ctx: *Ctx(SimT), iy0: i64, iy1: i64, res: *threading.ChunkResult) void {
            _ = res;
            const NV = SimT.nv;
            const self = ctx.sim;
            var iz: i64 = 0;
            while (iz < self.nzi()) : (iz += 1) {
                var iy: i64 = iy0;
                while (iy < iy1) : (iy += 1) {
                    var ix: i64 = 0;
                    while (ix < self.nxi()) : (ix += 1) {
                        // one offset per cell, index the stack buffers per iv (P2 #4)
                        var a_v: [NV]f64 = undefined;
                        var b_v: [NV]f64 = undefined;
                        var c_v: [NV]f64 = undefined;
                        ctx.a_f.load(ix, iy, iz, &a_v);
                        ctx.b_f.load(ix, iy, iz, &b_v);
                        ctx.c_f.load(ix, iy, iz, &c_v);
                        var d_v: [NV]f64 = undefined;
                        for (0..NV) |iv| d_v[iv] = a_v[iv] + ctx.f1 * b_v[iv] + ctx.f2 * c_v[iv];
                        ctx.dst.store(ix, iy, iz, &d_v);
                    }
                }
            }
        }
    }.w;
}
