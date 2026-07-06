//! Row-parallel dispatch for the per-cell inversions (u2p / implicit).
//! `parallelRows` splits the iy-rows across OS threads; the sim-specific
//! worker bodies stay in sim.zig and tally into `ChunkResult`. Rows are
//! disjoint and each per-cell inversion only touches its own cell (geometry
//! reads are `*const`), so the parallel result is bit-identical to the serial
//! one — the golden tests run at nthreads=1 and a determinism test pins it.

const std = @import("std");

const Error = @import("../relele.zig").Error || error{OutOfMemory};

/// Per-worker scratch: the first error a chunk hit, plus its slice of the
/// implicit-solver counters (summed after the join — integer add is
/// order-independent, so the result is identical to serial).
pub const ChunkResult = struct { err: ?Error = null, n_fail: u64 = 0, n_iters: u64 = 0 };

/// Run `worker` over the iy-rows, split across opt.nthreads OS threads when >1.
pub fn parallelRows(
    comptime SimT: type,
    sim: *SimT,
    comptime worker: fn (sim: *SimT, iy0: i64, iy1: i64, res: *ChunkResult) void,
) Error!void {
    const ny = sim.nyi();
    const nt = @max(@as(usize, 1), sim.opt.nthreads);
    if (nt <= 1 or ny <= 1) {
        var res = ChunkResult{};
        worker(sim, 0, ny, &res);
        sim.n_radimp_failures += res.n_fail;
        sim.n_radimp_iters += res.n_iters;
        if (res.err) |e| return e;
        return;
    }
    const maxt = 64;
    const n_workers: usize = @intCast(@min(@as(i64, @intCast(@min(nt, maxt))), ny));
    var results = [_]ChunkResult{.{}} ** maxt;
    var threads = [_]?std.Thread{null} ** maxt;
    var i: usize = 0;
    while (i < n_workers) : (i += 1) {
        const iy0 = @divTrunc(ny * @as(i64, @intCast(i)), @as(i64, @intCast(n_workers)));
        const iy1 = @divTrunc(ny * @as(i64, @intCast(i + 1)), @as(i64, @intCast(n_workers)));
        threads[i] = std.Thread.spawn(.{}, worker, .{ sim, iy0, iy1, &results[i] }) catch null;
        // spawn failure → run this chunk inline (still correct)
        if (threads[i] == null) worker(sim, iy0, iy1, &results[i]);
    }
    for (0..n_workers) |k| if (threads[k]) |t| t.join();

    var e: ?Error = null;
    for (results[0..n_workers]) |r| {
        sim.n_radimp_failures += r.n_fail;
        sim.n_radimp_iters += r.n_iters;
        if (r.err) |ee| e = ee;
    }
    if (e) |ee| return ee;
}
