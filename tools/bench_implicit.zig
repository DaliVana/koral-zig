//! Implicit-solver benchmark: the scalar vs the SIMD (lane-batched) FD
//! Jacobian over the C-oracle battery (tests/golden/rad_implicit.kgld;
//! 336 PUFFY grid cells spanning ~10 decades of κΔt stiffness). The same
//! battery drives the C timing companion oracle/harness_bench_implicit.c,
//! so all three numbers (C, Zig scalar, Zig SIMD) measure the identical
//! physical workload through the full solve_implicit_lab rung ladder.
//!
//! Before timing, the scalar and SIMD paths are cross-checked bitwise on
//! every cell (the simd_tests.zig gate, re-asserted on this battery).
//!
//! Usage: zig build bench-implicit [-- <rounds>]   (default 30)

const std = @import("std");
const koral = @import("koral");

const cfg = koral.config.puffy;
const L = koral.VarLayout(cfg);
const NV = L.count;
const ImplT = koral.solve.implicit.Solver(cfg);
const golden = koral.testing.golden;

const NIN = 37;
const NOUT = 18;
const gam: f64 = 5.0 / 3.0;

fn nowNs() f64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(f64, @floatFromInt(ts.sec)) * 1e9 + @as(f64, @floatFromInt(ts.nsec));
}

const Rec = struct { geo: koral.Geometry, dt: f64, pp0: [NV]f64 };

const ImplicitParams = koral.solve.implicit.ImplicitParams;
const RadParams = koral.solve.invert_rad.RadParams;
const Params = koral.physics.radforce.Params;

/// Shared state for the threaded throughput measurement. The battery is
/// embarrassingly parallel (each solveImplicitLab is pure; no shared mutable
/// state, the P1 thread-safety contract), so we hand out cell·round work items
/// via a dynamic ticket (fetchAdd); the ~17× floor/torus cost spread means a
/// static split would straggle, exactly as in the production tile scheduler.
const Task = struct {
    recs: []const Rec,
    ip: *const ImplicitParams,
    p: *const Params,
    rp: RadParams,
    total: usize, // recs.len * rounds
    ticket: std.atomic.Value(usize) align(64) = .{ .raw = 0 },
    sink: std.atomic.Value(usize) align(64) = .{ .raw = 0 },
};

fn benchWorker(t: *Task) void {
    const nrec = t.recs.len;
    var ls: usize = 0;
    while (true) {
        const k = t.ticket.fetchAdd(1, .monotonic);
        if (k >= t.total) break;
        const rec = &t.recs[k % nrec];
        var pp = rec.pp0;
        var uu: [NV]f64 = undefined;
        const res = ImplT.solveImplicitLab(&uu, &pp, &rec.geo, rec.dt, gam, t.rp, t.p, t.ip);
        ls +%= res.iters;
    }
    _ = t.sink.fetchAdd(ls, .monotonic);
}

/// Effective system ns/solve at `nthreads`: total wall time / total solves,
/// dynamically load-balanced. Single spawn/join per call (amortized to nil).
fn measureThroughput(
    allocator: std.mem.Allocator,
    nthreads: usize,
    recs: []const Rec,
    ip: *const ImplicitParams,
    p: *const Params,
    rp: RadParams,
    rounds: usize,
) !f64 {
    const helpers = try allocator.alloc(std.Thread, nthreads - 1);
    defer allocator.free(helpers);

    var best: f64 = 1e300;
    for (0..3) |_| { // a few passes, keep the fastest (least scheduler noise)
        var task = Task{ .recs = recs, .ip = ip, .p = p, .rp = rp, .total = recs.len * rounds };
        const t0 = nowNs();
        for (helpers) |*h| h.* = try std.Thread.spawn(.{}, benchWorker, .{&task});
        benchWorker(&task); // main participates as worker 0
        for (helpers) |h| h.join();
        const el = nowNs() - t0;
        std.mem.doNotOptimizeAway(task.sink.load(.monotonic));
        best = @min(best, el);
    }
    return best / @as(f64, @floatFromInt(recs.len * rounds)); // ns/solve
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    var args = std.process.Args.Iterator.init(init.minimal.args);
    defer args.deinit();
    _ = args.next(); // program name
    const path = args.next() orelse "tests/golden/rad/rad_implicit.kgld";
    const rounds: usize = if (args.next()) |a| try std.fmt.parseInt(usize, a, 10) else 30;
    const nthreads: usize = if (args.next()) |a| try std.fmt.parseInt(usize, a, 10) else 1;

    const raw = std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(1 << 24)) catch |err| {
        std.debug.print("bench_implicit: cannot read '{s}': {s}\n", .{ path, @errorName(err) });
        return err;
    };
    defer allocator.free(raw);

    if (raw.len < 24 or !std.mem.eql(u8, raw[0..4], "KGLD")) return error.BadGoldenFile;
    if (std.mem.readInt(u32, raw[4..8], .little) != 1) return error.BadGoldenVersion;
    const nrec: usize = @intCast(std.mem.readInt(u64, raw[8..16], .little));
    if (std.mem.readInt(u32, raw[16..20], .little) != NIN) return error.BadGoldenFile;
    if (std.mem.readInt(u32, raw[20..24], .little) != NOUT) return error.BadGoldenFile;

    const recs = try allocator.alloc(Rec, nrec);
    defer allocator.free(recs);
    for (0..nrec) |ir| {
        var vals: [NIN]f64 = undefined;
        const off = 24 + ir * (NIN + NOUT) * 8;
        for (0..NIN) |i| {
            vals[i] = @bitCast(std.mem.readInt(u64, raw[off + i * 8 ..][0..8], .little));
        }
        var pp0: [NV]f64 = undefined;
        for (0..NV) |iv| pp0[iv] = vals[24 + iv];
        recs[ir] = .{
            .geo = golden.geomFromRecord(vals[0..23], .mks2, @splat(0)),
            .dt = vals[23],
            .pp0 = pp0,
        };
    }

    const p = koral.physics.radforce.Params.puffy();
    const rp = koral.solve.invert_rad.RadParams.puffy;
    var ip_scalar = koral.solve.implicit.ImplicitParams.puffy;
    ip_scalar.simd_jacobian = false;
    var ip_simd = koral.solve.implicit.ImplicitParams.puffy;
    ip_simd.simd_jacobian = true;
    // the same pair without the 1dprim bisection pre-pass — the bisect
    // runs the residual chain scalar outside the Jacobian, so this pair
    // isolates what the lane batch does to the Newton loop itself
    var ip_scalar_nb = ip_scalar;
    ip_scalar_nb.start_with_bisect = false;
    var ip_simd_nb = ip_simd;
    ip_simd_nb.start_with_bisect = false;

    // ---- bitwise cross-check (untimed) ------------------------------------
    var nok: usize = 0;
    var mismatch: usize = 0;
    for (recs) |*rec| {
        var pp_a = rec.pp0;
        var pp_b = rec.pp0;
        var uu_a: [NV]f64 = undefined;
        var uu_b: [NV]f64 = undefined;
        const ra = ImplT.solveImplicitLab(&uu_a, &pp_a, &rec.geo, rec.dt, gam, rp, &p, &ip_scalar);
        const rb = ImplT.solveImplicitLab(&uu_b, &pp_b, &rec.geo, rec.dt, gam, rp, &p, &ip_simd);
        if (ra.ok != rb.ok or ra.rung != rb.rung or ra.iters != rb.iters) mismatch += 1;
        for (0..NV) |iv| {
            if (@as(u64, @bitCast(pp_a[iv])) != @as(u64, @bitCast(pp_b[iv]))) mismatch += 1;
        }
        if (ra.ok) nok += 1;
    }
    std.debug.print(
        "battery: {d} cells, {d} converge, scalar/simd bitwise mismatches: {d}\n",
        .{ nrec, nok, mismatch },
    );
    if (mismatch != 0) return error.SimdScalarMismatch;

    // ---- timing ------------------------------------------------------------
    const variants = [_]struct { name: []const u8, ip: *const koral.solve.implicit.ImplicitParams }{
        .{ .name = "scalar          ", .ip = &ip_scalar },
        .{ .name = "simd            ", .ip = &ip_simd },
        .{ .name = "scalar no-bisect", .ip = &ip_scalar_nb },
        .{ .name = "simd   no-bisect", .ip = &ip_simd_nb },
    };
    var avg_ns = [_]f64{0} ** variants.len;
    var best_ns = [_]f64{1e300} ** variants.len;

    for (variants, 0..) |v, vi| {
        for (0..rounds + 1) |r| { // round 0 = warmup
            var sink: usize = 0;
            const t0 = nowNs();
            for (recs) |*rec| {
                var pp = rec.pp0;
                var uu: [NV]f64 = undefined;
                const res = ImplT.solveImplicitLab(&uu, &pp, &rec.geo, rec.dt, gam, rp, &p, v.ip);
                sink +%= res.iters;
            }
            const el = nowNs() - t0;
            std.mem.doNotOptimizeAway(sink);
            if (r > 0) {
                avg_ns[vi] += el;
                best_ns[vi] = @min(best_ns[vi], el);
            }
        }
        avg_ns[vi] /= @as(f64, @floatFromInt(rounds));
        std.debug.print(
            "{s}: battery avg {d:.3} ms, best {d:.3} ms | {d:.0} ns/solve (avg over {d} rounds)\n",
            .{ v.name, avg_ns[vi] / 1e6, best_ns[vi] / 1e6, avg_ns[vi] / @as(f64, @floatFromInt(nrec)), rounds },
        );
    }
    std.debug.print(
        "simd speedup on the implicit solve: {d:.3}x (avg), {d:.3}x (best) | Newton-only (no bisect): {d:.3}x (avg)\n",
        .{ avg_ns[0] / avg_ns[1], best_ns[0] / best_ns[1], avg_ns[2] / avg_ns[3] },
    );

    // ---- threaded throughput (production config = SIMD) --------------------
    // effective system ns/solve = wall / total solves, at 1 vs `nthreads`.
    const eff1 = try measureThroughput(allocator, 1, recs, &ip_simd, &p, rp, rounds);
    std.debug.print("\nthreaded throughput (simd, dynamic ticket): 1 thread = {d:.0} ns/solve\n", .{eff1});
    if (nthreads > 1) {
        const effn = try measureThroughput(allocator, nthreads, recs, &ip_simd, &p, rp, rounds);
        std.debug.print(
            "                                            {d} threads = {d:.0} ns/solve  ({d:.2}x scaling, {d:.1}% efficiency)\n",
            .{ nthreads, effn, eff1 / effn, 100.0 * (eff1 / effn) / @as(f64, @floatFromInt(nthreads)) },
        );
    }
}
