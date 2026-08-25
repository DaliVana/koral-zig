//! P1 of the parallelization plan (docs/PARALLELIZATION_PLAN_2026-07-06.md
//! §3.2): a persistent worker team with dynamic tiling, behind a small
//! band-parallel dispatch API used by every per-cell/per-face pass.
//!
//! Design (and why the result stays bit-identical to serial):
//!  * `Team` spawns nthreads−1 helper OS threads once (at Sim.init) and
//!    parks them on a futex'd region counter between regions; no per-pass
//!    spawn/join. The main thread participates in every region and then
//!    waits (futex again) for the helpers' completion count. (Zig 0.16
//!    moved Mutex/Condition behind the std.Io event-loop interface, so the
//!    park/wake goes straight to the OS futex: linux futex / darwin ulock,
//!    the same calls std.Io.Threaded makes internally.)
//!  * A region is (task, ctx, [lo,hi), tile): workers grab contiguous
//!    tiles of the index range through an atomic ticket counter, so cheap
//!    tiles fly and expensive tiles straggle-balance (the implicit solver's
//!    ~17× torus/floor cell-cost spread is the motivating case).
//!  * Every pass body writes only cells/columns/faces of its own band and
//!    reads only state that is frozen during the region (geometry cache,
//!    the pre-pass fields), so *which* worker runs a tile cannot change any
//!    written value.
//!  * The only cross-band results are the `ChunkResult` reductions: integer
//!    sums (implicit counters) and f64 max/min (the CFL denominator); all
//!    order-insensitive, merged once per region in a fixed order. Each
//!    worker accumulates into a stack-local ChunkResult across its tiles
//!    and publishes once at region end (no false sharing on hot counters).
//!  * Errors: a worker records the first error it hits in its ChunkResult;
//!    the merged result carries one of them (they are all OutOfMemory-class
//!    aborts). Thread-spawn failure at Team.init just means fewer helpers.
//!
//! The golden tests all run at nthreads=1 (team == null ⇒ the worker runs
//! inline over the whole range); the threading tests pin nthreads>1 to be
//! bit-identical to that serial path.

const std = @import("std");
const builtin = @import("builtin");

const Error = @import("relele.zig").Error || error{OutOfMemory};

/// Minimal cross-platform futex on a u32 word (std's own moved behind
/// std.Io in 0.16). `wait` returns when *ptr != expect, on a wake, or
/// spuriously: callers always re-check their condition in a loop.
const futex = struct {
    fn wait(ptr: *const std.atomic.Value(u32), expect: u32) void {
        switch (builtin.os.tag) {
            .linux => {
                _ = std.os.linux.futex_4arg(&ptr.raw, .{ .cmd = .WAIT, .private = true }, expect, null);
            },
            .driverkit, .ios, .maccatalyst, .macos, .tvos, .visionos, .watchos => {
                _ = std.c.__ulock_wait(
                    .{ .op = .COMPARE_AND_WAIT, .NO_ERRNO = true },
                    &ptr.raw,
                    expect,
                    0, // no timeout
                );
            },
            // portability stopgap: busy-poll with a scheduler yield
            else => std.Thread.yield() catch {},
        }
    }

    fn wake(ptr: *const std.atomic.Value(u32), n_waiters: u32) void {
        switch (builtin.os.tag) {
            .linux => {
                _ = std.os.linux.futex_3arg(
                    &ptr.raw,
                    .{ .cmd = .WAKE, .private = true },
                    @min(n_waiters, std.math.maxInt(i32)),
                );
            },
            .driverkit, .ios, .maccatalyst, .macos, .tvos, .visionos, .watchos => {
                while (true) {
                    const status = std.c.__ulock_wake(
                        .{ .op = .COMPARE_AND_WAIT, .NO_ERRNO = true, .WAKE_ALL = n_waiters > 1 },
                        &ptr.raw,
                        0,
                    );
                    if (status >= 0) return;
                    switch (@as(std.c.E, @enumFromInt(-status))) {
                        .INTR, .CANCELED => continue,
                        else => return, // NOENT: nobody waiting
                    }
                }
            },
            else => {},
        }
    }
};

const inf = std.math.inf(f64);

// ---- per-core pinning (MPI plan P4b: node-width hardening) ----------------
//
// Linux-only: the mask sched_getaffinity(0) reports is the cgroup cpuset the
// launcher (Slurm/mpiexec) granted this rank, so distributing threads over
// exactly those ids stays inside the allocation by construction. The main
// thread takes the first allowed cpu and helper i the (i+1)-th, wrapping if
// the team is wider than the mask. Elsewhere (macOS has no thread→core
// binding API) both helpers are no-ops and the flag is inert — the efficacy
// measurement at 128/192 threads is a cluster-CI item, not a laptop one.

/// The allowed CPU ids in mask order, or null (non-Linux / error / empty).
fn allowedCpus(allocator: std.mem.Allocator) ?[]u32 {
    if (comptime builtin.os.tag != .linux) return null;
    var set: std.os.linux.cpu_set_t = @splat(0);
    if (std.os.linux.sched_getaffinity(0, @sizeOf(std.os.linux.cpu_set_t), &set) != 0) return null;
    const total: usize = std.os.linux.CPU_COUNT(set);
    if (total == 0) return null;
    const cpus = allocator.alloc(u32, total) catch return null;
    var k: usize = 0;
    for (set, 0..) |word, wi| {
        var bit: usize = 0;
        while (bit < @bitSizeOf(usize)) : (bit += 1) {
            if (word & (@as(usize, 1) << @intCast(bit)) != 0) {
                cpus[k] = @intCast(wi * @bitSizeOf(usize) + bit);
                k += 1;
            }
        }
    }
    std.debug.assert(k == total);
    return cpus;
}

/// Bind the CALLING thread to one cpu (pid 0 = self). Best-effort.
fn pinSelf(cpu: u32) void {
    if (comptime builtin.os.tag != .linux) return;
    var set: std.os.linux.cpu_set_t = @splat(0);
    set[cpu / @bitSizeOf(usize)] = @as(usize, 1) << @intCast(cpu % @bitSizeOf(usize));
    std.os.linux.sched_setaffinity(0, &set) catch {};
}

/// Per-worker scratch merged after each region. Passes use what they need:
/// the implicit solver its counters, the wavespeed pass its CFL-denominator
/// partials: everything else only the error slot. All merges are
/// order-insensitive, so the merged result is independent of the tile
/// schedule (and identical to serial).
pub const ChunkResult = struct {
    err: ?Error = null,
    n_fail: u64 = 0,
    n_iters: u64 = 0,
    /// successful implicit solves (so iters/solves gives an average iteration
    /// count for the heartbeat; C tracks GLOBALINTSLOT_NIMP* per category).
    n_solves: u64 = 0,
    /// running max/min of the CFL denominator (wavespeed pass only);
    /// neutral at ∓inf so non-participating passes merge as no-ops.
    tsd_max: f64 = -inf,
    tsd_min: f64 = inf,

    pub fn merge(self: *ChunkResult, other: *const ChunkResult) void {
        if (self.err == null) self.err = other.err;
        self.n_fail += other.n_fail;
        self.n_iters += other.n_iters;
        self.n_solves += other.n_solves;
        if (other.tsd_max > self.tsd_max) self.tsd_max = other.tsd_max;
        if (other.tsd_min < self.tsd_min) self.tsd_min = other.tsd_min;
    }
};

/// Aim for this many tiles per worker per region; enough granularity for
/// the ticket counter to absorb per-tile cost imbalance, few enough that
/// ticket traffic stays negligible (§3.2 "dynamic tiling").
const tiles_per_worker = 8;

/// The persistent worker team. One per Sim (created when nthreads > 1),
/// shared by all passes; only the main thread publishes regions.
pub const Team = struct {
    const Task = *const fn (ctx: *anyopaque, lo: i64, hi: i64, res: *ChunkResult) void;

    allocator: std.mem.Allocator,
    helpers: []std.Thread,
    /// one publish slot per helper, written once at that helper's region end
    results: []ChunkResult,
    /// pin=true on Linux: the allowed-cpu list each thread binds into
    /// (main → [0], helper i → [(i+1) % len]); null = no pinning.
    pin_cpus: ?[]u32 = null,

    // region state — plain fields published by the release-increment of
    // `gen` and read by helpers after their acquire-load of it
    task: Task = undefined,
    ctx: *anyopaque = undefined,
    hi: i64 = 0,
    tile: i64 = 1,
    shutdown: bool = false,
    /// the dynamic-tile ticket counter
    ticket: std.atomic.Value(i64) = .init(0),
    /// region generation; helpers park on this futex word between regions
    gen: std.atomic.Value(u32) = .init(0),
    /// helpers still inside the current region; main parks on this word
    n_running: std.atomic.Value(u32) = .init(0),

    /// Spawn `nthreads − 1` helpers (the main thread is worker 0). If some
    /// spawns fail the team just runs narrower; never an error. `pin`
    /// binds each thread (main included) to one cpu of the process's
    /// affinity mask (Linux; no-op elsewhere. See allowedCpus above).
    pub fn init(allocator: std.mem.Allocator, nthreads: usize, pin: bool) !*Team {
        const n_helpers = @max(nthreads, 1) - 1;
        const team = try allocator.create(Team);
        errdefer allocator.destroy(team);
        const results = try allocator.alloc(ChunkResult, n_helpers);
        errdefer allocator.free(results);
        const helpers = try allocator.alloc(std.Thread, n_helpers);

        team.* = .{ .allocator = allocator, .helpers = helpers[0..0], .results = results };
        if (pin) {
            // Before any spawn, so every helperMain sees the final list.
            team.pin_cpus = allowedCpus(allocator);
            if (team.pin_cpus) |cpus| pinSelf(cpus[0]);
        }
        for (0..n_helpers) |i| {
            const t = std.Thread.spawn(.{}, helperMain, .{ team, i }) catch break;
            helpers[i] = t;
            team.helpers = helpers[0 .. i + 1];
        }
        return team;
    }

    /// How many distinct cpus the team is pinned over (null = not pinned).
    pub fn pinnedWidth(team: *const Team) ?usize {
        return if (team.pin_cpus) |c| c.len else null;
    }

    pub fn deinit(team: *Team) void {
        team.shutdown = true;
        _ = team.gen.fetchAdd(1, .release);
        futex.wake(&team.gen, std.math.maxInt(u32));
        for (team.helpers) |t| t.join();
        const allocator = team.allocator;
        allocator.free(team.helpers.ptr[0..team.results.len]);
        allocator.free(team.results);
        if (team.pin_cpus) |cpus| allocator.free(cpus);
        allocator.destroy(team);
    }

    /// Run one region: publish, participate, wait, merge. Main-thread only,
    /// non-reentrant (no pass launches a region from inside another).
    pub fn run(team: *Team, ctx: *anyopaque, task: Task, lo: i64, hi: i64) ChunkResult {
        const n = hi - lo;
        const n_workers: i64 = @intCast(team.helpers.len + 1);
        const tile = @max(1, @divTrunc(n, tiles_per_worker * n_workers));

        team.task = task;
        team.ctx = ctx;
        team.hi = hi;
        team.tile = tile;
        team.ticket.store(lo, .monotonic);
        team.n_running.store(@intCast(team.helpers.len), .monotonic);
        _ = team.gen.fetchAdd(1, .release); // publishes everything above
        futex.wake(&team.gen, std.math.maxInt(u32));

        var merged = ChunkResult{};
        runTiles(team, task, ctx, hi, tile, &merged);

        // wait until every helper has published its result
        while (true) {
            const r = team.n_running.load(.acquire);
            if (r == 0) break;
            futex.wait(&team.n_running, r);
        }

        // fixed merge order: main first, then helpers 0..n — and every
        // reduction is order-insensitive anyway
        for (team.results) |*r| merged.merge(r);
        return merged;
    }

    fn helperMain(team: *Team, wid: usize) void {
        if (team.pin_cpus) |cpus| pinSelf(cpus[(wid + 1) % cpus.len]);
        var seen: u32 = 0;
        while (true) {
            // park until a new region (or shutdown) bumps the generation
            while (true) {
                const g = team.gen.load(.acquire);
                if (g != seen) {
                    seen = g;
                    break;
                }
                futex.wait(&team.gen, g);
            }
            if (team.shutdown) return;

            var local = ChunkResult{};
            runTiles(team, team.task, team.ctx, team.hi, team.tile, &local);
            team.results[wid] = local;

            // last helper out wakes the waiting main thread
            if (team.n_running.fetchSub(1, .release) == 1) {
                futex.wake(&team.n_running, 1);
            }
        }
    }

    /// The dynamic-tile loop: grab [t_lo, t_lo+tile) slices until the range
    /// is exhausted. `local` lives on this worker's stack across all its
    /// tiles (accumulate locally, publish once).
    fn runTiles(team: *Team, task: Task, ctx: *anyopaque, hi: i64, tile: i64, local: *ChunkResult) void {
        while (true) {
            const t_lo = team.ticket.fetchAdd(tile, .monotonic);
            if (t_lo >= hi) return;
            task(ctx, t_lo, @min(t_lo + tile, hi), local);
        }
    }
};

/// Run `worker` over [lo, hi): inline when `team` is null (the serial /
/// nthreads=1 path; the golden-tested reference), else as one team region
/// with dynamic tiles. Returns the merged ChunkResult; the caller checks
/// `.err` and consumes whatever reductions the pass produces.
pub fn parallelRange(
    comptime CtxT: type,
    ctx: *CtxT,
    team: ?*Team,
    lo: i64,
    hi: i64,
    comptime worker: fn (ctx: *CtxT, b_lo: i64, b_hi: i64, res: *ChunkResult) void,
) ChunkResult {
    if (hi <= lo) return .{};
    if (team) |t| {
        const Tramp = struct {
            fn task(raw: *anyopaque, b_lo: i64, b_hi: i64, res: *ChunkResult) void {
                worker(@ptrCast(@alignCast(raw)), b_lo, b_hi, res);
            }
        };
        return t.run(@ptrCast(ctx), Tramp.task, lo, hi);
    }
    var res = ChunkResult{};
    worker(ctx, lo, hi, &res);
    return res;
}

/// `parallelRange` for the passes whose only `ChunkResult` use is the error
/// slot: the great majority. The worker returns `Error!void` directly and the
/// first failing band's error is propagated (`merge` keeps the first), so each
/// call site drops both its hand-written `catch |e| { res.err = e; }` shim and
/// its `if (res.err) |e| return e;` tail. Passes that also *reduce* keep the
/// explicit `ChunkResult` form: wavespeeds (tsd_max/min) and the implicit
/// operator (iteration/solve/failure counters).
pub fn parallelRangeErr(
    comptime CtxT: type,
    ctx: *CtxT,
    team: ?*Team,
    lo: i64,
    hi: i64,
    comptime worker: fn (ctx: *CtxT, b_lo: i64, b_hi: i64) Error!void,
) Error!void {
    const Shim = struct {
        fn w(c: *CtxT, b_lo: i64, b_hi: i64, res: *ChunkResult) void {
            worker(c, b_lo, b_hi) catch |e| {
                res.err = e;
            };
        }
    };
    if (parallelRange(CtxT, ctx, team, lo, hi, Shim.w).err) |e| return e;
}

const CopyCtx = struct { dst: []f64, src: []const f64 };

fn copyWorker(c: *CopyCtx, lo: i64, hi: i64, res: *ChunkResult) void {
    _ = res;
    const a: usize = @intCast(lo);
    const b: usize = @intCast(hi);
    @memcpy(c.dst[a..b], c.src[a..b]);
}

/// dst = src, split across the team (full-field stage copies).
pub fn parallelCopy(team: ?*Team, dst: []f64, src: []const f64) void {
    std.debug.assert(dst.len == src.len);
    var ctx = CopyCtx{ .dst = dst, .src = src };
    _ = parallelRange(CopyCtx, &ctx, team, 0, @intCast(dst.len), copyWorker);
}

const ZeroCtx = struct { dst: []f64 };

fn zeroWorker(c: *ZeroCtx, lo: i64, hi: i64, res: *ChunkResult) void {
    _ = res;
    @memset(c.dst[@intCast(lo)..@intCast(hi)], 0);
}

/// @memset(dst, 0), split across the team.
pub fn parallelZero(team: ?*Team, dst: []f64) void {
    var ctx = ZeroCtx{ .dst = dst };
    _ = parallelRange(ZeroCtx, &ctx, team, 0, @intCast(dst.len), zeroWorker);
}

// ---- unit tests ----------------------------------------------------------

test "Team: dynamic tiles cover the range exactly once" {
    const allocator = std.testing.allocator;
    const team = try Team.init(allocator, 4, false);
    defer team.deinit();

    var hits = [_]std.atomic.Value(u32){.init(0)} ** 103;
    const Ctx = struct { hits: []std.atomic.Value(u32) };
    var ctx = Ctx{ .hits = &hits };
    const W = struct {
        fn w(c: *Ctx, lo: i64, hi: i64, res: *ChunkResult) void {
            var i = lo;
            while (i < hi) : (i += 1) {
                _ = c.hits[@intCast(i)].fetchAdd(1, .monotonic);
                res.n_iters += 1;
            }
        }
    };
    // several regions in a row (reuse of the parked team)
    for (0..5) |_| {
        for (&hits) |*h| h.store(0, .monotonic);
        const res = parallelRange(Ctx, &ctx, team, 0, 103, W.w);
        try std.testing.expectEqual(@as(u64, 103), res.n_iters);
        for (&hits) |*h| try std.testing.expectEqual(@as(u32, 1), h.load(.monotonic));
    }
}

test "Team: reductions merge like serial" {
    const allocator = std.testing.allocator;
    const team = try Team.init(allocator, 3, false);
    defer team.deinit();

    const Ctx = struct {};
    var ctx = Ctx{};
    const W = struct {
        fn w(c: *Ctx, lo: i64, hi: i64, res: *ChunkResult) void {
            _ = c;
            var i = lo;
            while (i < hi) : (i += 1) {
                res.n_fail += 1;
                const v: f64 = @floatFromInt(@mod(i * 37, 101));
                if (v > res.tsd_max) res.tsd_max = v;
                if (v < res.tsd_min) res.tsd_min = v;
            }
        }
    };
    const par = parallelRange(Ctx, &ctx, team, 0, 500, W.w);
    const ser = parallelRange(Ctx, &ctx, null, 0, 500, W.w);
    try std.testing.expectEqual(ser.n_fail, par.n_fail);
    try std.testing.expectEqual(ser.tsd_max, par.tsd_max);
    try std.testing.expectEqual(ser.tsd_min, par.tsd_min);
}

test "parallelCopy and parallelZero" {
    const allocator = std.testing.allocator;
    const team = try Team.init(allocator, 4, false);
    defer team.deinit();

    const n = 10_007;
    const src = try allocator.alloc(f64, n);
    defer allocator.free(src);
    const dst = try allocator.alloc(f64, n);
    defer allocator.free(dst);
    for (src, 0..) |*v, i| v.* = @floatFromInt(i);

    parallelCopy(team, dst, src);
    for (dst, src) |d, s| try std.testing.expectEqual(s, d);
    parallelZero(team, dst);
    for (dst) |d| try std.testing.expectEqual(@as(f64, 0), d);
}
