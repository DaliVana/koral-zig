//! Slow-light rendering: the time-ordered streaming sweep.
//!
//! Every backward-traced ray's coordinate time x⁰ is STRICTLY DECREASING:
//! in (M)KS coordinates g^tt < 0 everywhere outside the horizon, so t is a
//! global time function and any future-directed null vector has k⁰ > 0;
//! integrating with Δλ < 0 therefore runs monotonically into the past, with
//! no exceptions in the ergosphere or near the horizon (KS time is
//! horizon-regular: this is exactly why slow light is clean in these
//! coordinates and pathological in Boyer-Lindquist). Monotonicity turns
//! slow light into a single streaming pass over the dump series:
//!
//!  A. FREE FLIGHT; all rays advance from the camera through the static
//!     outer region (r > r_slow), sampling the reference frame (Scene.data),
//!     until they first cross into r < r_slow or finish outright. Outside
//!     r_slow the flow is treated as stationary; for PUFFY that region is
//!     floor atmosphere with orbital times ≫ the series span.
//!  B. SWEEP; a window of two consecutive frames [t_lo, t_hi] slides from
//!     the latest useful pair toward the past. Each round, every active ray
//!     advances (WindowSampler interpolating at its own retarded time x⁰)
//!     until x⁰ < t_lo; then it parks, and the window slides once all are
//!     parked. Each frame is read from disk EXACTLY ONCE, and at most two
//!     frames (+ the reference) are resident: an 18 GB series renders in a
//!     ~600 MB footprint. A ray that leaves r_slow spatially is done with
//!     the time-varying zone FOR GOOD; a Kerr null geodesic has at most
//!     one radial turning point outside the photon shell, so it can never
//!     re-enter: and joins the tail list. If the series runs out while
//!     rays are still inside (their x⁰ predates the first frame), a final
//!     round holds the first frame (clamp, counted in Stats).
//!  C. TAILS; rays that exited r_slow finish through the static outer
//!     region on the reference frame, to escape/capture/τ_max.
//!
//! Pause/resume never touches the arithmetic inside a step; with a static
//! series the sweep is BIT-IDENTICAL to plain fast-light tracing (gated in
//! render_tests.zig), and the result is deterministic at any thread count
//! (rays are independent; reductions are per-ray, in spec order).

const std = @import("std");
const config = @import("../config.zig");
const render = @import("render.zig");
const series = @import("series.zig");

// ---- ray plan --------------------------------------------------------------

/// One camera ray of a render plan: continuous image position (fx, fy) in
/// [0,W]×[0,H], the destination pixel, and the fraction of that pixel's
/// area this ray represents (Σ weight over a pixel = 1; the weights are
/// powers of 1/4, so the sum is exact).
pub const RaySpec = struct {
    pix: u32,
    fx: f64,
    fy: f64,
    weight: f64,
};

/// The uniform plan: ss×ss box-averaged rays per pixel, at exactly the
/// subpixel positions renderImage uses.
pub fn uniformPlan(allocator: std.mem.Allocator, cam: *const render.Camera) ![]RaySpec {
    const ssn = @max(cam.ss, 1);
    const ssf: f64 = @floatFromInt(ssn);
    const w = 1.0 / (ssf * ssf);
    const specs = try allocator.alloc(RaySpec, cam.width * cam.height * ssn * ssn);
    var i: usize = 0;
    for (0..cam.height) |py| {
        for (0..cam.width) |px| {
            for (0..ssn) |sy| {
                for (0..ssn) |sx| {
                    specs[i] = .{
                        .pix = @intCast(py * cam.width + px),
                        .fx = @as(f64, @floatFromInt(px)) + (@as(f64, @floatFromInt(sx)) + 0.5) / ssf,
                        .fy = @as(f64, @floatFromInt(py)) + (@as(f64, @floatFromInt(sy)) + 0.5) / ssf,
                        .weight = w,
                    };
                    i += 1;
                }
            }
        }
    }
    return specs;
}

// ---- options / stats -------------------------------------------------------

pub const SlowOpts = struct {
    /// coordinate time of arrival at the camera. The CLI sets
    /// t_cam = t_obs + r_cam, so t_obs reads as "the sim time imaged at
    /// the hole" (the flat-space travel time is compensated; only delay
    /// DIFFERENCES across the image are physical).
    t_cam: f64,
    /// radius outside which the flow is sampled from the static reference
    /// frame: inside, from the time-interpolated series
    r_slow: f64 = 40.0,
    /// per-spec arrival times (parallel to `specs`), overriding t_cam ray
    /// by ray. This is how ONE sweep serves MANY observation epochs (a
    /// light curve or movie): specs of all epochs; each epoch aimed at
    /// its own pixel range; advance through the same sliding window, so
    /// the series still streams exactly once. x⁰ stays monotone per ray;
    /// the window logic never assumed a shared arrival time.
    t_cam_of: ?[]const f64 = null,
};

pub const Stats = struct {
    rays: usize = 0,
    /// rays that entered the slow zone
    entered: usize = 0,
    /// rays that left it again and finished on the reference frame
    tails: usize = 0,
    /// sweep window slides
    rounds: usize = 0,
    /// rays still active when the series ran out (finished on the held
    /// first frame)
    past_start: usize = 0,
    /// samples clamped to the series edges (WindowSampler counters)
    clamped_above: u64 = 0,
    clamped_below: u64 = 0,
    captured: usize = 0,
    escaped: usize = 0,
    thick: usize = 0,
    exhausted: usize = 0,
};

// ---- pause controllers -----------------------------------------------------

const EnterSlow = struct {
    r_slow: f64,
    pub inline fn pause(self: EnterSlow, _: *const render.Scene, _: *const render.RayState, r: f64) bool {
        return r < self.r_slow;
    }
};

const SweepPause = struct {
    r_slow: f64,
    t_lo: f64,
    pub inline fn pause(self: SweepPause, _: *const render.Scene, st: *const render.RayState, r: f64) bool {
        return st.x[0] < self.t_lo or r > self.r_slow;
    }
};

const ExitSlow = struct {
    r_slow: f64,
    pub inline fn pause(self: ExitSlow, _: *const render.Scene, _: *const render.RayState, r: f64) bool {
        return r > self.r_slow;
    }
};

// ---- the sweep -------------------------------------------------------------

const chunk = 64;

/// Render `specs` through the frame series into `out` (overwritten; must
/// cover every spec's pix; one width×height image normally, epochs×that
/// for a batched light curve/movie via SlowOpts.t_cam_of). `source` must
/// provide times()/acquire()/release() (see series.FileSource /
/// SliceSource); times ascend. Scene.data must be the REFERENCE frame
/// (the static-zone snapshot) and stay acquired by the caller for the
/// duration. Deterministic at any `nthreads`.
pub fn renderSlow(
    comptime cfg: config.Config,
    allocator: std.mem.Allocator,
    s: *const render.Scene,
    cam: *const render.Camera,
    source: anytype,
    specs: []const RaySpec,
    out: []f64,
    opts: render.TraceOpts,
    sopts: SlowOpts,
    nthreads: usize,
    progress: bool,
) !Stats {
    std.debug.assert(out.len >= cam.width * cam.height);
    std.debug.assert(sopts.r_slow > s.r_capture and sopts.r_slow < cam.r);
    if (sopts.t_cam_of) |tc| std.debug.assert(tc.len == specs.len);
    var stats = Stats{ .rays = specs.len };

    const states = try allocator.alloc(render.RayState, specs.len);
    defer allocator.free(states);
    for (states, specs, 0..) |*st, sp, i| {
        var x0 = cam.x0;
        x0[0] = if (sopts.t_cam_of) |tc| tc[i] else sopts.t_cam;
        st.* = .{ .x = x0, .k = cam.rayAt(sp.fx, sp.fy) };
    }

    // -- phase A: free flight to the slow zone (reference frame) --
    {
        const Ctx = struct {
            s: *const render.Scene,
            states: []render.RayState,
            next: *std.atomic.Value(usize),
            opts: render.TraceOpts,
            r_slow: f64,
            fn run(c: *const @This()) void {
                const ctl = EnterSlow{ .r_slow = c.r_slow };
                while (true) {
                    const base = c.next.fetchAdd(chunk, .monotonic);
                    if (base >= c.states.len) return;
                    for (c.states[base..@min(base + chunk, c.states.len)]) |*st| {
                        render.advanceRay(cfg, c.s, render.SnapshotSampler{}, st, c.opts, ctl);
                    }
                }
            }
        };
        var next: std.atomic.Value(usize) = .init(0);
        const ctx = Ctx{ .s = s, .states = states, .next = &next, .opts = opts, .r_slow = sopts.r_slow };
        spawnRun(Ctx, &ctx, nthreads);
    }

    // active = rays paused at the slow-zone boundary, in spec order
    var active: std.ArrayList(u32) = .empty;
    defer active.deinit(allocator);
    var tails: std.ArrayList(u32) = .empty;
    defer tails.deinit(allocator);
    for (states, 0..) |*st, i| {
        if (st.status == .active) try active.append(allocator, @intCast(i));
    }
    stats.entered = active.items.len;

    // -- phase B: the time-ordered sweep --
    const times = source.times();
    if (active.items.len > 0 and times.len >= 2) {
        // fast-forward the window top: skip pairs entirely above every ray
        var t_max = -std.math.inf(f64);
        for (active.items) |i| t_max = @max(t_max, states[i].x[0]);
        var hi_idx: usize = times.len - 1;
        while (hi_idx >= 2 and times[hi_idx - 1] > t_max) hi_idx -= 1;

        var hi_data = try source.acquire(hi_idx);
        while (active.items.len > 0) {
            const lo_idx = hi_idx - 1;
            const lo_data = try source.acquire(lo_idx);
            const t_lo = times[lo_idx];
            const t_hi = times[hi_idx];
            stats.rounds += 1;
            if (progress) {
                std.debug.print("  sweep window [{d:.2}, {d:.2}] M  active={d}\n", .{ t_lo, t_hi, active.items.len });
            }

            {
                const Ctx = struct {
                    s: *const render.Scene,
                    states: []render.RayState,
                    list: []const u32,
                    next: *std.atomic.Value(usize),
                    opts: render.TraceOpts,
                    r_slow: f64,
                    lo: *const render.DumpData,
                    hi: *const render.DumpData,
                    t_lo: f64,
                    t_hi: f64,
                    above: *std.atomic.Value(u64),
                    below: *std.atomic.Value(u64),
                    fn run(c: *const @This()) void {
                        var smp = series.WindowSampler{ .lo = c.lo, .hi = c.hi, .t_lo = c.t_lo, .t_hi = c.t_hi };
                        const ctl = SweepPause{ .r_slow = c.r_slow, .t_lo = c.t_lo };
                        while (true) {
                            const base = c.next.fetchAdd(chunk, .monotonic);
                            if (base >= c.list.len) break;
                            for (c.list[base..@min(base + chunk, c.list.len)]) |idx| {
                                render.advanceRay(cfg, c.s, &smp, &c.states[idx], c.opts, ctl);
                            }
                        }
                        _ = c.above.fetchAdd(smp.n_above, .monotonic);
                        _ = c.below.fetchAdd(smp.n_below, .monotonic);
                    }
                };
                var next: std.atomic.Value(usize) = .init(0);
                var above: std.atomic.Value(u64) = .init(0);
                var below: std.atomic.Value(u64) = .init(0);
                const ctx = Ctx{
                    .s = s,
                    .states = states,
                    .list = active.items,
                    .next = &next,
                    .opts = opts,
                    .r_slow = sopts.r_slow,
                    .lo = lo_data,
                    .hi = hi_data,
                    .t_lo = t_lo,
                    .t_hi = t_hi,
                    .above = &above,
                    .below = &below,
                };
                spawnRun(Ctx, &ctx, nthreads);
                stats.clamped_above += above.load(.monotonic);
                stats.clamped_below += below.load(.monotonic);
            }

            // classify: done / spatially out (tail) / parked on time
            var wi: usize = 0;
            for (active.items) |idx| {
                const st = &states[idx];
                if (st.status != .active) continue;
                const r = @exp(st.x[1]) + s.mp.mksr0;
                if (r > sopts.r_slow) {
                    try tails.append(allocator, idx);
                    continue;
                }
                active.items[wi] = idx;
                wi += 1;
            }
            active.shrinkRetainingCapacity(wi);

            source.release(hi_idx);
            hi_idx = lo_idx;
            hi_data = lo_data;
            if (hi_idx == 0) break;
        }

        // series exhausted with rays still inside: hold the first frame
        if (active.items.len > 0) {
            stats.past_start = active.items.len;
            const Ctx = struct {
                s: *const render.Scene,
                states: []render.RayState,
                list: []const u32,
                next: *std.atomic.Value(usize),
                opts: render.TraceOpts,
                r_slow: f64,
                first: *const render.DumpData,
                t0: f64,
                fn run(c: *const @This()) void {
                    // pointer-equal pair → plain snapshot sampling of frame 0
                    var smp = series.WindowSampler{ .lo = c.first, .hi = c.first, .t_lo = c.t0, .t_hi = c.t0 };
                    const ctl = ExitSlow{ .r_slow = c.r_slow };
                    while (true) {
                        const base = c.next.fetchAdd(chunk, .monotonic);
                        if (base >= c.list.len) return;
                        for (c.list[base..@min(base + chunk, c.list.len)]) |idx| {
                            render.advanceRay(cfg, c.s, &smp, &c.states[idx], c.opts, ctl);
                        }
                    }
                }
            };
            var next: std.atomic.Value(usize) = .init(0);
            const ctx = Ctx{
                .s = s,
                .states = states,
                .list = active.items,
                .next = &next,
                .opts = opts,
                .r_slow = sopts.r_slow,
                .first = hi_data,
                .t0 = times[0],
            };
            spawnRun(Ctx, &ctx, nthreads);
            for (active.items) |idx| {
                if (states[idx].status == .active) try tails.append(allocator, idx);
            }
            active.shrinkRetainingCapacity(0);
        }
        source.release(hi_idx);
    } else if (active.items.len > 0) {
        // single-frame series: the "slow" zone is that frame, statically
        for (active.items) |idx| try tails.append(allocator, idx);
        active.shrinkRetainingCapacity(0);
    }

    // -- phase C: frozen tails on the reference frame --
    stats.tails = tails.items.len;
    if (tails.items.len > 0) {
        const Ctx = struct {
            s: *const render.Scene,
            states: []render.RayState,
            list: []const u32,
            next: *std.atomic.Value(usize),
            opts: render.TraceOpts,
            fn run(c: *const @This()) void {
                while (true) {
                    const base = c.next.fetchAdd(chunk, .monotonic);
                    if (base >= c.list.len) return;
                    for (c.list[base..@min(base + chunk, c.list.len)]) |idx| {
                        render.advanceRay(cfg, c.s, render.SnapshotSampler{}, &c.states[idx], c.opts, render.NeverPause{});
                    }
                }
            }
        };
        var next: std.atomic.Value(usize) = .init(0);
        const ctx = Ctx{ .s = s, .states = states, .list = tails.items, .next = &next, .opts = opts };
        spawnRun(Ctx, &ctx, nthreads);
    }

    // -- reduce (sequential, spec order: deterministic at any thread count) --
    @memset(out, 0);
    for (specs, states) |sp, *st| {
        var intensity = st.intensity;
        if (opts.screen) intensity = if (st.status == .captured) 0.0 else 1.0;
        out[sp.pix] += sp.weight * intensity;
        switch (st.status) {
            .captured => stats.captured += 1,
            .escaped => stats.escaped += 1,
            .thick => stats.thick += 1,
            .exhausted => stats.exhausted += 1,
            .active => unreachable, // every phase runs to terminal or hands off
        }
    }
    return stats;
}

/// Work-stealing team in renderImage's style: spawn nthreads−1 workers, the
/// caller participates, failed spawns just narrow the team.
fn spawnRun(comptime Ctx: type, ctx: *const Ctx, nthreads: usize) void {
    var threads: [63]std.Thread = undefined;
    const want: usize = @min(@max(nthreads, 1) - 1, threads.len);
    var spawned: usize = 0;
    for (0..want) |i| {
        threads[i] = std.Thread.spawn(.{}, Ctx.run, .{ctx}) catch break;
        spawned = i + 1;
    }
    Ctx.run(ctx);
    for (threads[0..spawned]) |t| t.join();
}
