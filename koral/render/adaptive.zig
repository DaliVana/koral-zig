//! Adaptive image-plane refinement around the critical curve.
//!
//! Uniform pixels undersample the photon ring badly: the n-th lensed
//! subring is exponentially demagnified (width ∝ e^{−γn}, γ ≈ π at a = 0),
//! so at typical fov/size the n ≥ 2 structure is far sub-pixel. Refining
//! blindly is wasteful; the decisive observation is that geodesics are
//! FLUID-INDEPENDENT: a vacuum pre-pass (opts.screen tracing, no sampling)
//! maps exactly where the image needs work before any radiative transfer
//! is spent.
//!
//! Per pre-pass ray we record (captured?, flight time). Two neighbors
//! straddling the critical curve disagree in capture; two neighbors one
//! subring apart differ in flight time by ~one photon half-orbit (~16 M at
//! a = 0; also precisely the delay structure slow light resolves, which
//! is why the refinement criterion is a TIME threshold). Pixels whose four
//! corners disagree are subdivided as a quadtree: cells whose own corners
//! agree stop early, so the work concentrates in an O(2^d) band along the
//! curve, not O(4^d) over the pixel.
//!
//! The output is a flat []RaySpec plan (weights are exact powers of 1/4
//! summing to 1 per pixel) consumed either by the fast-light renderPlan
//! below or by the slow-light sweep; the pre-pass is time-independent
//! (stationary metric), so one plan serves both.

const std = @import("std");
const config = @import("../config.zig");
const render = @import("render.zig");
const sweep = @import("sweep.zig");

pub const RaySpec = sweep.RaySpec;

pub const PlanOpts = struct {
    /// max quadtree depth per pixel (0 = uniform plan everywhere)
    depth: usize = 0,
    /// corner flight-time disagreement that triggers refinement [M].
    /// One photon half-orbit is ~16 M; 5 M also catches the strong
    /// Shapiro-delay gradient hugging the curve.
    dt_thresh: f64 = 5.0,
    /// step budget for PRE-PASS probes only (RT rays keep opts.max_steps).
    /// A probe that cannot resolve capture-vs-escape within this budget is
    /// pinned to the photon shell; it reports a huge flight time and no
    /// capture, which is precisely the "critical, refine here" signal, so
    /// letting it orbit for the full RT budget buys nothing.
    probe_max_steps: usize = 5_000,
};

pub const Plan = struct {
    specs: []RaySpec,
    /// per-pixel refinement mask (diagnostics/tests)
    marked: []bool,
    /// vacuum pre-pass rays traced (corner grid + refinement probes)
    vacuum_rays: usize,

    pub fn deinit(self: *Plan, allocator: std.mem.Allocator) void {
        allocator.free(self.specs);
        allocator.free(self.marked);
        self.* = undefined;
    }
};

const Corner = struct { captured: bool, t_flight: f64 };

fn vacuumRay(comptime cfg: config.Config, s: *const render.Scene, cam: *const render.Camera, fx: f64, fy: f64, opts: render.TraceOpts) Corner {
    var o = opts;
    o.screen = true;
    const res = render.traceRay(cfg, s, cam.x0, cam.rayAt(fx, fy), o);
    // Correct the endpoint time to the exact capture/escape sphere: the
    // final step OVERSHOOTS the stop radius by up to one step (tens of M
    // of coordinate time at coarse eps in the log-r outskirts), which
    // would drown the few-M flight-time signal in per-ray jitter. First
    // order in the endpoint k (same trick as the timing gates); skipped
    // for exhausted rays parked mid-field (their t_flight is huge and
    // "critical" regardless).
    var t_end = res.x[0];
    const r_end = @exp(res.x[1]) + s.mp.mksr0;
    const target = if (res.captured) s.r_capture else s.r_escape;
    const drdl = (r_end - s.mp.mksr0) * res.k[1]; // dr/dλ
    if (@abs(target - r_end) < 0.5 * r_end and @abs(drdl) > 1e-30) {
        t_end += (target - r_end) * res.k[0] / drdl;
    }
    return .{ .captured = res.captured, .t_flight = cam.x0[0] - t_end };
}

/// Corners in cell order [00, 10, 01, 11] (x fastest). Disagreement =
/// mixed capture, or a flight-time spread above the threshold among
/// like-fated corners.
fn disagree(c: [4]Corner, dt_thresh: f64) bool {
    for (c[1..]) |ci| {
        if (ci.captured != c[0].captured) return true;
    }
    var lo = c[0].t_flight;
    var hi = c[0].t_flight;
    for (c[1..]) |ci| {
        lo = @min(lo, ci.t_flight);
        hi = @max(hi, ci.t_flight);
    }
    return hi - lo > dt_thresh;
}

/// Build the refinement plan: a (W+1)×(H+1) vacuum corner grid marks the
/// pixels whose corners disagree; unmarked pixels get the uniform ss×ss
/// box, marked pixels a DFS quadtree of RT rays (leaf = cell whose corners
/// agree or depth == max, one ray at the cell center, weight = its area).
/// Deterministic at any nthreads: pixels are independent, and the plan is
/// gathered in pixel order regardless of which worker produced it.
pub fn planRays(
    comptime cfg: config.Config,
    allocator: std.mem.Allocator,
    s: *const render.Scene,
    cam: *const render.Camera,
    opts: render.TraceOpts,
    popts: PlanOpts,
    nthreads: usize,
) !Plan {
    const w = cam.width;
    const h = cam.height;
    const npix = w * h;
    const marked = try allocator.alloc(bool, npix);
    errdefer allocator.free(marked);

    if (popts.depth == 0) {
        @memset(marked, false);
        return .{ .specs = try sweep.uniformPlan(allocator, cam), .marked = marked, .vacuum_rays = 0 };
    }

    // probes classify, they don't do transfer — cap their step budget
    var probe_opts = opts;
    probe_opts.max_steps = @min(opts.max_steps, popts.probe_max_steps);

    // -- vacuum corner grid, threaded by row --
    const corners = try allocator.alloc(Corner, (w + 1) * (h + 1));
    defer allocator.free(corners);
    {
        const Ctx = struct {
            s: *const render.Scene,
            cam: *const render.Camera,
            corners: []Corner,
            next: *std.atomic.Value(usize),
            opts: render.TraceOpts,
            w: usize,
            h: usize,
            fn run(c: *const @This()) void {
                while (true) {
                    const cy = c.next.fetchAdd(1, .monotonic);
                    if (cy > c.h) return;
                    for (0..c.w + 1) |cx| {
                        c.corners[cy * (c.w + 1) + cx] = vacuumRay(cfg, c.s, c.cam, @floatFromInt(cx), @floatFromInt(cy), c.opts);
                    }
                }
            }
        };
        var next: std.atomic.Value(usize) = .init(0);
        const ctx = Ctx{ .s = s, .cam = cam, .corners = corners, .next = &next, .opts = probe_opts, .w = w, .h = h };
        spawnRun(Ctx, &ctx, nthreads);
    }

    for (0..h) |py| {
        for (0..w) |px| {
            marked[py * w + px] = disagree(pixelCorners(corners, w, px, py), popts.dt_thresh);
        }
    }

    // -- per-pixel spec generation (threaded; deterministic gather) --
    const PerPix = struct { worker: u16, off: u32, len: u32 };
    const perpix = try allocator.alloc(PerPix, npix);
    defer allocator.free(perpix);

    const max_workers = 64;
    var lists: [max_workers]std.ArrayList(RaySpec) = @splat(.empty);
    defer for (&lists) |*l| l.deinit(allocator);
    var probes: [max_workers]usize = @splat(0);

    {
        const Ctx = struct {
            s: *const render.Scene,
            cam: *const render.Camera,
            corners: []const Corner,
            marked: []const bool,
            perpix: []PerPix,
            lists: []std.ArrayList(RaySpec),
            probes: []usize,
            wslot: std.atomic.Value(usize) = .init(0),
            next: std.atomic.Value(usize) = .init(0),
            allocator: std.mem.Allocator,
            opts: render.TraceOpts,
            popts: PlanOpts,
            w: usize,
            h: usize,
            err: std.atomic.Value(bool) = .init(false),

            fn run(c: *@This()) void {
                const slot = c.wslot.fetchAdd(1, .monotonic);
                const list = &c.lists[slot];
                const ssn = @max(c.cam.ss, 1);
                const ssf: f64 = @floatFromInt(ssn);
                while (true) {
                    const pix = c.next.fetchAdd(1, .monotonic);
                    if (pix >= c.w * c.h) return;
                    const px = pix % c.w;
                    const py = pix / c.w;
                    const off: u32 = @intCast(list.items.len);
                    if (!c.marked[pix]) {
                        for (0..ssn) |sy| {
                            for (0..ssn) |sx| {
                                list.append(c.allocator, .{
                                    .pix = @intCast(pix),
                                    .fx = @as(f64, @floatFromInt(px)) + (@as(f64, @floatFromInt(sx)) + 0.5) / ssf,
                                    .fy = @as(f64, @floatFromInt(py)) + (@as(f64, @floatFromInt(sy)) + 0.5) / ssf,
                                    .weight = 1.0 / (ssf * ssf),
                                }) catch {
                                    c.err.store(true, .monotonic);
                                    return;
                                };
                            }
                        }
                    } else {
                        c.subdivide(list, slot, @intCast(pix), @floatFromInt(px), @floatFromInt(py), 1.0, pixelCorners(c.corners, c.w, px, py), 0) catch {
                            c.err.store(true, .monotonic);
                            return;
                        };
                    }
                    c.perpix[pix] = .{ .worker = @intCast(slot), .off = off, .len = @intCast(list.items.len - off) };
                }
            }

            fn subdivide(c: *@This(), list: *std.ArrayList(RaySpec), slot: usize, pix: u32, x0: f64, y0: f64, sz: f64, cs: [4]Corner, depth: usize) !void {
                if (depth >= c.popts.depth or !disagree(cs, c.popts.dt_thresh)) {
                    try list.append(c.allocator, .{
                        .pix = pix,
                        .fx = x0 + 0.5 * sz,
                        .fy = y0 + 0.5 * sz,
                        .weight = sz * sz,
                    });
                    return;
                }
                const hs = 0.5 * sz;
                const xm = x0 + hs;
                const ym = y0 + hs;
                const cn = vacuumRay(cfg, c.s, c.cam, xm, y0, c.opts);
                const cw = vacuumRay(cfg, c.s, c.cam, x0, ym, c.opts);
                const cc = vacuumRay(cfg, c.s, c.cam, xm, ym, c.opts);
                const ce = vacuumRay(cfg, c.s, c.cam, x0 + sz, ym, c.opts);
                const cs_ = vacuumRay(cfg, c.s, c.cam, xm, y0 + sz, c.opts);
                c.probes[slot] += 5;
                try c.subdivide(list, slot, pix, x0, y0, hs, .{ cs[0], cn, cw, cc }, depth + 1);
                try c.subdivide(list, slot, pix, xm, y0, hs, .{ cn, cs[1], cc, ce }, depth + 1);
                try c.subdivide(list, slot, pix, x0, ym, hs, .{ cw, cc, cs[2], cs_ }, depth + 1);
                try c.subdivide(list, slot, pix, xm, ym, hs, .{ cc, ce, cs_, cs[3] }, depth + 1);
            }
        };
        var ctx = Ctx{
            .s = s,
            .cam = cam,
            .corners = corners,
            .marked = marked,
            .perpix = perpix,
            .lists = lists[0..],
            .probes = probes[0..],
            .allocator = allocator,
            .opts = probe_opts,
            .popts = popts,
            .w = w,
            .h = h,
        };
        spawnRunMut(Ctx, &ctx, @min(nthreads, max_workers));
        if (ctx.err.load(.monotonic)) return error.OutOfMemory;
    }

    // gather in pixel order — deterministic regardless of worker scheduling
    var total: usize = 0;
    for (perpix) |pp| total += pp.len;
    const specs = try allocator.alloc(RaySpec, total);
    errdefer allocator.free(specs);
    var wi: usize = 0;
    for (perpix) |pp| {
        const src = lists[pp.worker].items[pp.off..][0..pp.len];
        @memcpy(specs[wi..][0..src.len], src);
        wi += src.len;
    }

    var nprobe: usize = (w + 1) * (h + 1);
    for (probes) |p| nprobe += p;
    return .{ .specs = specs, .marked = marked, .vacuum_rays = nprobe };
}

inline fn pixelCorners(corners: []const Corner, w: usize, px: usize, py: usize) [4]Corner {
    return .{
        corners[py * (w + 1) + px],
        corners[py * (w + 1) + px + 1],
        corners[(py + 1) * (w + 1) + px],
        corners[(py + 1) * (w + 1) + px + 1],
    };
}

/// Fast-light render of a ray plan: traceRay per spec, then a sequential
/// spec-order reduction into pixels (deterministic at any thread count).
pub fn renderPlan(
    comptime cfg: config.Config,
    allocator: std.mem.Allocator,
    s: *const render.Scene,
    cam: *const render.Camera,
    specs: []const RaySpec,
    out: []f64,
    opts: render.TraceOpts,
    nthreads: usize,
) !void {
    std.debug.assert(out.len == cam.width * cam.height);
    const results = try allocator.alloc(f64, specs.len);
    defer allocator.free(results);
    {
        const Ctx = struct {
            s: *const render.Scene,
            cam: *const render.Camera,
            specs: []const RaySpec,
            results: []f64,
            next: *std.atomic.Value(usize),
            opts: render.TraceOpts,
            fn run(c: *const @This()) void {
                const chunk = 16;
                while (true) {
                    const base = c.next.fetchAdd(chunk, .monotonic);
                    if (base >= c.specs.len) return;
                    for (base..@min(base + chunk, c.specs.len)) |i| {
                        const sp = c.specs[i];
                        c.results[i] = render.traceRay(cfg, c.s, c.cam.x0, c.cam.rayAt(sp.fx, sp.fy), c.opts).intensity;
                    }
                }
            }
        };
        var next: std.atomic.Value(usize) = .init(0);
        const ctx = Ctx{ .s = s, .cam = cam, .specs = specs, .results = results, .next = &next, .opts = opts };
        spawnRun(Ctx, &ctx, nthreads);
    }
    @memset(out, 0);
    for (specs, results) |sp, ri| out[sp.pix] += sp.weight * ri;
}

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

fn spawnRunMut(comptime Ctx: type, ctx: *Ctx, nthreads: usize) void {
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
