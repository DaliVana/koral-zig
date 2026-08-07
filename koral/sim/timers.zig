//! P0 of the parallelization plan (docs/PARALLELIZATION_PLAN_2026-07-06.md
//! §7): cheap wall-clock instrumentation of every per-step pass in
//! sim.step(), reported at the driver's scalar cadence.
//!
//! Attribution is self-time: begin()/end() keep a small pass stack and
//! charge elapsed time to the innermost active pass only, so nested timed
//! calls (op_implicit → cell_fixup, apply_dynamo → set_bc/calc_u2p,
//! op_explicit → wavespeeds/sweep/…/u2p) never double-count. The sum over
//! all buckets equals the wall time spent inside step(); the `.step`
//! bucket's own share is the untimed remainder (save_timesteps,
//! copy_entropycount, inter-pass gaps) and prints as "(other)".
//!
//! Cost: one monotonic clock read per begin/end, ~60 pairs per RK2IMEX
//! step — microseconds against millisecond-scale passes, and no effect on
//! any FP result. Always on; printing/resetting is the driver's choice.

const std = @import("std");
const builtin = @import("builtin");

/// One bucket per pass of the plan's §1 inventory (declaration order is
/// roughly step order; reports print in this order for stable rows).
pub const Pass = enum {
    step, // whole step(); self time = untimed remainder, printed last
    radvisc, // calc_Rij_visc_total
    implicit, // op_implicit inversion sweep (minus nested fixup)
    u2p, // calc_u2p inversion sweep (minus nested fixup/bc)
    fixup, // cell_fixup, all flavors
    bc, // set_bc incl. corner fill
    halo, // MPI halo exchange episodes (Startall+Waitall wait time)
    collect, // MPI collectives (end-of-step dt fold, scale-height sums)
    correct, // correct_polaraxis
    wavespeeds, // calc_wavespeeds + save_wavespeeds (+ dt reduce)
    sweep, // reconstruction + one-sided fluxes, all dims
    fluxes, // f_calc_fluxes_at_faces (LAXF/HLL combine)
    fluxct, // flux-CT EMF averaging
    update, // conserved update + metric source
    dynamo, // apply_dynamo (minus nested set_bc/calc_u2p)
    stage, // RK2IMEX stage arithmetic (full-field copies/derivs/combines)
    entropy, // update_entropy
};

pub const n_passes = @typeInfo(Pass).@"enum".fields.len;

/// Monotonic wall clock in nanoseconds. Linux goes through the syscall
/// wrapper (no libc needed); everywhere else through std.c — fine on
/// Darwin, which always links libSystem.
pub fn nowNs() u64 {
    switch (builtin.os.tag) {
        .linux => {
            var ts: std.os.linux.timespec = undefined;
            _ = std.os.linux.clock_gettime(.MONOTONIC, &ts);
            return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
        },
        else => {
            var ts: std.c.timespec = undefined;
            _ = std.c.clock_gettime(.MONOTONIC, &ts);
            return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
        },
    }
}

pub const PassTimers = struct {
    /// accumulated self time per pass
    ns: [n_passes]u64 = @splat(0),
    calls: [n_passes]u64 = @splat(0),
    /// active-pass stack (main thread only; workers never touch timers)
    stack: [8]Pass = undefined,
    depth: usize = 0,
    /// timestamp of the last begin/end transition
    mark: u64 = 0,

    pub fn begin(self: *PassTimers, pass: Pass) void {
        self.beginAt(pass, nowNs());
    }

    pub fn end(self: *PassTimers) void {
        self.endAt(nowNs());
    }

    fn beginAt(self: *PassTimers, pass: Pass, now: u64) void {
        if (self.depth > 0) {
            self.ns[@intFromEnum(self.stack[self.depth - 1])] += now - self.mark;
        }
        std.debug.assert(self.depth < self.stack.len);
        self.stack[self.depth] = pass;
        self.depth += 1;
        self.calls[@intFromEnum(pass)] += 1;
        self.mark = now;
    }

    fn endAt(self: *PassTimers, now: u64) void {
        std.debug.assert(self.depth > 0);
        self.depth -= 1;
        self.ns[@intFromEnum(self.stack[self.depth])] += now - self.mark;
        self.mark = now;
    }

    /// Zero the accumulators (call between steps, i.e. at depth 0).
    pub fn reset(self: *PassTimers) void {
        std.debug.assert(self.depth == 0);
        self.ns = @splat(0);
        self.calls = @splat(0);
    }

    /// Print the per-pass table for the steps since the last reset().
    /// Rows are in Pass declaration order (stable across reports), the
    /// `.step` self-time remainder last as "(other)".
    pub fn printReport(self: *const PassTimers) void {
        std.debug.assert(self.depth == 0);
        const steps = self.calls[@intFromEnum(Pass.step)];
        if (steps == 0) return;
        const fsteps: f64 = @floatFromInt(steps);

        var total: u64 = 0;
        for (self.ns) |v| total += v;
        const ftotal: f64 = @floatFromInt(total);

        std.debug.print(
            "timings: {d} steps, {d:.3} s stepping, {d:.1} ms/step\n",
            .{ steps, ftotal / 1e9, ftotal / 1e6 / fsteps },
        );
        // indices 1..n first (declaration order), the .step remainder last
        for (1..n_passes + 1) |k| {
            const i = k % n_passes;
            if (self.calls[i] == 0) continue;
            const name = if (i == 0) "(other)" else @tagName(@as(Pass, @enumFromInt(i)));
            const p_ns: f64 = @floatFromInt(self.ns[i]);
            std.debug.print(
                "  {s:<10} {d:8.2} ms/step {d:5.1}%  x{d:.1}/step\n",
                .{
                    name,
                    p_ns / 1e6 / fsteps,
                    100.0 * p_ns / ftotal,
                    @as(f64, @floatFromInt(self.calls[i])) / fsteps,
                },
            );
        }
    }
};

test "self-time attribution across nested passes" {
    var t = PassTimers{};
    // step ── u2p ── fixup, with gaps attributed to the innermost pass
    t.beginAt(.step, 100);
    t.beginAt(.u2p, 110); // step self += 10
    t.beginAt(.fixup, 130); // u2p self += 20
    t.endAt(160); // fixup self += 30
    t.endAt(170); // u2p self += 10
    t.beginAt(.stage, 175); // step self += 5
    t.endAt(195); // stage self += 20
    t.endAt(200); // step self += 5
    try std.testing.expectEqual(@as(u64, 20), t.ns[@intFromEnum(Pass.step)]);
    try std.testing.expectEqual(@as(u64, 30), t.ns[@intFromEnum(Pass.u2p)]);
    try std.testing.expectEqual(@as(u64, 30), t.ns[@intFromEnum(Pass.fixup)]);
    try std.testing.expectEqual(@as(u64, 20), t.ns[@intFromEnum(Pass.stage)]);
    // buckets sum exactly to the step wall time
    var total: u64 = 0;
    for (t.ns) |v| total += v;
    try std.testing.expectEqual(@as(u64, 100), total);
    try std.testing.expectEqual(@as(u64, 1), t.calls[@intFromEnum(Pass.step)]);
    try std.testing.expectEqual(@as(u64, 1), t.calls[@intFromEnum(Pass.u2p)]);
    try std.testing.expectEqual(@as(usize, 0), t.depth);

    // a second step accumulates on top; reset() zeroes
    t.beginAt(.step, 300);
    t.endAt(340);
    try std.testing.expectEqual(@as(u64, 60), t.ns[@intFromEnum(Pass.step)]);
    try std.testing.expectEqual(@as(u64, 2), t.calls[@intFromEnum(Pass.step)]);
    t.reset();
    try std.testing.expectEqual(@as(u64, 0), t.ns[@intFromEnum(Pass.step)]);
    try std.testing.expectEqual(@as(u64, 0), t.calls[@intFromEnum(Pass.u2p)]);
}

test "top-level passes outside a step count standalone" {
    // init-time calls (set_bc, wavespeeds) run at depth 0: the gap between
    // them must not be attributed to anything.
    var t = PassTimers{};
    t.beginAt(.bc, 1000);
    t.endAt(1100);
    t.beginAt(.wavespeeds, 5000); // 3900 ns gap at depth 0 — uncounted
    t.endAt(5500);
    try std.testing.expectEqual(@as(u64, 100), t.ns[@intFromEnum(Pass.bc)]);
    try std.testing.expectEqual(@as(u64, 500), t.ns[@intFromEnum(Pass.wavespeeds)]);
    var total: u64 = 0;
    for (t.ns) |v| total += v;
    try std.testing.expectEqual(@as(u64, 600), total);
}

test "nowNs is monotonic and nonzero" {
    const a = nowNs();
    const b = nowNs();
    try std.testing.expect(a > 0);
    try std.testing.expect(b >= a);
}
