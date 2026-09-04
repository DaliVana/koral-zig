//! The generic run driver: everything a problem executable does that is not
//! about the problem. CLI parsing, params, the MPI world and φ-ring,
//! decomposition, Sim construction, restart, the RK2IMEX loop, the
//! heartbeat, scalar/KDMP/Silo output at the configured cadence, and the
//! NaN abort. Extracted verbatim from problems/puffy/main.zig (sim.zig
//! redesign step 1, 2026-09-04) so a second problem does not copy it.
//!
//! A problem hands `run` a comptime type `P` declaring:
//!
//!   pub const name: []const u8            log prefix and Silo file stem
//!   pub const cfg: config.Config          the comptime physics/coords
//!   pub const default_params: []const u8  params path when none is given
//!   pub const scalar_radii: ScalarRadii   r_lum / r_scale for scalars.dat
//!   pub const Physics: type               the problem's value type
//!   pub fn setup(comptime SimT, allocator, io, p: *const Params) !Setup
//!       Setup has .physics, .grid_global, .options, and deinit(allocator).
//!   pub fn initAll(comptime SimT, sim: *SimT, phys: *const Physics) !f64
//!       Fill the domain (init + postinit); returns the β-normalization fac.
//!   pub fn banner(comptime SimT, phys, p, grid_global) void   (optional)
//!       Problem-specific startup notes; printed on rank 0 only.
//!   pub fn afterInit(comptime SimT, sim, phys, is_root) !void  (optional)
//!       Runs after a fresh init (not after a restart), e.g. seed diagnostics.
//!   pub fn reportSetupError(err, p) void                       (optional)
//!       Friendlier text for a failed setup before the error propagates.
//!
//! CLI:  <problem> [params.toml] [--restart <file|dir>]
//!   --restart FILE  continue from that checkpoint;
//!   --restart DIR   continue from the newest prims#####.kdmp in DIR;
//!   (omitted)       start from scratch (C: NORESTART).

const std = @import("std");
const builtin = @import("builtin");
const sim_mod = @import("sim.zig");
const params_mod = @import("params.zig");
const dump = @import("io/dump.zig");
const silo = @import("io/silo.zig");
const comm_mod = @import("comm/comm.zig");
const decomp_mod = @import("comm/decomp.zig");

pub const ScalarRadii = struct { lum: f64, scale: f64 };

const heartbeat_interval_ns: u64 = 1_000_000_000;

fn writeScalars(comptime name: []const u8, io: std.Io, out_dir: []const u8, log: []const u8) void {
    var buf: [1024]u8 = undefined;
    const path = std.fmt.bufPrint(&buf, "{s}/scalars.dat", .{out_dir}) catch return;
    std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = log }) catch |err| {
        std.debug.print(name ++ ": cannot write {s}: {s}\n", .{ path, @errorName(err) });
    };
}

fn writeSiloDump(comptime SimT: type, comptime name: []const u8, io: std.Io, out_dir: []const u8, allocator: std.mem.Allocator, s: *const SimT, idx: usize) void {
    if (comptime !silo.enabled) return;
    var dbuf: [1024]u8 = undefined;
    const dir = std.fmt.bufPrint(&dbuf, "{s}/silo", .{out_dir}) catch return;
    std.Io.Dir.cwd().createDirPath(io, dir) catch {};
    var buf: [1024]u8 = undefined;
    const path = std.fmt.bufPrintZ(&buf, "{s}/silo/" ++ name ++ "{d:0>4}.silo", .{ out_dir, idx }) catch return;
    const cycle: i32 = @intCast(@min(s.nstep, @as(u64, std.math.maxInt(i32))));
    silo.write(SimT, s, allocator, path, s.t, cycle, .{}) catch |err| {
        std.debug.print(name ++ ": cannot write {s}: {s}\n", .{ path, @errorName(err) });
    };
}

fn writeFrame(
    comptime SimT: type,
    comptime name: []const u8,
    io: std.Io,
    allocator: std.mem.Allocator,
    s: *const SimT,
    comm: *comm_mod.Backend,
    out_dir: []const u8,
    idx: u32,
) !void {
    var pbuf: [1024]u8 = undefined;
    const path = try std.fmt.bufPrint(&pbuf, "{s}/prims{d:0>5}.kdmp", .{ out_dir, idx });
    try dump.writePrim(SimT, s, comm, allocator, io, path, idx);
    if (s.decomp.ntz <= 1) writeSiloDump(SimT, name, io, out_dir, allocator, s, idx);
}

const Heartbeat = struct {
    last_ns: u64,
    prev_fail: u64 = 0,
    prev_iters: u64 = 0,
    prev_solves: u64 = 0,
    ring_fail: u64 = 0,
};

fn countFixupFlags(comptime SimT: type, s: *const SimT) struct { hd: u64, rad: u64 } {
    const L = SimT.Layout;
    var hd: u64 = 0;
    var rad: u64 = 0;
    var iz: i64 = 0;
    while (iz < s.nzi()) : (iz += 1) {
        var iy: i64 = 0;
        while (iy < s.nyi()) : (iy += 1) {
            var ix: i64 = 0;
            while (ix < s.nxi()) : (ix += 1) {
                if (s.getFlag(.hd_fixup, ix, iy, iz) != 0) hd += 1;
                if (comptime L.hasVar(.ee)) {
                    if (s.getFlag(.rad_fixup, ix, iy, iz) != 0) rad += 1;
                }
            }
        }
    }
    return .{ .hd = hd, .rad = rad };
}

fn printHeartbeat(comptime SimT: type, s: *const SimT, dt: f64, step_wall_ns: u64, hb: *Heartbeat) void {
    const g = s.decomp.global;
    const ncells: f64 = @floatFromInt(g.nx * g.ny * g.nz);
    const wall_s = @as(f64, @floatFromInt(@max(step_wall_ns, 1))) / 1.0e9;
    const znps = ncells / wall_s;
    const tgpd = dt / wall_s * 86400.0;

    const d_fail = s.n_radimp_failures - hb.prev_fail;
    const d_iters = s.n_radimp_iters - hb.prev_iters;
    const d_solves = s.n_radimp_solves - hb.prev_solves;
    const avg_it: f64 = if (d_solves > 0)
        @as(f64, @floatFromInt(d_iters)) / @as(f64, @floatFromInt(d_solves))
    else
        0;
    const fx = countFixupFlags(SimT, s);

    if (s.decomp.ntz > 1) {
        var total_ns: u64 = 0;
        for (s.timers.ns) |v| total_ns += v;
        const comm_ns = s.timers.ns[@intFromEnum(sim_mod.Pass.halo)] +
            s.timers.ns[@intFromEnum(sim_mod.Pass.collect)];
        const mpi_pct: f64 = if (total_ns > 0)
            100.0 * @as(f64, @floatFromInt(comm_ns)) / @as(f64, @floatFromInt(total_ns))
        else
            0;
        std.debug.print(
            "st #{d:>6} t={e:.5} dt={e:.2} znps={d:.0} tgpd={e:.2} fail# {d} {d} {d}(r0) imp# {d} it {d:.1}(r0) mpi={d:.1}%\n",
            .{ s.nstep, s.t, dt, znps, tgpd, hb.ring_fail, fx.hd, fx.rad, d_solves, avg_it, mpi_pct },
        );
        hb.ring_fail = 0;
    } else {
        std.debug.print(
            "st #{d:>6} t={e:.5} dt={e:.2} znps={d:.0} tgpd={e:.2} fail# {d} {d} {d} imp# {d} it {d:.1}\n",
            .{ s.nstep, s.t, dt, znps, tgpd, d_fail, fx.hd, fx.rad, d_solves, avg_it },
        );
    }

    hb.prev_fail = s.n_radimp_failures;
    hb.prev_iters = s.n_radimp_iters;
    hb.prev_solves = s.n_radimp_solves;
}

/// The whole executable for problem `P` (see the module header for the
/// contract). Serial and MPI builds share this path: `Comm` is the
/// comptime-selected backend, a no-op ring serially.
pub fn run(comptime P: type, init: std.process.Init) !void {
    const SimT = sim_mod.Sim(P.cfg);
    const L = SimT.Layout;
    const name = P.name;
    const r_lum: f64 = P.scalar_radii.lum;
    const r_scale: f64 = P.scalar_radii.scale;

    const allocator = init.gpa;
    const io = init.io;

    const Comm = comm_mod.Backend;
    try Comm.initWorld();
    defer Comm.finalizeWorld();
    const world = Comm.worldSize();
    errdefer if (world > 1) Comm.abortWorld(1);

    var args = std.process.Args.Iterator.init(init.minimal.args);
    defer args.deinit();
    _ = args.next();
    var params_path: []const u8 = P.default_params;
    var restart_arg: ?[]const u8 = null;
    var have_params = false;
    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--restart") or std.mem.eql(u8, arg, "-r")) {
            restart_arg = args.next() orelse {
                std.debug.print(name ++ ": --restart needs a file or directory argument\n", .{});
                return error.MissingRestartArg;
            };
        } else if (std.mem.startsWith(u8, arg, "--restart=")) {
            restart_arg = arg["--restart=".len..];
        } else if (!have_params) {
            params_path = arg;
            have_params = true;
        } else {
            std.debug.print(name ++ ": unexpected extra argument '{s}'\n", .{arg});
            return error.BadArgs;
        }
    }

    var p = params_mod.Params.load(allocator, io, params_path) catch |err| {
        std.debug.print(name ++ ": cannot load params from '{s}': {s}\n", .{ params_path, @errorName(err) });
        return err;
    };
    defer p.deinit(allocator);

    var st = P.setup(SimT, allocator, io, &p) catch |err| {
        if (comptime @hasDecl(P, "reportSetupError")) P.reportSetupError(err, &p);
        return err;
    };
    defer st.deinit(allocator);
    const phys = &st.physics;

    const ntz: usize = if (p.ntz == 0) world else p.ntz;
    if (ntz != world) {
        std.debug.print(name ++ ": params ntz={d} but the job was launched with {d} rank(s) — they must agree (ntz=0 means auto)\n", .{ p.ntz, world });
        return error.RankCountMismatch;
    }
    var comm = try Comm.init(ntz);
    defer comm.deinit();
    const is_root = comm.rank() == 0;
    errdefer if (ntz > 1) comm.abortJob(1);

    const dc = decomp_mod.decompose(st.grid_global, ntz, comm.rank()) catch |err| {
        std.debug.print(
            name ++ ": cannot decompose {d}×{d}×{d} over ntz={d}: {s} (φ-only splits need nz % ntz == 0 and nz/ntz ≥ NG={d}; 2D never decomposes)\n",
            .{ p.nx, p.ny, p.nz, ntz, @errorName(err), st.grid_global.ng },
        );
        return err;
    };
    var opt = st.options;
    opt.comm = &comm;
    opt.decomp = dc;
    var s = try SimT.init(allocator, dc.local, opt);
    defer s.deinit();

    if (p.pin_threads) {
        if (s.team) |tm| {
            if (tm.pinnedWidth()) |w| {
                std.debug.print(name ++ ": rank {d}: {d} threads pinned over {d} allowed cpus\n", .{ comm.rank(), p.nthreads, w });
            } else {
                std.debug.print(name ++ ": rank {d}: pin_threads requested but not applied (non-Linux host or no affinity mask)\n", .{comm.rank()});
            }
        }
    }
    if (ntz > 1) {
        std.debug.print(
            name ++ ": rank {d}/{d} owns φ-cells [{d},{d}) of {d} — {d} threads\n",
            .{ comm.rank(), ntz, dc.tok, dc.tok + @as(i64, @intCast(dc.local.nz)), st.grid_global.nz, p.nthreads },
        );
        if (is_root) std.debug.print(
            name ++ ": MPI φ-ring of {d} ranks × {d} threads; scalars.dat via rank 0, KDMP checkpoints collective (silo via kdmp2silo)\n",
            .{ ntz, p.nthreads },
        );
    }

    if (is_root) {
        std.debug.print(
            name ++ ": NV={d} grid {d}×{d}×{d} (+{d} ghosts) | threads={d} | build={s}\n",
            .{ L.count, st.grid_global.nx, st.grid_global.ny, st.grid_global.nz, st.grid_global.ng, p.nthreads, @tagName(builtin.mode) },
        );
        if (comptime @hasDecl(P, "banner")) P.banner(SimT, phys, &p, st.grid_global);
    }
    if (is_root and builtin.mode == .Debug) {
        std.debug.print(
            name ++ ": *** WARNING: Debug build — expect a many-fold slowdown. " ++
                "Rebuild with `-Doptimize=ReleaseFast` for production runs. ***\n",
            .{},
        );
    }

    var out_idx: u32 = 0;
    var restarted = false;
    if (restart_arg) |arg| {
        const file = dump.resolveRestartPath(io, allocator, arg) catch |err| {
            if (err == error.NoRestartFile) {
                std.debug.print(name ++ ": no .kdmp checkpoint found in directory '{s}'\n", .{arg});
            }
            return err;
        };
        defer allocator.free(file);
        const h = dump.loadPrim(SimT, &s, &comm, allocator, io, file) catch |err| {
            if (err == error.RestartRankDisagreement) {
                if (is_root) std.debug.print(
                    name ++ ": ranks restarted from DIFFERENT checkpoints — the restart path must name the same file on every node\n",
                    .{},
                );
            } else {
                std.debug.print(name ++ ": cannot load restart '{s}': {s}\n", .{ file, @errorName(err) });
            }
            return err;
        };
        s.t = h.t;
        s.nstep = h.nstep;
        out_idx = h.out_idx;
        restarted = true;
        try s.finishInit();
        if (is_root) std.debug.print(
            name ++ ": RESTARTED from {s} — t={e:.6}, nstep={d}, continuing from frame #{d}\n",
            .{ file, h.t, h.nstep, h.out_idx },
        );
    } else {
        const fac = try P.initAll(SimT, &s, phys);
        s.initTimestepGuess();
        if (is_root) std.debug.print(name ++ ": init done — β-normalization fac = {e:.6}\n", .{fac});
        if (comptime @hasDecl(P, "afterInit")) try P.afterInit(SimT, &s, phys, is_root);
    }

    std.Io.Dir.cwd().createDirPath(io, p.out_dir) catch {};
    var log: std.ArrayList(u8) = .empty;
    defer log.deinit(allocator);
    try log.appendSlice(allocator, dump.scalar_header);

    var next_out: f64 = if (restarted and p.dtout1 > 0)
        (@floor(s.t / p.dtout1) + 1.0) * p.dtout1
    else
        p.tstart + p.dtout1;
    {
        const row = try dump.scalarRow(SimT, &s, 0.0, r_lum, r_scale);
        if (is_root) {
            try dump.appendScalarLine(&log, allocator, row);
            writeScalars(name, io, p.out_dir, log.items);
        }
        if (!restarted) try writeFrame(SimT, name, io, allocator, &s, &comm, p.out_dir, out_idx);
    }

    s.timers.reset();
    var hb = Heartbeat{ .last_ns = sim_mod.nowNs() };
    while (s.t < p.tmax and s.nstep < p.nstep_max) {
        var dt = s.cflDt();
        if (s.t + dt > p.tmax) dt = p.tmax - s.t;

        if (!(dt > 0) or !std.math.isFinite(dt)) {
            std.debug.print(
                name ++ ": invalid timestep dt={e} at step {d} t={d} (tstepdenmax={e}) — aborting\n",
                .{ dt, s.nstep, s.t, s.tstepdenmax },
            );
            return error.InvalidTimestep;
        }

        const step_t0 = sim_mod.nowNs();
        s.step(dt) catch |err| {
            std.debug.print(name ++ ": step {d} failed at t={d}: {s}\n", .{ s.nstep, s.t, @errorName(err) });
            return err;
        };
        const step_end = sim_mod.nowNs();
        hb.ring_fail += s.n_radimp_fail_step;

        if (is_root and step_end - hb.last_ns > heartbeat_interval_ns) {
            printHeartbeat(SimT, &s, dt, step_end - step_t0, &hb);
            hb.last_ns = step_end;
        }

        const time_due = s.t >= next_out;
        const step_due = p.nout_step > 0 and s.nstep % p.nout_step == 0;
        if (time_due or step_due or s.nstep >= p.nstep_max) {
            out_idx += 1;
            const row = try dump.scalarRow(SimT, &s, dt, r_lum, r_scale);
            if (is_root) {
                try dump.appendScalarLine(&log, allocator, row);
                writeScalars(name, io, p.out_dir, log.items);
            }
            try writeFrame(SimT, name, io, allocator, &s, &comm, p.out_dir, out_idx);
            if (is_root) {
                std.debug.print(
                    name ++ ": t={d:.2} nstep={d} dt={e:.3} | Ṁ={e:.3} L={e:.3} H/R={d:.3} β⁻¹={e:.3} | nan={d} hdfix={d} radimpfail={d}\n",
                    .{ s.t, s.nstep, dt, row.mdot, row.radlum, row.scaleheight, row.max_pmag_ptot, row.n_nan, row.n_hd_fixup, row.n_radimp_fail },
                );
                s.timers.printReport();
            }
            s.timers.reset();
            if (row.n_nan > 0) {
                const local = dump.collectDiag(SimT, &s).n_nan;
                if (local > 0)
                    std.debug.print(name ++ ": NaN detected on rank {d} ({d} cells) — aborting\n", .{ comm.rank(), local });
                return error.NanDetected;
            }
            if (time_due) next_out += p.dtout1;
        }
    }
    if (is_root) std.debug.print(name ++ ": done (t={d}, {d} steps)\n", .{ s.t, s.nstep });
}
