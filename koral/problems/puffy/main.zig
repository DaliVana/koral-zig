//! PUFFY: radiative MHD limotorus around a Schwarzschild BH (a = 0)
//! (port of koral_lite/PROBLEMS/PUFFY). MASS defaults to 10 M☉ but is set
//! from the params file, so presets can retarget the scale (e.g. the
//! Sagittarius A* preset at ~4.3e6 M☉. See koral/problems/puffy/puffy3d_sgra.toml).
//!
//! Thin driver: parse CLI → puffy.setup → Sim → RK2IMEX loop → I/O.
//! Physics lives in `puffy.Physics`; checkpoints go through `koral.io.dump`.
//!
//! CLI:  puffy [params.toml] [--restart <file|dir>]
//!   --restart FILE  continue from that checkpoint;
//!   --restart DIR   continue from the newest prims#####.kdmp in DIR;
//!   (omitted)       start from scratch (C: NORESTART).
//! Bridge a C serial restart (res####.head/.dat) into a KDMP with the
//! `res2kdmp` tool (tools/res2kdmp.zig).

const std = @import("std");
const builtin = @import("builtin");
const koral = @import("koral");

const cfg = koral.config.puffy;
const L = koral.VarLayout(cfg);
const puffy = koral.problems.puffy;
const dump = koral.io.dump;
const silo = koral.io.silo;

const SimT = koral.Sim(cfg);

/// PUFFY diagnostic radii (postproc.c calc_scalars): luminosity at the outer
/// shell and scale height at r = 15 M. Mass-flux radius is the horizon.
const r_lum: f64 = 5000.0;
const r_scale: f64 = 15.0;

const heartbeat_interval_ns: u64 = 1_000_000_000;

fn writeScalars(io: std.Io, out_dir: []const u8, log: []const u8) void {
    var buf: [1024]u8 = undefined;
    const path = std.fmt.bufPrint(&buf, "{s}/scalars.dat", .{out_dir}) catch return;
    std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = log }) catch |err| {
        std.debug.print("puffy: cannot write {s}: {s}\n", .{ path, @errorName(err) });
    };
}

fn writeSiloDump(io: std.Io, out_dir: []const u8, allocator: std.mem.Allocator, s: *const SimT, idx: usize) void {
    if (comptime !silo.enabled) return;
    var dbuf: [1024]u8 = undefined;
    const dir = std.fmt.bufPrint(&dbuf, "{s}/silo", .{out_dir}) catch return;
    std.Io.Dir.cwd().createDirPath(io, dir) catch {};
    var buf: [1024]u8 = undefined;
    const path = std.fmt.bufPrintZ(&buf, "{s}/silo/puffy{d:0>4}.silo", .{ out_dir, idx }) catch return;
    const cycle: i32 = @intCast(@min(s.nstep, @as(u64, std.math.maxInt(i32))));
    silo.write(SimT, s, allocator, path, s.t, cycle, .{}) catch |err| {
        std.debug.print("puffy: cannot write {s}: {s}\n", .{ path, @errorName(err) });
    };
}

fn writeFrame(
    io: std.Io,
    allocator: std.mem.Allocator,
    s: *const SimT,
    comm: *koral.comm.Comm,
    out_dir: []const u8,
    idx: u32,
) !void {
    var pbuf: [1024]u8 = undefined;
    const path = try std.fmt.bufPrint(&pbuf, "{s}/prims{d:0>5}.kdmp", .{ out_dir, idx });
    try dump.writePrim(SimT, s, comm, allocator, io, path, idx);
    if (s.decomp.ntz <= 1) writeSiloDump(io, out_dir, allocator, s, idx);
}

const Heartbeat = struct {
    last_ns: u64,
    prev_fail: u64 = 0,
    prev_iters: u64 = 0,
    prev_solves: u64 = 0,
    ring_fail: u64 = 0,
};

fn countFixupFlags(s: *const SimT) struct { hd: u64, rad: u64 } {
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

fn printHeartbeat(s: *const SimT, dt: f64, step_wall_ns: u64, hb: *Heartbeat) void {
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
    const fx = countFixupFlags(s);

    if (s.decomp.ntz > 1) {
        var total_ns: u64 = 0;
        for (s.timers.ns) |v| total_ns += v;
        const comm_ns = s.timers.ns[@intFromEnum(koral.sim.Pass.halo)] +
            s.timers.ns[@intFromEnum(koral.sim.Pass.collect)];
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

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    const Comm = koral.comm.Comm;
    try Comm.initWorld();
    defer Comm.finalizeWorld();
    const world = Comm.worldSize();
    errdefer if (world > 1) Comm.abortWorld(1);

    var args = std.process.Args.Iterator.init(init.minimal.args);
    defer args.deinit();
    _ = args.next();
    var params_path: []const u8 = "koral/problems/puffy/puffy.toml";
    var restart_arg: ?[]const u8 = null;
    var have_params = false;
    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--restart") or std.mem.eql(u8, arg, "-r")) {
            restart_arg = args.next() orelse {
                std.debug.print("puffy: --restart needs a file or directory argument\n", .{});
                return error.MissingRestartArg;
            };
        } else if (std.mem.startsWith(u8, arg, "--restart=")) {
            restart_arg = arg["--restart=".len..];
        } else if (!have_params) {
            params_path = arg;
            have_params = true;
        } else {
            std.debug.print("puffy: unexpected extra argument '{s}'\n", .{arg});
            return error.BadArgs;
        }
    }

    var p = koral.Params.load(allocator, io, params_path) catch |err| {
        std.debug.print("puffy: cannot load params from '{s}': {s}\n", .{ params_path, @errorName(err) });
        return err;
    };
    defer p.deinit(allocator);

    var st = puffy.setup(SimT, allocator, io, &p) catch |err| {
        if (err == error.InvalidDomain) {
            const phys = puffy.Physics.fromParams(&p);
            std.debug.print(
                "puffy: invalid domain — RMAX ({d}) must exceed RMIN ({d}).\n",
                .{ phys.rmax, phys.rmin },
            );
        } else if (p.mesa_table.len > 0) {
            std.debug.print("puffy: cannot load MESA table '{s}': {s}\n", .{ p.mesa_table, @errorName(err) });
        }
        return err;
    };
    defer st.deinit(allocator);
    if (st.mesa != null) {
        std.debug.print("puffy: MESA opacity table loaded from '{s}'\n", .{p.mesa_table});
    }
    const phys = &st.physics;

    const ntz: usize = if (p.ntz == 0) world else p.ntz;
    if (ntz != world) {
        std.debug.print("puffy: params ntz={d} but the job was launched with {d} rank(s) — they must agree (ntz=0 means auto)\n", .{ p.ntz, world });
        return error.RankCountMismatch;
    }
    var comm = try Comm.init(ntz);
    defer comm.deinit();
    const is_root = comm.rank() == 0;
    errdefer if (ntz > 1) comm.abortJob(1);

    const dc = koral.comm.decompose(st.grid_global, ntz, comm.rank()) catch |err| {
        std.debug.print(
            "puffy: cannot decompose {d}×{d}×{d} over ntz={d}: {s} (φ-only splits need nz % ntz == 0 and nz/ntz ≥ NG={d}; 2D never decomposes)\n",
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
                std.debug.print("puffy: rank {d}: {d} threads pinned over {d} allowed cpus\n", .{ comm.rank(), p.nthreads, w });
            } else {
                std.debug.print("puffy: rank {d}: pin_threads requested but not applied (non-Linux host or no affinity mask)\n", .{comm.rank()});
            }
        }
    }
    if (ntz > 1) {
        std.debug.print(
            "puffy: rank {d}/{d} owns φ-cells [{d},{d}) of {d} — {d} threads\n",
            .{ comm.rank(), ntz, dc.tok, dc.tok + @as(i64, @intCast(dc.local.nz)), st.grid_global.nz, p.nthreads },
        );
        if (is_root) std.debug.print(
            "puffy: MPI φ-ring of {d} ranks × {d} threads; scalars.dat via rank 0, KDMP checkpoints collective (silo via kdmp2silo)\n",
            .{ ntz, p.nthreads },
        );
    }

    const u = koral.Units.init(phys.mass);
    const r_hor = koral.metric.core.rHorizonBL(phys.mp.a);
    const cells_inside = if (phys.rmin < r_hor)
        (@log(r_hor - phys.mp.mksr0) - @log(phys.rmin - phys.mp.mksr0)) /
            ((@log(phys.rmax - phys.mp.mksr0) - @log(phys.rmin - phys.mp.mksr0)) / @as(f64, @floatFromInt(p.nx)))
    else
        0.0;
    if (is_root) std.debug.print(
        "puffy: NV={d} grid {d}×{d}×{d} (+{d} ghosts) | M={d} M☉ (GM/c³={e:.4}s) a={d} r_h={d:.4} RMIN={d:.4} (~{d:.1} cells inside) RMAX={d:.1} | threads={d} | build={s}\n",
        .{ L.count, st.grid_global.nx, st.grid_global.ny, st.grid_global.nz, st.grid_global.ng, phys.mass, u.gmc3(), phys.mp.a, r_hor, phys.rmin, cells_inside, phys.rmax, p.nthreads, @tagName(builtin.mode) },
    );
    if (is_root and phys.rmin >= r_hor) {
        std.debug.print(
            "puffy: *** NOTE: RMIN={d:.4} is OUTSIDE the horizon r_h={d:.4}; the plain-copy " ++
                "inner boundary is not a clean excision. (An explicit params `rmin` override " ++
                "defeats the auto horizon-tracking — drop it to restore a clean inner edge.) ***\n",
            .{ phys.rmin, r_hor },
        );
    }
    if (is_root and phys.mp.a != 0.0) {
        std.debug.print(
            "puffy: *** NOTE: a≠0 is UNVALIDATED — C KORAL PUFFY uses BHSPIN=0, so there is " ++
                "no bit-for-bit oracle at this spin. Treat spinning runs as experiments. ***\n",
            .{},
        );
    }
    if (is_root and builtin.mode == .Debug) {
        std.debug.print(
            "puffy: *** WARNING: Debug build — expect a many-fold slowdown. " ++
                "Rebuild with `-Doptimize=ReleaseFast` for production runs. ***\n",
            .{},
        );
    }

    var out_idx: u32 = 0;
    var restarted = false;
    if (restart_arg) |arg| {
        const file = dump.resolveRestartPath(io, allocator, arg) catch |err| {
            if (err == error.NoRestartFile) {
                std.debug.print("puffy: no .kdmp checkpoint found in directory '{s}'\n", .{arg});
            }
            return err;
        };
        defer allocator.free(file);
        const h = dump.loadPrim(SimT, &s, &comm, allocator, io, file) catch |err| {
            if (err == error.RestartRankDisagreement) {
                if (is_root) std.debug.print(
                    "puffy: ranks restarted from DIFFERENT checkpoints — the restart path must name the same file on every node\n",
                    .{},
                );
            } else {
                std.debug.print("puffy: cannot load restart '{s}': {s}\n", .{ file, @errorName(err) });
            }
            return err;
        };
        s.t = h.t;
        s.nstep = h.nstep;
        out_idx = h.out_idx;
        restarted = true;
        try s.finishInit();
        if (is_root) std.debug.print(
            "puffy: RESTARTED from {s} — t={e:.6}, nstep={d}, continuing from frame #{d}\n",
            .{ file, h.t, h.nstep, h.out_idx },
        );
    } else {
        const fac = try puffy.initAllWith(SimT, &s, phys);
        s.initTimestepGuess();
        if (is_root) std.debug.print("puffy: init done — β-normalization fac = {e:.6}\n", .{fac});

        {
            const sq = try puffy.seedQualityWith(SimT, &s, phys);
            const m = s.globalSum(sq.mass);
            const qr = s.globalSum(sq.qr_m) / @max(m, 1e-300);
            const qth = s.globalSum(sq.qth_m) / @max(m, 1e-300);
            if (is_root and m > 0 and qth > 1e-30) {
                const target = 6.0;
                const mb_needed = phys.maxbeta * (target / qth) * (target / qth);
                std.debug.print(
                    "puffy: seed MRI quality <Q_r>={d:.2} <Q_th>={d:.2} (mass-weighted; growth floor ~{d:.0}; perturb={d})\n",
                    .{ qr, qth, target, phys.perturb },
                );
                if (qth < target) {
                    if (mb_needed <= 0.2) {
                        std.debug.print(
                            "puffy: *** NOTE: seed <Q_th> is BELOW the growth floor — the linear MRI will be " ++
                                "grid-suppressed. maxbeta = {d:.3} would reach <Q_th> = {d:.0} on this grid.\n",
                            .{ mb_needed, target },
                        );
                    } else {
                        std.debug.print(
                            "puffy: *** NOTE: seed <Q_th> is BELOW the growth floor, and no maxbeta <= 0.2 can fix " ++
                                "it on this grid (would need {d:.3}) — raise ny instead.\n",
                            .{mb_needed},
                        );
                    }
                }
            }
        }
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
            writeScalars(io, p.out_dir, log.items);
        }
        if (!restarted) try writeFrame(io, allocator, &s, &comm, p.out_dir, out_idx);
    }

    s.timers.reset();
    var hb = Heartbeat{ .last_ns = koral.sim.nowNs() };
    while (s.t < p.tmax and s.nstep < p.nstep_max) {
        var dt = s.cflDt();
        if (s.t + dt > p.tmax) dt = p.tmax - s.t;

        if (!(dt > 0) or !std.math.isFinite(dt)) {
            std.debug.print(
                "puffy: invalid timestep dt={e} at step {d} t={d} (tstepdenmax={e}) — aborting\n",
                .{ dt, s.nstep, s.t, s.tstepdenmax },
            );
            return error.InvalidTimestep;
        }

        const step_t0 = koral.sim.nowNs();
        s.step(dt) catch |err| {
            std.debug.print("puffy: step {d} failed at t={d}: {s}\n", .{ s.nstep, s.t, @errorName(err) });
            return err;
        };
        const step_end = koral.sim.nowNs();
        hb.ring_fail += s.n_radimp_fail_step;

        if (is_root and step_end - hb.last_ns > heartbeat_interval_ns) {
            printHeartbeat(&s, dt, step_end - step_t0, &hb);
            hb.last_ns = step_end;
        }

        const time_due = s.t >= next_out;
        const step_due = p.nout_step > 0 and s.nstep % p.nout_step == 0;
        if (time_due or step_due or s.nstep >= p.nstep_max) {
            out_idx += 1;
            const row = try dump.scalarRow(SimT, &s, dt, r_lum, r_scale);
            if (is_root) {
                try dump.appendScalarLine(&log, allocator, row);
                writeScalars(io, p.out_dir, log.items);
            }
            try writeFrame(io, allocator, &s, &comm, p.out_dir, out_idx);
            if (is_root) {
                std.debug.print(
                    "puffy: t={d:.2} nstep={d} dt={e:.3} | Ṁ={e:.3} L={e:.3} H/R={d:.3} β⁻¹={e:.3} | nan={d} hdfix={d} radimpfail={d}\n",
                    .{ s.t, s.nstep, dt, row.mdot, row.radlum, row.scaleheight, row.max_pmag_ptot, row.n_nan, row.n_hd_fixup, row.n_radimp_fail },
                );
                s.timers.printReport();
            }
            s.timers.reset();
            if (row.n_nan > 0) {
                const local = dump.collectDiag(SimT, &s).n_nan;
                if (local > 0)
                    std.debug.print("puffy: NaN detected on rank {d} ({d} cells) — aborting\n", .{ comm.rank(), local });
                return error.NanDetected;
            }
            if (time_due) next_out += p.dtout1;
        }
    }
    if (is_root) std.debug.print("puffy: done (t={d}, {d} steps)\n", .{ s.t, s.nstep });
}

comptime {
    std.debug.assert(L.count == 13);
    std.debug.assert(L.index(.rho) == 0 and L.index(.entr) == 5);
    std.debug.assert(L.index(.b1) == 6 and L.index(.ee) == 9 and L.index(.fz) == 12);
}
