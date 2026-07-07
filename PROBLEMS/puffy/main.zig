//! PUFFY — radiative MHD limotorus around a Schwarzschild BH (a = 0)
//! (port of koral_lite/PROBLEMS/PUFFY). MASS defaults to 10 M☉ but is set
//! from the params file, so presets can retarget the scale (e.g. the
//! Sagittarius A* preset at ~4.3e6 M☉ — see PROBLEMS/puffy/puffy3d_sgra.toml).
//!
//! M13: the full driver. Runs the ko.c init sequence (limotorus + calc_BfromA
//! + β-normalization), then the RK2IMEX time loop (radiation implicit source,
//! radiative shear viscosity, mean-field dynamo, polar-axis correction), and
//! writes the scalar diagnostics time series (Ṁ, luminosity, H/R, held β) plus
//! periodic binary primitive dumps. Serial by default; -D…/nthreads drives an
//! optional std.Thread.Pool over the per-cell inversions.

const std = @import("std");
const builtin = @import("builtin");
const koral = @import("koral");

/// Comptime physics/algorithm configuration (PROBLEMS/PUFFY/define.h):
/// MHD + M1 radiation, PPM, LAXF, RK2IMEX, MKS2 coordinates.
const cfg = koral.config.puffy;
const L = koral.VarLayout(cfg);
const puffy = koral.problems.puffy;
const scalars = koral.io.scalars;
const dump = koral.io.dump;
const silo = koral.io.silo;

const SimT = koral.Sim(cfg);

/// PUFFY diagnostic radii (postproc.c calc_scalars): luminosity at the outer
/// shell and scale height at r = 15 M. The mass-flux radius is the event
/// horizon, computed per-run from the spin via `rHorizonBL(a)` (→ 2 at a = 0).
const r_lum: f64 = 5000.0; // > RMAX → the outer radial shell
const r_scale: f64 = 15.0;

/// Wall-clock throttle for the per-step heartbeat (C: fprintf when
/// end_time−fprintf_time > 1 s, problem.c:979). Decouples the console cadence
/// from the step cadence — ~1 line/s on a fast run, one per step when a single
/// step already exceeds the interval.
const heartbeat_interval_ns: u64 = 1_000_000_000;

fn options(p: *const koral.Params) SimT.Options {
    return .{
        .coords = .mks2,
        .mp = puffy.mp,
        .gam = puffy.gam,
        .tsteplim = p.tsteplim,
        .floors = koral.solve.invert.FloorParams.puffy,
        .rad = koral.solve.invert_rad.RadParams.puffy,
        .opac = koral.physics.radforce.Params.puffyMass(p.mass),
        .implicit = koral.solve.implicit.ImplicitParams.puffy,
        .correct_polaraxis = true,
        .nccorrectpolar = 2,
        .radviscosity = true,
        .dynamo = true,
        .bc_x = .specific,
        .bc_y = .specific,
        .specific_bc = &puffy.Bc(SimT).calc,
        .nthreads = p.nthreads,
    };
}

/// Domain diagnostics for one output row (finite/NaN + fixup counters).
const Diag = struct {
    n_nan: u64 = 0,
    n_hd_fixup: u64 = 0,
    n_rad_fixup: u64 = 0,
};

fn collectDiag(s: *const SimT) Diag {
    var d = Diag{};
    var iz: i64 = 0;
    while (iz < s.nzi()) : (iz += 1) {
        var iy: i64 = 0;
        while (iy < s.nyi()) : (iy += 1) {
            var ix: i64 = 0;
            while (ix < s.nxi()) : (ix += 1) {
                var pp: [SimT.nv]f64 = undefined;
                s.p.load(ix, iy, iz, &pp);
                for (pp) |v| {
                    if (!std.math.isFinite(v)) {
                        d.n_nan += 1;
                        break;
                    }
                }
                if (s.getFlag(.hd_fixup, ix, iy, iz) != 0) d.n_hd_fixup += 1;
                if (comptime L.hasVar(.ee)) {
                    if (s.getFlag(.rad_fixup, ix, iy, iz) != 0) d.n_rad_fixup += 1;
                }
            }
        }
    }
    return d;
}

/// Compute the scalar row for the current state.
fn scalarRow(s: *SimT, dt: f64) !dump.ScalarRow {
    const r_horizon = koral.metric.core.rHorizonBL(s.opt.mp.a);
    const ix_h = scalars.radialShellIndex(SimT, s, r_horizon);
    const ix_l = scalars.radialShellIndex(SimT, s, r_lum);
    const mass = scalars.totalMass(SimT, s);
    const mdot = -(try scalars.mdot(SimT, s, ix_h)); // >0 for accretion
    const lum = try scalars.lum(SimT, s, ix_l);
    const h = scalars.scaleHeightAt(SimT, s, r_scale);
    const maxb = try scalars.maxPmagPtot(SimT, s);
    const diag = collectDiag(s);
    return .{
        .t = s.t,
        .dt = dt,
        .nstep = s.nstep,
        .mass = mass,
        .mdot = mdot,
        .radlum = lum.radlum,
        .totallum = lum.totallum,
        .scaleheight = h,
        .max_pmag_ptot = maxb,
        .n_hd_fixup = diag.n_hd_fixup,
        .n_radimp_fail = s.n_radimp_failures,
        .n_nan = diag.n_nan,
    };
}

fn writeScalars(io: std.Io, out_dir: []const u8, allocator: std.mem.Allocator, log: []const u8) void {
    var buf: [1024]u8 = undefined;
    const path = std.fmt.bufPrint(&buf, "{s}/scalars.dat", .{out_dir}) catch return;
    _ = allocator;
    std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = log }) catch |err| {
        std.debug.print("puffy: cannot write {s}: {s}\n", .{ path, @errorName(err) });
    };
}

fn writePrimDump(io: std.Io, out_dir: []const u8, allocator: std.mem.Allocator, s: *const SimT, idx: usize) void {
    const bytes = allocator.alloc(u8, dump.primDumpSize(SimT, s)) catch return;
    defer allocator.free(bytes);
    const n = dump.serializePrimDump(SimT, s, bytes);
    var buf: [1024]u8 = undefined;
    const path = std.fmt.bufPrint(&buf, "{s}/prims{d:0>5}.kdmp", .{ out_dir, idx }) catch return;
    std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = bytes[0..n] }) catch |err| {
        std.debug.print("puffy: cannot write {s}: {s}\n", .{ path, @errorName(err) });
    };
}

/// Write a VisIt-openable `.silo` snapshot (koral/io/silo.zig). No-op unless
/// built with `-Dsilo`; the internal comptime guard keeps this side-effect-free
/// (and Silo-symbol-free) otherwise.
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

/// Rolling snapshot of the run-cumulative implicit counters so the heartbeat
/// can report per-interval deltas, plus the wall clock of the last print.
const Heartbeat = struct {
    last_ns: u64,
    prev_fail: u64 = 0,
    prev_iters: u64 = 0,
    prev_solves: u64 = 0,
};

/// Cells currently flagged for a HD / radiation u2p fixup — a snapshot of the
/// last inversion (C accumulates these per step into the fail# fields).
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

/// C-style per-step progress line (problem.c:982), throttled by the caller.
///   * znps — domain zones per wall-second (throughput)
///   * tgpd — simulated GM/c³ per wall-day (projected reach at this speed)
///   * fail# — this interval's {radimp failures, HD fixups, rad fixups}
///   * imp# — implicit solves this interval and their mean Newton iterations
/// No `mpi=` field (shared-memory only — no MPI communication to attribute).
fn printHeartbeat(s: *const SimT, dt: f64, step_wall_ns: u64, hb: *Heartbeat) void {
    const ncells: f64 = @floatFromInt(s.grid.nx * s.grid.ny * s.grid.nz);
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

    std.debug.print(
        "st #{d:>6} t={e:.5} dt={e:.2} znps={d:.0} tgpd={e:.2} fail# {d} {d} {d} imp# {d} it {d:.1}\n",
        .{ s.nstep, s.t, dt, znps, tgpd, d_fail, fx.hd, fx.rad, d_solves, avg_it },
    );

    hb.prev_fail = s.n_radimp_failures;
    hb.prev_iters = s.n_radimp_iters;
    hb.prev_solves = s.n_radimp_solves;
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    var args = std.process.Args.Iterator.init(init.minimal.args);
    defer args.deinit();
    _ = args.next(); // program name
    const params_path = args.next() orelse "PROBLEMS/puffy/puffy.toml";

    var p = koral.Params.load(allocator, io, params_path) catch |err| {
        std.debug.print("puffy: cannot load params from '{s}': {s}\n", .{ params_path, @errorName(err) });
        return err;
    };
    defer p.deinit(allocator);

    // MASS and BHSPIN are the physical scales the params file may retarget
    // (e.g. the Sagittarius A* presets: ~4.3e6 M☉, and a ≈ 0.5–0.9). Set them
    // before options()/initAll so the whole chain agrees: MASS drives the
    // torus thermo/opacity/radiation-floor unit scale (consts()/atmConsts()
    // and — via options() → radforce.puffyMass(p.mass) — the stepping opacity);
    // BHSPIN (mp.a) drives the metric, the limotorus construction, and the
    // dynamo's horizon/ISCO/Ωₖ. options() copies puffy.mp into the sim, so mp.a
    // must be set first. RMIN is recomputed from the spin BEFORE makeGridNz so
    // the inner radial boundary stays inside the shrinking Kerr horizon (a
    // params `rmin > 0` is an explicit override; otherwise rminForSpin keeps
    // the fiducial fractional excision depth — bit-exactly 1.85 at a = 0).
    // RMAX/MKSR0/MKSH0 stay fixed PUFFY constants, independent of mass/spin.
    puffy.mass = p.mass;
    puffy.mp.a = p.bhspin;
    puffy.rmin = if (p.rmin > 0.0) p.rmin else puffy.rminForSpin(p.bhspin);

    var s = try SimT.init(allocator, puffy.makeGridNz(p.nx, p.ny, p.nz), options(&p));
    defer s.deinit();

    const u = koral.Units.init(p.mass);
    const r_hor = koral.metric.core.rHorizonBL(p.bhspin);
    const cells_inside = if (puffy.rmin < r_hor)
        (@log(r_hor - puffy.mp.mksr0) - @log(puffy.rmin - puffy.mp.mksr0)) /
            ((@log(puffy.rmax - puffy.mp.mksr0) - @log(puffy.rmin - puffy.mp.mksr0)) / @as(f64, @floatFromInt(p.nx)))
    else
        0.0;
    std.debug.print(
        "puffy: NV={d} grid {d}×{d}×{d} (+{d} ghosts) | M={d} M☉ (GM/c³={e:.4}s) a={d} r_h={d:.4} RMIN={d:.4} (~{d:.1} cells inside) | threads={d} | build={s}\n",
        .{ L.count, s.grid.nx, s.grid.ny, s.grid.nz, s.grid.ng, p.mass, u.gmc3(), p.bhspin, r_hor, puffy.rmin, cells_inside, p.nthreads, @tagName(builtin.mode) },
    );
    if (puffy.rmin >= r_hor) {
        std.debug.print(
            "puffy: *** NOTE: RMIN={d:.4} is OUTSIDE the horizon r_h={d:.4}; the plain-copy " ++
                "inner boundary is not a clean excision. (An explicit params `rmin` override " ++
                "defeats the auto horizon-tracking — drop it to restore a clean inner edge.) ***\n",
            .{ puffy.rmin, r_hor },
        );
    }
    if (p.bhspin != 0.0) {
        std.debug.print(
            "puffy: *** NOTE: a≠0 is UNVALIDATED — C KORAL PUFFY uses BHSPIN=0, so there is " ++
                "no bit-for-bit oracle at this spin. Treat spinning runs as experiments. ***\n",
            .{},
        );
    }
    // A production GRRMHD run in Debug pays every bounds-check/assert and no
    // inlining — a many-fold slowdown that can silently cost days. Zig 0.16's
    // preferred_optimize_mode defaults to Debug too and would break the
    // documented `-Doptimize=ReleaseFast` invocation, so we don't touch the
    // build default; we make an accidental Debug run loud instead.
    if (builtin.mode == .Debug) {
        std.debug.print(
            "puffy: *** WARNING: Debug build — expect a many-fold slowdown. " ++
                "Rebuild with `-Doptimize=ReleaseFast` for production runs. ***\n",
            .{},
        );
    }

    // ---- initialization (ko.c:140-263 + the solve preamble dt guess) ----
    const fac = try puffy.initAll(SimT, &s);
    s.initTimestepGuess();
    std.debug.print("puffy: init done — β-normalization fac = {e:.6}\n", .{fac});

    // scalars.dat log + output cadence (DTOUT1 in code time)
    std.Io.Dir.cwd().createDirPath(io, p.out_dir) catch {};
    var log: std.ArrayList(u8) = .empty;
    defer log.deinit(allocator);
    try log.appendSlice(allocator, dump.scalar_header);

    var out_idx: usize = 0;
    var next_out: f64 = p.tstart + p.dtout1;
    {
        const row = try scalarRow(&s, 0.0);
        try dump.appendScalarLine(&log, allocator, row);
        writeScalars(io, p.out_dir, allocator, log.items);
        writePrimDump(io, p.out_dir, allocator, &s, out_idx);
        writeSiloDump(io, p.out_dir, allocator, &s, out_idx);
    }

    // ---- RK2IMEX time loop -------------------------------------------------
    s.timers.reset(); // drop init-time bc/u2p/wavespeed samples
    var hb = Heartbeat{ .last_ns = koral.sim.nowNs() };
    while (s.t < p.tmax and s.nstep < p.nstep_max) {
        var dt = 1.0 / s.tstepdenmax; // CFL dt from the previous step's speeds
        if (s.t + dt > p.tmax) dt = p.tmax - s.t;

        const step_t0 = koral.sim.nowNs();
        s.step(dt) catch |err| {
            std.debug.print("puffy: step {d} failed at t={d}: {s}\n", .{ s.nstep, s.t, @errorName(err) });
            return err;
        };
        const step_end = koral.sim.nowNs();

        // C-style throttled per-step heartbeat (~1 Hz wall clock).
        if (step_end - hb.last_ns > heartbeat_interval_ns) {
            printHeartbeat(&s, dt, step_end - step_t0, &hb);
            hb.last_ns = step_end;
        }

        // Output when the code-time cadence is due (C: DTOUT1), or every
        // nout_step steps (a step-based convenience for interactive watching),
        // or on the final step. Only the time gate advances next_out.
        const time_due = s.t >= next_out;
        const step_due = p.nout_step > 0 and s.nstep % p.nout_step == 0;
        if (time_due or step_due or s.nstep >= p.nstep_max) {
            out_idx += 1;
            const row = try scalarRow(&s, dt);
            try dump.appendScalarLine(&log, allocator, row);
            writeScalars(io, p.out_dir, allocator, log.items);
            if (p.dtout2 > 0) writePrimDump(io, p.out_dir, allocator, &s, out_idx);
            writeSiloDump(io, p.out_dir, allocator, &s, out_idx);
            std.debug.print(
                "puffy: t={d:.2} nstep={d} dt={e:.3} | Ṁ={e:.3} L={e:.3} H/R={d:.3} β⁻¹={e:.3} | nan={d} hdfix={d} radimpfail={d}\n",
                .{ s.t, s.nstep, dt, row.mdot, row.radlum, row.scaleheight, row.max_pmag_ptot, row.n_nan, row.n_hd_fixup, row.n_radimp_fail },
            );
            // P0 (parallelization plan §7): per-pass wall-clock table for
            // the steps since the previous output row.
            s.timers.printReport();
            s.timers.reset();
            if (row.n_nan > 0) {
                std.debug.print("puffy: NaN detected — aborting\n", .{});
                return;
            }
            if (time_due) next_out += p.dtout1;
        }
    }
    std.debug.print("puffy: done (t={d}, {d} steps)\n", .{ s.t, s.nstep });
}

comptime {
    // The C-diffability contract (mnemonics.h) — refuse to compile if broken.
    std.debug.assert(L.count == 13);
    std.debug.assert(L.index(.rho) == 0 and L.index(.entr) == 5);
    std.debug.assert(L.index(.b1) == 6 and L.index(.ee) == 9 and L.index(.fz) == 12);
}
