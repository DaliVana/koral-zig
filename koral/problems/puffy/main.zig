//! PUFFY — radiative MHD limotorus around a Schwarzschild BH (a = 0)
//! (port of koral_lite/PROBLEMS/PUFFY). MASS defaults to 10 M☉ but is set
//! from the params file, so presets can retarget the scale (e.g. the
//! Sagittarius A* preset at ~4.3e6 M☉ — see koral/problems/puffy/puffy3d_sgra.toml).
//!
//! M13: the full driver. Runs the ko.c init sequence (limotorus + calc_BfromA
//! + β-normalization) — or, with `--restart <file|dir>`, continues from a KDMP
//! checkpoint (C: fread_restartfile_bin) — then the RK2IMEX time loop
//! (radiation implicit source, radiative shear viscosity, mean-field dynamo,
//! polar-axis correction), and writes the scalar diagnostics time series (Ṁ,
//! luminosity, H/R, held β) plus periodic KDMP primitive checkpoints (each a
//! complete restart point). Serial by default; nthreads drives the persistent
//! worker team over the per-cell inversions.
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

// The Sim.Options construction and the params-file physics overrides live in
// puffy.zig (simOptions / applyPhysicsOverrides) so kdmp2silo reconstructs a
// checkpoint's exact configuration from the same code path as this driver.

/// Domain diagnostics for one output row (finite/NaN + fixup counters).
const Diag = struct {
    n_nan: u64 = 0,
    n_hd_fixup: u64 = 0,
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
            }
        }
    }
    return d;
}

/// Compute the scalar row for the current state — globally folded (MPI
/// plan §8.3): each rank sums its slab and two collectives (one SUM with
/// the counters riding along, one MAX) make every rank return the same
/// row. Serially / at 1 rank the folds are identity and the values are
/// BITWISE the pre-MPI ones. The counter columns are true ring totals
/// (review §10.2: n_radimp_fail previously printed a rank-local value).
/// Every rank must call this at the same point — it is collective.
fn scalarRow(s: *SimT, dt: f64) !dump.ScalarRow {
    const r_horizon = koral.metric.core.rHorizonBL(s.opt.mp.a);
    const ix_h = scalars.radialShellIndex(SimT, s, r_horizon);
    const ix_l = scalars.radialShellIndex(SimT, s, r_lum);
    const diag = collectDiag(s);
    var counts = [3]f64{
        @floatFromInt(diag.n_hd_fixup),
        @floatFromInt(s.n_radimp_failures),
        @floatFromInt(diag.n_nan),
    };
    const gs = try scalars.globalScalars(SimT, s, ix_h, ix_l, r_scale, counts[0..]);
    return .{
        .t = s.t,
        .dt = dt,
        .nstep = s.nstep,
        .mass = gs.mass,
        .mdot = -gs.mdot, // >0 for accretion (negation is FP-exact, so the
        // fold-then-negate order matches serial's negate-the-total bitwise)
        .radlum = gs.radlum,
        .totallum = gs.totallum,
        .scaleheight = gs.scaleheight,
        .max_pmag_ptot = gs.max_pmag_ptot,
        .n_hd_fixup = @intFromFloat(counts[0]),
        .n_radimp_fail = @intFromFloat(counts[1]),
        .n_nan = @intFromFloat(counts[2]),
    };
}

fn writeScalars(io: std.Io, out_dir: []const u8, log: []const u8) void {
    var buf: [1024]u8 = undefined;
    const path = std.fmt.bufPrint(&buf, "{s}/scalars.dat", .{out_dir}) catch return;
    std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = log }) catch |err| {
        std.debug.print("puffy: cannot write {s}: {s}\n", .{ path, @errorName(err) });
    };
}

/// Write a numbered KDMP checkpoint (`prims#####.kdmp`). Each such file is a
/// complete restart point (its header carries t/nstep/out_idx), so a later run
/// can continue from it via `--restart <file|dir>`.
fn writePrimDump(io: std.Io, out_dir: []const u8, allocator: std.mem.Allocator, s: *const SimT, idx: u32) void {
    const bytes = allocator.alloc(u8, dump.primDumpSize(SimT, s)) catch return;
    defer allocator.free(bytes);
    const n = dump.serializePrimDump(SimT, s, idx, bytes);
    var buf: [1024]u8 = undefined;
    const path = std.fmt.bufPrint(&buf, "{s}/prims{d:0>5}.kdmp", .{ out_dir, idx }) catch return;
    std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = bytes[0..n] }) catch |err| {
        std.debug.print("puffy: cannot write {s}: {s}\n", .{ path, @errorName(err) });
    };
}

/// The ntz>1 checkpoint path (MPI plan §8.1): rank 0 writes the 44-byte
/// header with GLOBAL dims, then every rank issues one collective
/// `write_at_all` of its contiguous φ-slab at the closed-form offset. The
/// file is byte-identical to what a serial run writes (gate 6a pins this),
/// so checkpoints are mutually readable at any rank count.
///
/// Returns an error rather than swallowing one: every failure below is
/// RANK-LOCAL (a per-rank allocation, this rank's path formatting), and
/// bailing out silently would leave this rank outside a collective
/// sequence its peers are already blocked in (`MPI_File_open`, then
/// `write_at_all`) — a job-wide hang. Propagating instead lets the
/// driver's `errdefer comm.abortJob` tear the job down, the same
/// discipline every other rank-local failure here follows.
fn writePrimDumpMpi(out_dir: []const u8, allocator: std.mem.Allocator, s: *const SimT, comm: *koral.comm.Comm, idx: u32) !void {
    const g = s.decomp.global;
    const body64 = try allocator.alloc(f64, dump.primBodySize(SimT, s) / 8);
    defer allocator.free(body64);
    const body = std.mem.sliceAsBytes(body64);
    _ = dump.serializePrimBody(SimT, s, body);

    var buf: [1024]u8 = undefined;
    const path = try std.fmt.bufPrintZ(&buf, "{s}/prims{d:0>5}.kdmp", .{ out_dir, idx });
    const total = dump.bodyOffset(g.nx, g.ny, SimT.nv, g.nz); // header + whole body
    var f = comm.fileCreate(path, total) catch |err| {
        std.debug.print("puffy: rank {d} cannot MPI-create {s}: {s}\n", .{ comm.rank(), path, @errorName(err) });
        return err;
    };
    if (comm.rank() == 0) {
        var hdr: [dump.header_size]u8 = undefined;
        _ = dump.writeDumpHeader(&hdr, .{
            .nx = @intCast(g.nx),
            .ny = @intCast(g.ny),
            .nz = @intCast(g.nz),
            .nv = @intCast(SimT.nv),
            .t = s.t,
            .nstep = s.nstep,
            .out_idx = idx,
        });
        comm.fileWriteAt(&f, 0, hdr[0..]);
    }
    comm.fileWriteAtAll(&f, dump.bodyOffset(g.nx, g.ny, SimT.nv, @intCast(s.decomp.tok)), body);
    comm.fileClose(&f);
}

/// Collective restart read (plan §8.1 — read is symmetric): every rank
/// reads the header at offset 0 (identical offsets are legal in
/// read_at_all), validates it against the GLOBAL grid, then reads its own
/// φ-slab. A serial-written checkpoint restarts at any rank count and
/// vice versa (gate 6b).
fn loadPrimDumpMpi(allocator: std.mem.Allocator, s: *SimT, comm: *koral.comm.Comm, path: []const u8) !dump.DumpHeader {
    var pbuf: [1024]u8 = undefined;
    const pathz = try std.fmt.bufPrintZ(&pbuf, "{s}", .{path});
    var f = try comm.fileOpenRead(pathz);
    defer comm.fileClose(&f);

    // Length check FIRST — the serial path gets this free (it reads the
    // whole file and compares the byte count), but a collective read is
    // sized by the caller's buffer, not the file: MPI reads what is there,
    // reports the shortfall only in a status we pass IGNORE for, and
    // returns SUCCESS. Without this, a checkpoint truncated by a walltime
    // kill loads its tail from uninitialized heap as primitives — and
    // `--restart <dir>` picks the NEWEST checkpoint, which is exactly the
    // file such a kill leaves partial. Verified: pre-guard, a 200 KB-short
    // file restarted "successfully" and then died as `NanInFlux`, pointing
    // the blame at the physics.
    const fsize = comm.fileSize(&f);
    if (fsize < dump.header_size) return error.Truncated;

    var hdr: [dump.header_size]u8 = undefined;
    comm.fileReadAtAll(&f, 0, hdr[0..]);
    const h = try dump.parseDumpHeader(hdr[0..]);
    const g = s.decomp.global;
    if (h.nx != g.nx or h.ny != g.ny or h.nz != g.nz or h.nv != SimT.nv)
        return error.DimMismatch;

    // Header validated ⇒ the exact file size is determined. Same `<`
    // comparison the serial loader uses. Every rank tests the same two
    // numbers, so the ring accepts or rejects as one.
    if (fsize < dump.bodyOffset(g.nx, g.ny, SimT.nv, g.nz)) return error.Truncated;

    const body64 = try allocator.alloc(f64, dump.primBodySize(SimT, s) / 8);
    defer allocator.free(body64);
    const body = std.mem.sliceAsBytes(body64);
    comm.fileReadAtAll(&f, dump.bodyOffset(g.nx, g.ny, SimT.nv, @intCast(s.decomp.tok)), body);
    try dump.loadPrimBody(SimT, s, body);
    return h;
}

/// Resolve the `--restart` argument to a concrete KDMP file path (caller owns
/// the result and must free it). A directory selects its highest-numbered
/// `*.kdmp` entry — the zero-padded `prims#####.kdmp` names sort lexically by
/// index, so the lexical maximum is the latest checkpoint. A plain file is
/// returned as-is. Errors if a directory holds no `.kdmp` file.
fn resolveRestartPath(io: std.Io, allocator: std.mem.Allocator, arg: []const u8) ![]u8 {
    var d = std.Io.Dir.cwd().openDir(io, arg, .{ .iterate = true }) catch |err| switch (err) {
        error.NotDir => return allocator.dupe(u8, arg), // a plain file
        else => return err,
    };
    defer d.close(io);

    var best: ?[]u8 = null;
    errdefer if (best) |b| allocator.free(b);
    var it = d.iterate();
    while (try it.next(io)) |entry| {
        if (entry.kind != .file) continue;
        if (!std.mem.endsWith(u8, entry.name, ".kdmp")) continue;
        if (best == null or std.mem.order(u8, entry.name, best.?) == .gt) {
            if (best) |b| allocator.free(b);
            best = try allocator.dupe(u8, entry.name);
        }
    }
    const name = best orelse {
        std.debug.print("puffy: no .kdmp checkpoint found in directory '{s}'\n", .{arg});
        return error.NoRestartFile;
    };
    defer allocator.free(name);
    return std.fmt.allocPrint(allocator, "{s}/{s}", .{ arg, name });
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
    /// ntz>1: implicit failures accumulated from the END-OF-STEP FOLD
    /// (`Sim.n_radimp_fail_step`, a ring MAX), summed over this interval by
    /// the main loop. This is the one heartbeat number that sees the whole
    /// ring, and it costs nothing — the fold already happens every step.
    /// It answers "is any rank in trouble", which the rank-local delta
    /// cannot: a blow-up confined to another rank's φ-slab is invisible to
    /// rank 0. (As a MAX-of-per-step it is a lower bound on the true total;
    /// scalars.dat carries the exact SUM-folded count.)
    ring_fail: u64 = 0,
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
///   * mpi= — ntz>1 only (plan §8.3): the share of stepping wall-clock spent
///     in MPI (halo exchange + collectives) since the last output row, from
///     the PassTimers buckets — comm cost is visible from day one.
///
/// SCOPE under MPI: only the leading `fail#` number is ring-global (the
/// end-of-step fold, accumulated by the caller); `znps` is the global zone
/// count as C reports it; **everything else on this line is rank 0's own
/// slab**, which the trailing `(r0)` marks. The exact global counters are
/// in scalars.dat, folded at the output cadence.
///
/// Do NOT "fix" the remaining columns by adding a collective here: this
/// function is called under `if (is_root and <wall-clock throttle>)`, so
/// only rank 0 reaches it, and only on some steps — a collective inside
/// would be entered by one rank alone and deadlock the job instantly. Any
/// global value this line wants must be folded on the every-rank path (as
/// `n_radimp_fail_step` already is) and merely *read* here.
fn printHeartbeat(s: *const SimT, dt: f64, step_wall_ns: u64, hb: *Heartbeat) void {
    // GLOBAL zone count, as C does (problem.c:856 uses TNX*TNY*TNZ over
    // rank-0's wall time). Using the local slab would divide the reported
    // throughput by the rank count and make strong-scaling runs look flat.
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
        // MPI share of the stepping wall since the last output reset.
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

    // MPI bracket (plan §6.3): MPI_Init_thread(FUNNELED) before anything —
    // notably before the Team spawns inside Sim.init. No-ops serially.
    const Comm = koral.comm.Comm;
    try Comm.initWorld();
    defer Comm.finalizeWorld();

    // CLI: `puffy [params.toml] [--restart <file|dir>]`. The first positional
    // is the params file; `--restart` (or `-r`) points at a KDMP checkpoint to
    // continue from — a file restarts from it directly, a directory continues
    // from its latest `prims#####.kdmp`. With no `--restart`, the run starts
    // from scratch (the C `NORESTART` behavior).
    var args = std.process.Args.Iterator.init(init.minimal.args);
    defer args.deinit();
    _ = args.next(); // program name
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

    // MASS and BHSPIN are the physical scales the params file may retarget
    // (e.g. the Sagittarius A* presets: ~4.3e6 M☉, and a ≈ 0.5–0.9). Set them
    // before simOptions()/initAll so the whole chain agrees: MASS drives the
    // torus thermo/opacity/radiation-floor unit scale (consts()/atmConsts()
    // and — via simOptions() → radforce.puffyMassChan(p.mass, …) — the stepping
    // opacity);
    // BHSPIN (mp.a) drives the metric, the limotorus construction, and the
    // dynamo's horizon/ISCO/Ωₖ. simOptions() copies puffy.mp into the sim, so mp.a
    // must be set first. RMIN is recomputed from the spin BEFORE makeGridNz so
    // the inner radial boundary stays inside the shrinking Kerr horizon (a
    // params `rmin > 0` is an explicit override; otherwise rminForSpin keeps
    // the fiducial fractional excision depth — bit-exactly 1.85 at a = 0). RMAX
    // is NOT auto-derived (the torus extent is dimensionless in GM units and
    // only weakly, inward, spin-dependent, so 500 M always contains it); it is
    // just an optional `rmax > 0` override to enlarge the domain for a long
    // outflow/wind run. MKSR0/MKSH0 stay fixed PUFFY constants.
    puffy.mass = p.mass;
    puffy.mp.a = p.bhspin;
    // Apply the optional physics overrides (coords, floors, opacities, torus,
    // atmosphere, solver) BEFORE deriving rmin/makeGrid/options — a plain
    // puffy.toml sets none of these and runs exactly as before; puffy_agn.toml
    // retargets to the koral_lite_puffy AGN config. This can set mp.mksr0, so
    // it must run before rminForSpin/makeGridNz (both read mp.mksr0).
    puffy.applyPhysicsOverrides(&p);
    puffy.rmin = if (p.rmin > 0.0) p.rmin else puffy.rminForSpin(p.bhspin);
    if (p.rmax > 0.0) puffy.rmax = p.rmax; // else keep the fiducial 500 M

    // Load the MESA Rosseland opacity table (C: MESA_KAPPA) and point the
    // opacity channels at it BEFORE simOptions()/init read `puffy.channels`. Owned
    // here for the whole run; freed on return. The file is chosen to match the
    // gas composition (hfrac→X, mfrac→Z) — see koral_lite_puffy's exact-match
    // get_MESA_opacity_filename; here it is given explicitly in the toml.
    var mtab: ?koral.physics.mesa.MesaTable = null;
    defer if (mtab) |*t| t.deinit();
    if (p.mesa_table.len > 0) {
        mtab = koral.physics.mesa.MesaTable.load(allocator, io, p.mesa_table) catch |err| {
            std.debug.print("puffy: cannot load MESA table '{s}': {s}\n", .{ p.mesa_table, @errorName(err) });
            return err;
        };
        puffy.channels.mesa = &mtab.?;
        std.debug.print("puffy: MESA opacity table loaded from '{s}'\n", .{p.mesa_table});
    }

    if (puffy.rmax <= puffy.rmin) {
        std.debug.print(
            "puffy: invalid domain — RMAX ({d}) must exceed RMIN ({d}).\n",
            .{ puffy.rmax, puffy.rmin },
        );
        return error.InvalidDomain;
    }

    // φ-only decomposition (MPI plan §5): params nx/ny/nz are GLOBAL; each
    // rank evolves its z-slab of the global grid. Serial builds and 1-rank
    // runs take the trivial decomposition and behave exactly as before.
    const wsize = Comm.worldSize();
    const ntz: usize = if (p.ntz == 0) wsize else p.ntz;
    if (ntz != wsize) {
        std.debug.print("puffy: params ntz={d} but the job was launched with {d} rank(s) — they must agree (ntz=0 means auto)\n", .{ p.ntz, wsize });
        return error.RankCountMismatch;
    }
    var comm = try Comm.init(ntz);
    defer comm.deinit();
    const is_root = comm.rank() == 0;
    // From here on a failure may be RANK-LOCAL (a u2p/flux error on one
    // tile, a NaN in one φ-slab). Returning would run the collective
    // teardown — MPI_Comm_free then MPI_Finalize's job-wide fence — on the
    // failing rank while its peers are still blocked in MPI_Waitall or
    // MPI_Allreduce: the job hangs for its full wall-clock allocation
    // instead of dying. MPI_Abort kills every rank; it never returns.
    // (Uniform failures — bad params, an invalid global dt — would exit
    // cleanly on their own, but routing them here too costs nothing and
    // removes the "is this error uniform?" judgement call from every site.)
    errdefer if (ntz > 1) comm.abortJob(1);
    const grid_global = puffy.makeGridNz(p.nx, p.ny, p.nz);
    const dc = koral.comm.decompose(grid_global, ntz, comm.rank()) catch |err| {
        std.debug.print(
            "puffy: cannot decompose {d}×{d}×{d} over ntz={d}: {s} (φ-only splits need nz % ntz == 0 and nz/ntz ≥ NG={d}; 2D never decomposes)\n",
            .{ p.nx, p.ny, p.nz, ntz, @errorName(err), grid_global.ng },
        );
        return err;
    };
    var opt = puffy.simOptions(SimT, &p);
    opt.comm = &comm;
    opt.decomp = dc;
    var s = try SimT.init(allocator, dc.local, opt);
    defer s.deinit();
    if (p.pin_threads) {
        // Self-identifying per rank (cluster gate 7: the affinity report is
        // the first thing to check — that each rank got its own cpuset and
        // the team is bound inside it).
        if (s.team) |tm| {
            if (tm.pinnedWidth()) |w| {
                std.debug.print("puffy: rank {d}: {d} threads pinned over {d} allowed cpus\n", .{ comm.rank(), p.nthreads, w });
            } else {
                std.debug.print("puffy: rank {d}: pin_threads requested but not applied (non-Linux host or no affinity mask)\n", .{comm.rank()});
            }
        }
    }
    if (ntz > 1) {
        // Every rank announces its own placement — this is what you check
        // first on a cluster (that the ring is laid out as intended and
        // each rank got the thread count you asked for), and the lines are
        // interleaved by the launcher, so each must be self-identifying.
        std.debug.print(
            "puffy: rank {d}/{d} owns φ-cells [{d},{d}) of {d} — {d} threads\n",
            .{ comm.rank(), ntz, dc.tok, dc.tok + @as(i64, @intCast(dc.local.nz)), grid_global.nz, p.nthreads },
        );
        if (is_root) std.debug.print(
            "puffy: MPI φ-ring of {d} ranks × {d} threads; scalars.dat via rank 0, KDMP checkpoints collective (silo via kdmp2silo)\n",
            .{ ntz, p.nthreads },
        );
    }

    const u = koral.Units.init(p.mass);
    const r_hor = koral.metric.core.rHorizonBL(p.bhspin);
    const cells_inside = if (puffy.rmin < r_hor)
        (@log(r_hor - puffy.mp.mksr0) - @log(puffy.rmin - puffy.mp.mksr0)) /
            ((@log(puffy.rmax - puffy.mp.mksr0) - @log(puffy.rmin - puffy.mp.mksr0)) / @as(f64, @floatFromInt(p.nx)))
    else
        0.0;
    if (is_root) std.debug.print(
        "puffy: NV={d} grid {d}×{d}×{d} (+{d} ghosts) | M={d} M☉ (GM/c³={e:.4}s) a={d} r_h={d:.4} RMIN={d:.4} (~{d:.1} cells inside) RMAX={d:.1} | threads={d} | build={s}\n",
        .{ L.count, grid_global.nx, grid_global.ny, grid_global.nz, grid_global.ng, p.mass, u.gmc3(), p.bhspin, r_hor, puffy.rmin, cells_inside, puffy.rmax, p.nthreads, @tagName(builtin.mode) },
    );
    if (is_root and puffy.rmin >= r_hor) {
        std.debug.print(
            "puffy: *** NOTE: RMIN={d:.4} is OUTSIDE the horizon r_h={d:.4}; the plain-copy " ++
                "inner boundary is not a clean excision. (An explicit params `rmin` override " ++
                "defeats the auto horizon-tracking — drop it to restore a clean inner edge.) ***\n",
            .{ puffy.rmin, r_hor },
        );
    }
    if (is_root and p.bhspin != 0.0) {
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
    if (is_root and builtin.mode == .Debug) {
        std.debug.print(
            "puffy: *** WARNING: Debug build — expect a many-fold slowdown. " ++
                "Rebuild with `-Doptimize=ReleaseFast` for production runs. ***\n",
            .{},
        );
    }

    // ---- initialization or restart ----------------------------------------
    // Fresh start: build the torus (ko.c:140-263). Restart/continue: load a
    // KDMP checkpoint (C: fread_restartfile_bin) — p stored verbatim, u = p2u
    // with no re-flooring — then rebuild ghosts + the dt guess (C: set_bc +
    // the problem.c dt preamble). Everything else (u, wavespeeds, flags) is
    // derived from p, so the primitives + t are the whole restart state.
    var out_idx: u32 = 0;
    var restarted = false;
    if (restart_arg) |arg| {
        // Each rank resolves independently — a directory picks the lexical
        // max, which is deterministic regardless of iteration order, so the
        // ring always agrees on the file.
        const file = try resolveRestartPath(io, allocator, arg);
        defer allocator.free(file);
        const h = blk: {
            if (ntz > 1) {
                break :blk loadPrimDumpMpi(allocator, &s, &comm, file) catch |err| {
                    std.debug.print("puffy: rank {d} cannot load restart '{s}': {s}\n", .{ comm.rank(), file, @errorName(err) });
                    return err;
                };
            }
            const bytes = std.Io.Dir.cwd().readFileAlloc(io, file, allocator, .limited(1 << 30)) catch |err| {
                std.debug.print("puffy: cannot read restart file '{s}': {s}\n", .{ file, @errorName(err) });
                return err;
            };
            defer allocator.free(bytes);
            break :blk dump.loadPrimDump(SimT, &s, bytes) catch |err| {
                std.debug.print("puffy: cannot load restart '{s}': {s}\n", .{ file, @errorName(err) });
                return err;
            };
        };
        s.t = h.t;
        s.nstep = h.nstep;
        out_idx = h.out_idx;
        restarted = true;
        // Ghosts and dt are recomputed, never stored: exchanged z-ghost
        // primitives first (no-op serially / at 1 rank), then set_bc's
        // physical fills + p2u, then the dt guess (its min_d* operands were
        // ring-folded in Sim.init).
        s.exchangeHalos();
        try s.setBc(s.t, true);
        s.initTimestepGuess();
        if (is_root) std.debug.print(
            "puffy: RESTARTED from {s} — t={e:.6}, nstep={d}, continuing from frame #{d}\n",
            .{ file, h.t, h.nstep, h.out_idx },
        );
    } else {
        const fac = try puffy.initAll(SimT, &s);
        s.initTimestepGuess();
        if (is_root) std.debug.print("puffy: init done — β-normalization fac = {e:.6}\n", .{fac});
    }

    // scalars.dat log + output cadence (DTOUT1 in code time)
    std.Io.Dir.cwd().createDirPath(io, p.out_dir) catch {};
    var log: std.ArrayList(u8) = .empty;
    defer log.deinit(allocator);
    try log.appendSlice(allocator, dump.scalar_header);

    // On restart, advance next_out to the next DTOUT1 boundary past the resumed
    // time (mirroring C's floor(t/dtout) trigger) so we don't immediately
    // re-emit a frame; on a fresh start the first frame is at tstart+dtout1.
    var next_out: f64 = if (restarted and p.dtout1 > 0)
        (@floor(s.t / p.dtout1) + 1.0) * p.dtout1
    else
        p.tstart + p.dtout1;
    {
        const row = try scalarRow(&s, 0.0); // collective — every rank
        if (is_root) {
            try dump.appendScalarLine(&log, allocator, row);
            writeScalars(io, p.out_dir, log.items);
        }
        // A fresh run writes frame 0 as its first checkpoint; a restart already
        // has frame out_idx on disk, so it only re-seeds the diagnostics.
        if (!restarted) {
            if (ntz == 1) {
                writePrimDump(io, p.out_dir, allocator, &s, out_idx);
                writeSiloDump(io, p.out_dir, allocator, &s, out_idx);
            } else {
                try writePrimDumpMpi(p.out_dir, allocator, &s, &comm, out_idx);
            }
        }
    }

    // ---- RK2IMEX time loop -------------------------------------------------
    s.timers.reset(); // drop init-time bc/u2p/wavespeed samples
    var hb = Heartbeat{ .last_ns = koral.sim.nowNs() };
    while (s.t < p.tmax and s.nstep < p.nstep_max) {
        var dt = s.cflDt(); // CFL dt from the previous step's speeds
        if (s.t + dt > p.tmax) dt = p.tmax - s.t;

        // Guard the timestep before stepping. A global blow-up leaves
        // tstepdenmax stuck at its −1 reset sentinel (NaN fails the `>` update
        // in save_wavespeeds), giving dt = −1 → time marches backwards and the
        // NaN check below never fires; a diverging denominator gives dt → 0 and
        // stalls. `!(dt > 0)` catches both; the isFinite guard is defensive.
        // Abort loudly with a non-zero exit rather than spinning out garbage
        // steps until nstep_max (P1 correctness).
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

        // Accumulate the already-folded per-step failure count on the
        // every-rank path — free (the fold happened inside step()), and the
        // only way the rank-0 heartbeat can see trouble on another rank's
        // slab, since the heartbeat itself must stay collective-free.
        hb.ring_fail += s.n_radimp_fail_step;

        // C-style throttled per-step heartbeat (~1 Hz wall clock; rank 0).
        if (is_root and step_end - hb.last_ns > heartbeat_interval_ns) {
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
            // Collective (two folds inside) — every rank computes the same
            // globally-summed row. Every rank reaches this branch on the
            // same step: the cadence keys off s.t/s.nstep, which the
            // end-of-step fold keeps globally identical.
            const row = try scalarRow(&s, dt);
            if (is_root) {
                try dump.appendScalarLine(&log, allocator, row);
                writeScalars(io, p.out_dir, log.items);
            }
            // The KDMP dump is the restart checkpoint (C writes it every
            // DTOUT1), so emit it on every output frame — a `--restart` on this
            // out_dir then continues from the newest one. Under MPI it is the
            // §8.1 collective write; silo waits for PMPIO (P4c — use
            // kdmp2silo on the checkpoints meanwhile).
            if (ntz == 1) {
                writePrimDump(io, p.out_dir, allocator, &s, out_idx);
                writeSiloDump(io, p.out_dir, allocator, &s, out_idx);
            } else {
                try writePrimDumpMpi(p.out_dir, allocator, &s, &comm, out_idx);
            }
            if (is_root) {
                std.debug.print(
                    "puffy: t={d:.2} nstep={d} dt={e:.3} | Ṁ={e:.3} L={e:.3} H/R={d:.3} β⁻¹={e:.3} | nan={d} hdfix={d} radimpfail={d}\n",
                    .{ s.t, s.nstep, dt, row.mdot, row.radlum, row.scaleheight, row.max_pmag_ptot, row.n_nan, row.n_hd_fixup, row.n_radimp_fail },
                );
                // P0 (parallelization plan §7): per-pass wall-clock table for
                // the steps since the previous output row.
                s.timers.printReport();
            }
            s.timers.reset();
            // row.n_nan is the ring total, so every rank makes the SAME
            // abort decision and they leave together through the normal
            // path (a rank-local return would strand the others in the
            // next collective; the errdefer MPI_Abort is the backstop).
            if (row.n_nan > 0) {
                const local = collectDiag(&s).n_nan;
                if (local > 0)
                    std.debug.print("puffy: NaN detected on rank {d} ({d} cells) — aborting\n", .{ comm.rank(), local });
                // Non-zero exit so batch scripts / CI treat a NaN-poisoned run
                // as a failure rather than success (P1 correctness).
                return error.NanDetected;
            }
            if (time_due) next_out += p.dtout1;
        }
    }
    if (is_root) std.debug.print("puffy: done (t={d}, {d} steps)\n", .{ s.t, s.nstep });
}

comptime {
    // The C-diffability contract (mnemonics.h) — refuse to compile if broken.
    std.debug.assert(L.count == 13);
    std.debug.assert(L.index(.rho) == 0 and L.index(.entr) == 5);
    std.debug.assert(L.index(.b1) == 6 and L.index(.ee) == 9 and L.index(.fz) == 12);
}
