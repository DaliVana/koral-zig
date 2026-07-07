//! PUFFY — radiative MHD limotorus around a 10 M☉ Schwarzschild BH
//! (port of koral_lite/PROBLEMS/PUFFY).
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

const SimT = koral.Sim(cfg);

/// PUFFY diagnostic radii (postproc.c calc_scalars): mass flux through the
/// horizon, luminosity at the outer shell, scale height at r = 15 M.
const r_horizon: f64 = 2.0; // rhorizonBL(a=0) = 1 + √(1−a²)
const r_lum: f64 = 5000.0; // > RMAX → the outer radial shell
const r_scale: f64 = 15.0;

fn options(p: *const koral.Params) SimT.Options {
    return .{
        .coords = .mks2,
        .mp = puffy.mp,
        .gam = puffy.gam,
        .tsteplim = p.tsteplim,
        .floors = koral.solve.invert.FloorParams.puffy,
        .rad = koral.solve.invert_rad.RadParams.puffy,
        .opac = koral.physics.radforce.Params.puffy(),
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

    var s = try SimT.init(allocator, puffy.makeGridNz(p.nx, p.ny, p.nz), options(&p));
    defer s.deinit();

    const u = koral.Units.init(p.mass);
    std.debug.print(
        "puffy: NV={d} grid {d}×{d}×{d} (+{d} ghosts) | M={d} M☉ (GM/c³={e:.4}s) | threads={d} | build={s}\n",
        .{ L.count, s.grid.nx, s.grid.ny, s.grid.nz, s.grid.ng, p.mass, u.gmc3(), p.nthreads, @tagName(builtin.mode) },
    );
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
    }

    // ---- RK2IMEX time loop -------------------------------------------------
    s.timers.reset(); // drop init-time bc/u2p/wavespeed samples
    while (s.t < p.tmax and s.nstep < p.nstep_max) {
        var dt = 1.0 / s.tstepdenmax; // CFL dt from the previous step's speeds
        if (s.t + dt > p.tmax) dt = p.tmax - s.t;
        s.step(dt) catch |err| {
            std.debug.print("puffy: step {d} failed at t={d}: {s}\n", .{ s.nstep, s.t, @errorName(err) });
            return err;
        };

        if (s.t >= next_out or s.nstep >= p.nstep_max) {
            out_idx += 1;
            const row = try scalarRow(&s, dt);
            try dump.appendScalarLine(&log, allocator, row);
            writeScalars(io, p.out_dir, allocator, log.items);
            if (p.dtout2 > 0) writePrimDump(io, p.out_dir, allocator, &s, out_idx);
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
            next_out += p.dtout1;
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
