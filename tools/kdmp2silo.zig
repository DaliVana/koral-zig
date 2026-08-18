//! kdmp2silo — convert PUFFY KDMP checkpoints into VisIt-openable `.silo`
//! files (MPI plan §8.2, Phase A). Under MPI the run writes KDMP only (the
//! §8.1 collective checkpoints); this SERIAL tool replays a checkpoint
//! through the existing io/silo.zig writer, so multi-rank runs get VisIt
//! output with zero MPI-Silo machinery (native PMPIO dumps land in P4c).
//!
//! The sim configuration is rebuilt from the SAME params file and the SAME
//! `puffy.setup` path as the driver, so the mesh coordinates and every
//! derived field (temperatures, opacities) match what the run itself would
//! have written.
//!
//! usage: kdmp2silo <params.toml> <file.kdmp | dumps-dir> [out-dir]
//!   file — convert one checkpoint to <out-dir>/<name>.silo
//!   dir  — convert every *.kdmp in it (lexical order)
//!   out-dir defaults to the checkpoint's own directory.
//!
//! Build with `-Dsilo` (e.g. `zig build kdmp2silo -Dsilo -Doptimize=ReleaseFast`);
//! without it the tool builds but refuses to run.

const std = @import("std");
const koral = @import("koral");

const cfg = koral.config.puffy;
const puffy = koral.problems.puffy;
const dump = koral.io.dump;
const silo = koral.io.silo;
const SimT = koral.Sim(cfg);

fn convertOne(
    io: std.Io,
    allocator: std.mem.Allocator,
    s: *SimT,
    kdmp_path: []const u8,
    out_dir: []const u8,
) !void {
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, kdmp_path, allocator, .limited(1 << 32)) catch |err| {
        std.debug.print("kdmp2silo: cannot read '{s}': {s}\n", .{ kdmp_path, @errorName(err) });
        return err;
    };
    defer allocator.free(bytes);
    const h = dump.loadPrimDump(SimT, s, bytes) catch |err| {
        std.debug.print("kdmp2silo: cannot load '{s}': {s}\n", .{ kdmp_path, @errorName(err) });
        return err;
    };
    s.t = h.t;
    s.nstep = h.nstep;
    // Ghosts are not stored in a KDMP; the divB stencil silo writes reads
    // one ring of them (same recompute as the driver's restart path).
    try s.finishInit();

    const base = std.fs.path.basename(kdmp_path);
    const stem = if (std.mem.lastIndexOfScalar(u8, base, '.')) |dot| base[0..dot] else base;
    var pbuf: [1024]u8 = undefined;
    const out_path = try std.fmt.bufPrintZ(&pbuf, "{s}/{s}.silo", .{ out_dir, stem });
    const cycle: i32 = @intCast(@min(h.nstep, @as(u64, std.math.maxInt(i32))));
    silo.write(SimT, s, allocator, out_path, h.t, cycle, .{}) catch |err| {
        std.debug.print("kdmp2silo: cannot write {s}: {s}\n", .{ out_path, @errorName(err) });
        return err;
    };
    std.debug.print("kdmp2silo: {s} -> {s} (t={e:.6}, nstep={d})\n", .{ kdmp_path, out_path, h.t, h.nstep });
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    if (comptime !silo.enabled) {
        std.debug.print("kdmp2silo: built without -Dsilo — rebuild with `zig build kdmp2silo -Dsilo -Doptimize=ReleaseFast`\n", .{});
        return error.SiloDisabled;
    }

    var args = std.process.Args.Iterator.init(init.minimal.args);
    defer args.deinit();
    _ = args.next(); // program name
    const params_path = args.next() orelse {
        std.debug.print("usage: kdmp2silo <params.toml> <file.kdmp | dumps-dir> [out-dir]\n", .{});
        return error.BadArgs;
    };
    const target_arg = args.next() orelse {
        std.debug.print("usage: kdmp2silo <params.toml> <file.kdmp | dumps-dir> [out-dir]\n", .{});
        return error.BadArgs;
    };
    const out_dir_arg = args.next();

    var p = koral.Params.load(allocator, io, params_path) catch |err| {
        std.debug.print("kdmp2silo: cannot load params from '{s}': {s}\n", .{ params_path, @errorName(err) });
        return err;
    };
    defer p.deinit(allocator);

    var st = puffy.setup(SimT, allocator, io, &p) catch |err| {
        std.debug.print("kdmp2silo: cannot set up from '{s}': {s}\n", .{ params_path, @errorName(err) });
        return err;
    };
    defer st.deinit(allocator);

    var s = try SimT.init(allocator, st.grid_global, st.options);
    defer s.deinit();

    // A directory converts every checkpoint in lexical order (the zero-
    // padded prims#####.kdmp names sort by frame index); a file just itself.
    var d = std.Io.Dir.cwd().openDir(io, target_arg, .{ .iterate = true }) catch |err| switch (err) {
        error.NotDir => {
            const out_dir = out_dir_arg orelse (std.fs.path.dirname(target_arg) orelse ".");
            try convertOne(io, allocator, &s, target_arg, out_dir);
            return;
        },
        else => return err,
    };
    defer d.close(io);

    var names: std.ArrayList([]u8) = .empty;
    defer {
        for (names.items) |n| allocator.free(n);
        names.deinit(allocator);
    }
    var it = d.iterate();
    while (try it.next(io)) |entry| {
        if (entry.kind != .file) continue;
        if (!std.mem.endsWith(u8, entry.name, ".kdmp")) continue;
        try names.append(allocator, try allocator.dupe(u8, entry.name));
    }
    if (names.items.len == 0) {
        std.debug.print("kdmp2silo: no .kdmp checkpoint found in '{s}'\n", .{target_arg});
        return error.NoCheckpoint;
    }
    std.mem.sort([]u8, names.items, {}, struct {
        fn lt(_: void, a: []u8, b: []u8) bool {
            return std.mem.order(u8, a, b) == .lt;
        }
    }.lt);

    const out_dir = out_dir_arg orelse target_arg;
    for (names.items) |name| {
        var fbuf: [1024]u8 = undefined;
        const path = try std.fmt.bufPrint(&fbuf, "{s}/{s}", .{ target_arg, name });
        try convertOne(io, allocator, &s, path, out_dir);
    }
}
