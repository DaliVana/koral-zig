//! Regenerate the self-golden baselines: `zig build update-self-goldens`.
//!
//! Writes every scenario in `koral/testing/selfscenarios.zig` to
//! `tests/selfgolden/`, overwriting what is there. Run this ONLY when you have
//! decided that a numerical change is intended — the committed files are the
//! record of what this repository used to compute, and regenerating without
//! reviewing the diff throws that record away silently. The failing test names
//! the deviation so you can judge it first.
//!
//! Unlike `tools/gen_golden.sh` (which needs clang, GSL and a sibling
//! koral_lite checkout to run the C oracle), this needs nothing but the Zig
//! toolchain: the baseline comes from this repository itself.

const std = @import("std");
const koral = @import("koral");

const scenarios = koral.testing.selfscenarios;

pub fn main(init: std.process.Init) !void {
    const a = init.gpa;
    const io = init.io;

    var args = std.process.Args.Iterator.init(init.minimal.args);
    defer args.deinit();
    _ = args.next(); // program name
    const out_dir = args.next() orelse {
        std.debug.print("usage: gen_self_golden <output-dir>\n", .{});
        return error.MissingOutputDir;
    };

    try std.Io.Dir.cwd().createDirPath(io, out_dir);

    for (scenarios.all) |sc| {
        var w = try scenarios.run(a, sc);
        defer w.deinit();

        const bytes = try w.toGzip(a);
        defer a.free(bytes);

        const path = try std.fmt.allocPrint(a, "{s}/{s}", .{ out_dir, sc.file });
        defer a.free(path);
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = bytes });
        std.debug.print(
            "wrote {s}  ({s}; {d} scalar records, {d} field snapshots, {d} KiB gzipped)\n",
            .{ path, sc.name, w.hdr.n_scalar_recs, w.hdr.n_field_recs, bytes.len / 1024 },
        );
    }
}
