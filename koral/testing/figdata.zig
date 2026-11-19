//! Data dumps for the paper figures (tools/paperfig.py).
//!
//! The reproduction tests assert; a figure needs the numbers they asserted
//! on. With `zig build test -Dfig-dir=DIR` every test that owns a figure
//! writes its data as a plain-text table DIR/<name>.dat, which
//! tools/paperfig.py turns into the figure. Without the option every call
//! here is a no-op, so the default suite writes nothing and costs nothing.
//!
//!     var fig = figdata.Table.init("waves_l1");
//!     defer fig.finish();
//!     fig.meta("columns", "nx l1", .{});
//!     fig.row(&.{ nx, l1 });
//!
//! Format: `# key = value` header lines, then one whitespace-separated row
//! of %.17e numbers per `row` call (numpy.loadtxt reads it as is). `block`
//! starts a new data set inside the same file: a `# block = <label>` line,
//! which the reader splits on. 2D maps are written as rows of (x, y, value)
//! with a `shape` header.
//!
//! A dump never fails a test: a file that cannot be written is reported on
//! stderr and skipped.

const std = @import("std");
const build_options = @import("build_options");

pub fn enabled() bool {
    return build_options.fig_dir.len > 0;
}

pub const Table = struct {
    active: bool,
    name: []const u8,
    aw: std.Io.Writer.Allocating,

    pub fn init(comptime name: []const u8) Table {
        return .{
            .active = enabled(),
            .name = name,
            // not the testing allocator: a dump must not show up in a
            // test's leak accounting
            .aw = .init(std.heap.page_allocator),
        };
    }

    /// `# key = <formatted value>`
    pub fn meta(self: *Table, comptime key: []const u8, comptime fmt: []const u8, args: anytype) void {
        if (!self.active) return;
        self.aw.writer.print("# " ++ key ++ " = " ++ fmt ++ "\n", args) catch {};
    }

    /// Start a new data set in the same file.
    pub fn block(self: *Table, comptime fmt: []const u8, args: anytype) void {
        self.meta("block", fmt, args);
    }

    pub fn row(self: *Table, vals: []const f64) void {
        if (!self.active) return;
        for (vals, 0..) |v, i| {
            self.aw.writer.print("{s}{e:.17}", .{ if (i == 0) "" else " ", v }) catch {};
        }
        self.aw.writer.writeByte('\n') catch {};
    }

    pub fn finish(self: *Table) void {
        defer self.aw.deinit();
        if (!self.active) return;
        const io = std.testing.io;
        const dir = build_options.fig_dir;
        std.Io.Dir.cwd().createDirPath(io, dir) catch {};
        var buf: [1024]u8 = undefined;
        const path = std.fmt.bufPrint(&buf, "{s}/{s}.dat", .{ dir, self.name }) catch return;
        std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = self.aw.written() }) catch |err| {
            std.debug.print("figdata: cannot write '{s}': {s}\n", .{ path, @errorName(err) });
            return;
        };
        std.debug.print("figdata: wrote {s}\n", .{path});
    }
};

test "figdata is inert without -Dfig-dir" {
    if (enabled()) return error.SkipZigTest;
    var t = Table.init("never_written");
    t.meta("columns", "a b", .{});
    t.row(&.{ 1.0, 2.0 });
    try std.testing.expectEqual(@as(usize, 0), t.aw.written().len);
    t.finish();
}
