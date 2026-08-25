//! Serial (single-process) communication backend; the default. Mirrors
//! comm/mpi/mpi.zig's API as no-ops so the numerics and drivers are written
//! once against `comm.Backend` (comm/comm.zig selects at comptime on
//! -Dmpi). Periodic wrap stays a boundary-stage concern here, exactly as a
//! 1-rank MPI run degenerates to (validation gate 2).

const std = @import("std");
const Grid = @import("../grid.zig").Grid;

pub const Serial = struct {
    pub const enabled = false;

    pub fn initWorld() !void {}
    pub fn finalizeWorld() void {}
    /// Single process: normal error propagation is correct, so callers
    /// gate this on a multi-rank world and never reach it here.
    pub fn abortWorld(code: u8) noreturn {
        std.process.exit(code);
    }
    pub fn worldSize() usize {
        return 1;
    }

    pub fn init(ntz: usize) !Serial {
        if (ntz != 1) return error.RankCountMismatch;
        return .{};
    }
    pub fn deinit(_: *Serial) void {}

    pub fn rank(_: *const Serial) usize {
        return 0;
    }
    pub fn size(_: *const Serial) usize {
        return 1;
    }

    pub fn bindExchange(_: *Serial, _: []f64, _: Grid, _: usize) !void {}
    pub fn unbindExchange(_: *Serial) void {}
    pub fn exchange(_: *Serial) void {}

    pub fn allreduceMax(_: *const Serial, _: []f64) void {}
    pub fn allreduceMin(_: *const Serial, _: []f64) void {}
    pub fn allreduceSum(_: *const Serial, _: []f64) void {}
    pub fn barrier(_: *const Serial) void {}

    /// MPI-IO mirror so drivers compile against either backend. Statically
    /// reachable but dynamically dead: every caller is behind an `ntz > 1`
    /// branch, and Serial.init rejects ntz != 1; a serial run does file
    /// I/O through plain std.Io (the golden path), never through these.
    pub const File = struct {};
    pub fn fileCreate(_: *const Serial, _: [*:0]const u8, _: u64) !File {
        return error.SerialHasNoMpiIo;
    }
    pub fn fileOpenRead(_: *const Serial, _: [*:0]const u8) !File {
        return error.SerialHasNoMpiIo;
    }
    pub fn fileClose(_: *const Serial, _: *File) void {}
    pub fn fileSync(_: *const Serial, _: *File) void {}
    pub fn fileSize(_: *const Serial, _: *File) u64 {
        return 0;
    }
    pub fn fileWriteAtAll(_: *const Serial, _: *File, _: u64, _: []const u8) void {}
    pub fn fileWriteAt(_: *const Serial, _: *File, _: u64, _: []const u8) void {}
    pub fn fileReadAtAll(_: *const Serial, _: *File, _: u64, _: []u8) void {}

    /// Single process: there is no job to tear down, so normal error
    /// propagation is correct and callers must never reach this. (The MPI
    /// backend's version never returns. See comm/mpi/mpi.zig.)
    pub fn abortJob(_: *const Serial, code: u8) noreturn {
        std.process.exit(code);
    }
};

test "serial comm: identity semantics" {
    try Serial.initWorld();
    defer Serial.finalizeWorld();
    var c = try Serial.init(1);
    defer c.deinit();
    try std.testing.expectEqual(@as(usize, 0), c.rank());
    try std.testing.expectEqual(@as(usize, 1), c.size());
    var buf = [3]f64{ 1.0, -2.5, 3.0 };
    c.allreduceMax(buf[0..]); // in-place identity
    try std.testing.expectEqual([3]f64{ 1.0, -2.5, 3.0 }, buf);
    try std.testing.expectError(error.RankCountMismatch, Serial.init(4));
}
