//! Serial (single-process) communication backend — the default. Mirrors
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

    /// Single process: there is no job to tear down, so normal error
    /// propagation is correct and callers must never reach this. (The MPI
    /// backend's version never returns — see comm/mpi/mpi.zig.)
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
