//! Serial (single-process) communication backend — the default for local
//! runs; the MPI backend implements the same API later (architecture doc §8).
//! The numerics only ever talk to this interface.

const std = @import("std");

pub const ReduceOp = enum { sum, min, max };

pub const Serial = struct {
    pub fn init() Serial {
        return .{};
    }

    pub fn rank(_: Serial) usize {
        return 0;
    }

    pub fn size(_: Serial) usize {
        return 1;
    }

    /// Single rank: interior halos don't exist; periodic wrap is applied by
    /// the boundary stage, not here. No-op.
    pub fn exchangeHalos(_: Serial, comptime T: type, _: []T) void {}

    pub fn allreduce(_: Serial, op: ReduceOp, value: f64) f64 {
        _ = op;
        return value;
    }

    pub fn barrier(_: Serial) void {}
};

test "serial comm: identity semantics" {
    const c = Serial.init();
    try std.testing.expectEqual(@as(usize, 0), c.rank());
    try std.testing.expectEqual(@as(usize, 1), c.size());
    try std.testing.expectEqual(@as(f64, 3.5), c.allreduce(.sum, 3.5));
}
