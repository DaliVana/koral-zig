//! Cell-centered field storage — Array-of-Structs, variable-fastest, exactly
//! C's get_u/set_u layout (ko.h:517-523):
//!
//!   data[iv + (ix+ngx)*NV + (iy+ngy)*SX*NV + (iz+ngz)*SY*SX*NV]
//!
//! Per-cell kernels use load/store into stack buffers ([NV]f64), never raw
//! storage pointers, so the layout can later flip to AoSoA without touching
//! kernels (architecture doc §6). Signed cell indices address ghosts
//! (ix ∈ [-ngx, nx+ngx)).

const std = @import("std");
const Grid = @import("grid.zig").Grid;

pub fn Field(comptime NV: usize) type {
    return struct {
        const Self = @This();
        pub const nv = NV;

        data: []f64,
        grid: Grid,
        allocator: std.mem.Allocator,

        pub fn init(allocator: std.mem.Allocator, grid: Grid) !Self {
            const data = try allocator.alloc(f64, grid.cellCount() * NV);
            @memset(data, 0);
            return .{ .data = data, .grid = grid, .allocator = allocator };
        }

        /// Like init but WITHOUT the zero-fill: the caller must initialize
        /// `data` before any read. Exists for NUMA first-touch (MPI plan
        /// P4b): Sim.init zeroes these via the team, so the first write to
        /// each page happens on a worker and the pages distribute across
        /// the node's memory controllers instead of all landing on the
        /// main thread's.
        pub fn initUninitialized(allocator: std.mem.Allocator, grid: Grid) !Self {
            const data = try allocator.alloc(f64, grid.cellCount() * NV);
            return .{ .data = data, .grid = grid, .allocator = allocator };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.data);
            self.* = undefined;
        }

        /// Flat offset of (ix,iy,iz), C's get_u index arithmetic.
        /// `inline`: O(NV × cells × passes)/step leaf on every hot loop — the
        /// win is Debug/ReleaseSafe (ReleaseFast already inlines it). No FP
        /// change → golden-safe (P2 #10).
        pub inline fn cellOffset(self: *const Self, ix: i64, iy: i64, iz: i64) usize {
            return self.grid.cellIndex(ix, iy, iz) * NV;
        }

        pub inline fn get(self: *const Self, iv: usize, ix: i64, iy: i64, iz: i64) f64 {
            std.debug.assert(iv < NV);
            return self.data[self.cellOffset(ix, iy, iz) + iv];
        }

        pub inline fn set(self: *Self, iv: usize, ix: i64, iy: i64, iz: i64, v: f64) void {
            std.debug.assert(iv < NV);
            self.data[self.cellOffset(ix, iy, iz) + iv] = v;
        }

        /// Copy one cell's variables into a stack buffer.
        pub inline fn load(self: *const Self, ix: i64, iy: i64, iz: i64, out: *[NV]f64) void {
            const off = self.cellOffset(ix, iy, iz);
            @memcpy(out, self.data[off..][0..NV]);
        }

        /// Write one cell's variables from a stack buffer.
        pub inline fn store(self: *Self, ix: i64, iy: i64, iz: i64, pp: *const [NV]f64) void {
            const off = self.cellOffset(ix, iy, iz);
            @memcpy(self.data[off..][0..NV], pp);
        }

        pub fn fill(self: *Self, v: f64) void {
            @memset(self.data, v);
        }
    };
}

//
// ---- M0 gate tests -------------------------------------------------------
//

test "field: load/store round-trip and get/set agreement" {
    const g = Grid.init(.{ .nx = 8, .ny = 6, .nz = 1, .ng = 2, .minx = 0, .maxx = 1 });
    var f = try Field(13).init(std.testing.allocator, g);
    defer f.deinit();

    var pp: [13]f64 = undefined;
    for (&pp, 0..) |*v, i| v.* = 100.0 + @as(f64, @floatFromInt(i));
    f.store(3, 4, 0, &pp);

    var qq: [13]f64 = undefined;
    f.load(3, 4, 0, &qq);
    try std.testing.expectEqualSlices(f64, &pp, &qq);
    try std.testing.expectEqual(@as(f64, 105.0), f.get(5, 3, 4, 0));

    f.set(5, 3, 4, 0, -1.0);
    try std.testing.expectEqual(@as(f64, -1.0), f.get(5, 3, 4, 0));
}

test "field: every (iv,cell) slot incl. ghosts has a unique address" {
    // Write a unique value everywhere via set(), then verify via get() and
    // check the flat array holds exactly the expected multiset.
    const g = Grid.init(.{ .nx = 4, .ny = 3, .nz = 2, .ng = 2, .minx = 0, .maxx = 1 });
    const NV = 5;
    var f = try Field(NV).init(std.testing.allocator, g);
    defer f.deinit();

    try std.testing.expectEqual(@as(usize, (4 + 4) * (3 + 4) * (2 + 4) * NV), f.data.len);

    var counter: f64 = 1.0;
    var iz: i64 = -2;
    while (iz < 4) : (iz += 1) {
        var iy: i64 = -2;
        while (iy < 5) : (iy += 1) {
            var ix: i64 = -2;
            while (ix < 6) : (ix += 1) {
                var iv: usize = 0;
                while (iv < NV) : (iv += 1) {
                    f.set(iv, ix, iy, iz, counter);
                    counter += 1.0;
                }
            }
        }
    }
    // No slot was overwritten → all values distinct and read back in order.
    counter = 1.0;
    iz = -2;
    while (iz < 4) : (iz += 1) {
        var iy: i64 = -2;
        while (iy < 5) : (iy += 1) {
            var ix: i64 = -2;
            while (ix < 6) : (ix += 1) {
                var iv: usize = 0;
                while (iv < NV) : (iv += 1) {
                    try std.testing.expectEqual(counter, f.get(iv, ix, iy, iz));
                    counter += 1.0;
                }
            }
        }
    }
}

test "field: memory layout is iv-fastest then x (C get_u ordering)" {
    const g = Grid.init(.{ .nx = 4, .ny = 4, .nz = 1, .ng = 2, .minx = 0, .maxx = 1 });
    const NV = 3;
    var f = try Field(NV).init(std.testing.allocator, g);
    defer f.deinit();
    // Adjacent iv within one cell → adjacent memory: iv=1 must land at
    // base+1 in the raw flat array (inspect f.data directly — the address
    // function alone can't prove ordering).
    f.set(0, 0, 0, 0, 111.0);
    f.set(1, 0, 0, 0, 222.0);
    try std.testing.expectEqual(@as(f64, 222.0), f.data[f.cellOffset(0, 0, 0) + 1]);
    // Next cell in x → offset advances by exactly NV.
    try std.testing.expectEqual(f.cellOffset(0, 0, 0) + NV, f.cellOffset(1, 0, 0));
    // Next cell in y → advances by SX*NV.
    try std.testing.expectEqual(f.cellOffset(0, 0, 0) + g.sx() * NV, f.cellOffset(0, 1, 0));
    // Ghost corner is slot 0.
    try std.testing.expectEqual(@as(usize, 0), f.cellOffset(-2, -2, 0));
}
