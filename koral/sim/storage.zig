//! Generic per-cell storage and bookkeeping for the evolution driver; the
//! face-centered buffer and the integer-flag / scalar-slot enums. These are
//! plain storage utilities, not evolution logic; `sim.zig` re-exports the
//! enums so external code keeps referencing them as `sim.Flag` / `sim.Scal`.

const std = @import("std");
const Grid = @import("../grid.zig").Grid;

/// Per-cell integer flags (C: cellflag; only the ones this path uses;
/// RADSOURCETYPEFLAG is a constant under IMPLICIT_LAB_RAD_SOURCE, skipped).
pub const Flag = enum(usize) {
    entropy, // ENTROPYFLAG — entropy inversion used this step
    entropy2, // ENTROPYFLAG2 — rad-energy borrowing failed (M7)
    entropy3, // ENTROPYFLAG3 — copy_entropycount snapshot
    hd_fixup, // HDFIXUPFLAG
    rad_fixup, // RADFIXUPFLAG (M7)
    radimp_fixup, // RADIMPFIXUPFLAG (M9) — implicit solver failed here
};
pub const n_flags = @typeInfo(Flag).@"enum".fields.len;

/// Cell-scalar slots (C: ahdxl..ahdz, aradxl..aradz global arrays +
/// cell_tstepden/cell_dt). The arad slots hold the τ-limited speeds used
/// for the fluxes; the unlimited ones only feed the timestep and are not
/// stored per cell (finite.c:415-437).
///
/// Grouped per-dimension so one flux dim-pass reads a single contiguous
/// 48-byte run (ahd{l,r,m} + arad{l,r,m}) per cell instead of scattering
/// six values across the record (P5, DoD). All access goes through the
/// ahd_*/arad_* tables or named literals, so the order is free.
pub const Scal = enum(usize) {
    ahdxl,
    ahdxr,
    ahdx,
    aradxl,
    aradxr,
    aradx,
    ahdyl,
    ahdyr,
    ahdy,
    aradyl,
    aradyr,
    arady,
    ahdzl,
    ahdzr,
    ahdz,
    aradzl,
    aradzr,
    aradz,
    tstepden,
    cell_dt,
};
pub const n_scal = @typeInfo(Scal).@"enum".fields.len;
pub const ahd_l = [3]Scal{ .ahdxl, .ahdyl, .ahdzl };
pub const ahd_r = [3]Scal{ .ahdxr, .ahdyr, .ahdzr };
pub const ahd_m = [3]Scal{ .ahdx, .ahdy, .ahdz };
pub const arad_l = [3]Scal{ .aradxl, .aradyl, .aradzl };
pub const arad_r = [3]Scal{ .aradxr, .aradyr, .aradzr };
pub const arad_m = [3]Scal{ .aradx, .arady, .aradz };

/// Face-centered storage for one direction: like Field but with one extra
/// slice in its own dimension (C: ubx/uby/ubz arrays, set_ubx/get_ub).
pub fn FaceStore(comptime NV: usize) type {
    return struct {
        const Self = @This();
        data: []f64,
        nx_s: usize,
        ny_s: usize,
        nz_s: usize,
        ngx: i64,
        ngy: i64,
        ngz: i64,
        // Stored so deinit is self-managed like Field.deinit — no
        // allocator threaded at the call site (P5: single deinit convention).
        a: std.mem.Allocator,

        pub fn init(allocator: std.mem.Allocator, g: Grid, dim: usize) !Self {
            const self = try initUninitialized(allocator, g, dim);
            @memset(self.data, 0);
            return self;
        }

        /// init without the zero-fill. See Field.initUninitialized (NUMA
        /// first-touch): Sim.init zeroes via the team instead.
        pub fn initUninitialized(allocator: std.mem.Allocator, g: Grid, dim: usize) !Self {
            const nx_s = g.sx() + @intFromBool(dim == 0);
            const ny_s = g.sy() + @intFromBool(dim == 1);
            const nz_s = g.sz() + @intFromBool(dim == 2);
            const data = try allocator.alloc(f64, nx_s * ny_s * nz_s * NV);
            return .{
                .data = data,
                .nx_s = nx_s,
                .ny_s = ny_s,
                .nz_s = nz_s,
                .ngx = @intCast(g.ngx),
                .ngy = @intCast(g.ngy),
                .ngz = @intCast(g.ngz),
                .a = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.a.free(self.data);
            self.* = undefined;
        }

        // `inline` on these leaf accessors: called O(NV × faces × passes)/step
        // on the hot path; Debug/ReleaseSafe do not inline otherwise. No FP
        // change → golden-safe (P2 #10).
        inline fn offset(self: *const Self, ix: i64, iy: i64, iz: i64) usize {
            const jx: usize = @intCast(ix + self.ngx);
            const jy: usize = @intCast(iy + self.ngy);
            const jz: usize = @intCast(iz + self.ngz);
            std.debug.assert(jx < self.nx_s and jy < self.ny_s and jz < self.nz_s);
            return ((jz * self.ny_s + jy) * self.nx_s + jx) * NV;
        }

        pub inline fn get(self: *const Self, iv: usize, ix: i64, iy: i64, iz: i64) f64 {
            return self.data[self.offset(ix, iy, iz) + iv];
        }
        pub inline fn set(self: *Self, iv: usize, ix: i64, iy: i64, iz: i64, v: f64) void {
            self.data[self.offset(ix, iy, iz) + iv] = v;
        }
        pub inline fn load(self: *const Self, ix: i64, iy: i64, iz: i64, out: *[NV]f64) void {
            const off = self.offset(ix, iy, iz);
            @memcpy(out, self.data[off..][0..NV]);
        }
        pub inline fn store(self: *Self, ix: i64, iy: i64, iz: i64, pp: *const [NV]f64) void {
            const off = self.offset(ix, iy, iz);
            @memcpy(self.data[off..][0..NV], pp);
        }
        pub fn zero(self: *Self) void {
            @memset(self.data, 0);
        }
    };
}
