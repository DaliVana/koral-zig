//! Runtime grid geometry: dimensions, ghost depth, spacing, cell-center /
//! face coordinates in *internal* coordinates (uniform by construction —
//! curvature lives entirely in the metric; C: set_grid, finite.c:1910).
//!
//! Dimension collapse follows C exactly (ko.h:249-327): a dimension of
//! size 1 carries no ghost cells and its index is pinned to 0.

const std = @import("std");

pub const Grid = struct {
    /// Active cells per dimension (C: NX, NY, NZ — serial, so == TNX…).
    nx: usize,
    ny: usize,
    nz: usize,
    /// Ghost depth of the *configuration* (C: NG).
    ng: usize,
    /// Per-dimension ghost depth after collapse (C: NGCX/NGCY/NGCZ).
    ngx: usize,
    ngy: usize,
    ngz: usize,
    /// Internal-coordinate domain bounds (C: MINX/MAXX …).
    minx: f64,
    maxx: f64,
    miny: f64,
    maxy: f64,
    minz: f64,
    maxz: f64,
    /// Uniform spacing (C: get_size_x).
    dx: f64,
    dy: f64,
    dz: f64,

    pub fn init(opts: struct {
        nx: usize,
        ny: usize = 1,
        nz: usize = 1,
        ng: usize,
        minx: f64,
        maxx: f64,
        miny: f64 = 0,
        maxy: f64 = 1,
        minz: f64 = 0,
        maxz: f64 = 1,
    }) Grid {
        std.debug.assert(opts.nx >= 1 and opts.ny >= 1 and opts.nz >= 1);
        return .{
            .nx = opts.nx,
            .ny = opts.ny,
            .nz = opts.nz,
            .ng = opts.ng,
            .ngx = if (opts.nx > 1) opts.ng else 0,
            .ngy = if (opts.ny > 1) opts.ng else 0,
            .ngz = if (opts.nz > 1) opts.ng else 0,
            .minx = opts.minx,
            .maxx = opts.maxx,
            .miny = opts.miny,
            .maxy = opts.maxy,
            .minz = opts.minz,
            .maxz = opts.maxz,
            .dx = (opts.maxx - opts.minx) / @as(f64, @floatFromInt(opts.nx)),
            .dy = (opts.maxy - opts.miny) / @as(f64, @floatFromInt(opts.ny)),
            .dz = (opts.maxz - opts.minz) / @as(f64, @floatFromInt(opts.nz)),
        };
    }

    /// Padded (storage) dimensions (C: SX = NX + 2*NGCX …).
    pub fn sx(g: Grid) usize {
        return g.nx + 2 * g.ngx;
    }
    pub fn sy(g: Grid) usize {
        return g.ny + 2 * g.ngy;
    }
    pub fn sz(g: Grid) usize {
        return g.nz + 2 * g.ngz;
    }

    /// Total stored cells including ghosts.
    pub fn cellCount(g: Grid) usize {
        return g.sx() * g.sy() * g.sz();
    }

    /// Cell-center internal coordinate. Signed index: ghosts are negative /
    /// ≥ n. Computed exactly as C does (set_grid, finite.c:1930):
    /// x(i) = ½(xb(i) + xb(i+1)) — for irrational bounds this differs from
    /// min + (i+½)dx by an ulp, and everything downstream (metric cache,
    /// sweeps) must see C's bits.
    pub fn xc(g: Grid, ix: i64) f64 {
        return 0.5 * (g.xl(ix) + g.xl(ix + 1));
    }
    pub fn yc(g: Grid, iy: i64) f64 {
        return 0.5 * (g.yl(iy) + g.yl(iy + 1));
    }
    pub fn zc(g: Grid, iz: i64) f64 {
        return 0.5 * (g.zl(iz) + g.zl(iz + 1));
    }

    /// Left-face internal coordinate of cell ix (C: get_xb).
    pub fn xl(g: Grid, ix: i64) f64 {
        return g.minx + @as(f64, @floatFromInt(ix)) * g.dx;
    }
    pub fn yl(g: Grid, iy: i64) f64 {
        return g.miny + @as(f64, @floatFromInt(iy)) * g.dy;
    }
    pub fn zl(g: Grid, iz: i64) f64 {
        return g.minz + @as(f64, @floatFromInt(iz)) * g.dz;
    }

    /// Cell size as C computes it — get_size_x(i,dim) = xb(i+1) − xb(i)
    /// (finite.c:2457). For irrational bounds this can differ from the
    /// nominal spacing by an ulp *per cell*, and the C evolution uses this
    /// per-cell value everywhere, so step-level diffing requires it.
    pub fn size(g: Grid, i: i64, dim: usize) f64 {
        return switch (dim) {
            0 => g.xl(i + 1) - g.xl(i),
            1 => g.yl(i + 1) - g.yl(i),
            2 => g.zl(i + 1) - g.zl(i),
            else => unreachable,
        };
    }

};

//
// ---- M0 gate tests -------------------------------------------------------
//

test "grid: dimension collapse matches C (NY=1 → no y-ghosts, SY=1)" {
    const g = Grid.init(.{ .nx = 384, .ny = 360, .nz = 1, .ng = 3, .minx = 0, .maxx = 1, .miny = 0, .maxy = 1, .minz = -0.25, .maxz = 0.25 });
    try std.testing.expectEqual(@as(usize, 3), g.ngx);
    try std.testing.expectEqual(@as(usize, 3), g.ngy);
    try std.testing.expectEqual(@as(usize, 0), g.ngz);
    try std.testing.expectEqual(@as(usize, 390), g.sx());
    try std.testing.expectEqual(@as(usize, 366), g.sy());
    try std.testing.expectEqual(@as(usize, 1), g.sz());
}

test "grid: uniform spacing and centers/faces (incl. ghost cells)" {
    const g = Grid.init(.{ .nx = 10, .ng = 2, .minx = 0.0, .maxx = 1.0 });
    try std.testing.expectApproxEqRel(@as(f64, 0.1), g.dx, 1e-15);
    try std.testing.expectApproxEqRel(@as(f64, 0.05), g.xc(0), 1e-15);
    try std.testing.expectApproxEqRel(@as(f64, 0.95), g.xc(9), 1e-15);
    // ghost centers extend the uniform lattice
    try std.testing.expectApproxEqRel(@as(f64, -0.05), g.xc(-1), 1e-15);
    try std.testing.expectApproxEqRel(@as(f64, 1.05), g.xc(10), 1e-15);
    // faces: xl(i+1) - xl(i) = dx, xc = (xl(i)+xl(i+1))/2
    try std.testing.expectApproxEqRel(g.dx, g.xl(4) - g.xl(3), 1e-15);
    try std.testing.expectApproxEqRel(g.xc(3), 0.5 * (g.xl(3) + g.xl(4)), 1e-15);
}
