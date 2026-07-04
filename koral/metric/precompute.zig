//! Precomputed metric cache (C: calc_metric, metric.c:26).
//!
//! Stores, for every cell incl. ghosts: the 4×5 metric block g (metric +
//! dlgdet + gdet), the 4×5 inverse block G (inverse + gttpert), Christoffels
//! (64 per cell, with the gdet-trace correction below), Jacobians to/from
//! the output coordinates; and the 4×5 g/G blocks at all x/y/z faces.
//!
//! Christoffel trace correction (C: calc_Krzysie_at_center with
//! MODYFIKUJKRZYSIE, on when GDETIN=1): the trace Γ^μ_κμ is adjusted so it
//! equals the *finite-difference* gdet gradient across the cell,
//! (√−g|_{face+} − √−g|_{face−})/(Δx √−g|_c), distributing the correction
//! over the four terms by their magnitude. This makes the geometric source
//! terms telescope against the discrete flux divergence — uniform states
//! stay uniform.

const std = @import("std");
const Grid = @import("../grid.zig").Grid;
const Geometry = @import("../geometry.zig").Geometry;
const metric = @import("metric.zig");
const coco = @import("coco.zig");
const config = @import("../config.zig");

pub const Coords = config.Coords;
pub const MetricParams = metric.MetricParams;

/// C: calc_Krzysie_at_center, metric.c:1076 (MODYFIKUJKRZYSIE branch).
/// Adjusts the trace Γ^μ_κμ of `d.kris` to the FD gdet gradient across cell
/// (ix,iy,iz), weighting the four terms by their magnitudes.
pub fn applyKrisCorrection(coords: Coords, mp: MetricParams, g: *const Grid, d: *metric.CoordData, ix: i64, iy: i64, iz: i64) void {
    const xc = [4]f64{ 0, g.xc(ix), g.yc(iy), g.zc(iz) };
    const gdet_c = d.gdet;
    const dxs = [3]f64{ g.dx, g.dy, g.dz };

    for (1..4) |kappa| {
        var ck: f64 = 0;
        var sk: f64 = 1.0e-300;
        for (0..4) |mu| {
            ck += d.kris[mu][kappa][mu];
            sk += @abs(d.kris[mu][kappa][mu]);
        }

        var x_hi = xc;
        var x_lo = xc;
        switch (kappa) {
            1 => {
                x_hi[1] = g.xl(ix + 1);
                x_lo[1] = g.xl(ix);
            },
            2 => {
                x_hi[2] = g.yl(iy + 1);
                x_lo[2] = g.yl(iy);
            },
            3 => {
                x_hi[3] = g.zl(iz + 1);
                x_lo[3] = g.zl(iz);
            },
            else => unreachable,
        }
        const gdet_hi = metric.compute(coords, mp, x_hi).gdet;
        const gdet_lo = metric.compute(coords, mp, x_lo).gdet;
        const dk = (gdet_hi - gdet_lo) / (dxs[kappa - 1] * gdet_c);

        for (0..4) |mu| {
            const wk = @abs(d.kris[mu][kappa][mu]) / sk;
            d.kris[mu][kappa][mu] += (dk - ck) * wk;
            d.kris[mu][mu][kappa] = d.kris[mu][kappa][mu];
        }
    }
}

/// Metric data at one cell center with the corrected Christoffels — what
/// the cache stores, computable standalone (for tests and small tools).
pub fn computeCorrected(coords: Coords, mp: MetricParams, g: *const Grid, ix: i64, iy: i64, iz: i64) metric.CoordData {
    var d = metric.compute(coords, mp, .{ 0, g.xc(ix), g.yc(iy), g.zc(iz) });
    applyKrisCorrection(coords, mp, g, &d, ix, iy, iz);
    return d;
}

/// Geometry at an arbitrary point, computed on the fly (C: fill_geometry_arb
/// without the grid arrays). Christoffels are not part of Geometry, so no
/// trace correction applies here.
pub fn geometryAt(coords: Coords, mp: MetricParams, x: [4]f64) Geometry {
    const d = metric.compute(coords, mp, x);
    var geo = Geometry{
        .coords = coords,
        .ix = 0,
        .iy = 0,
        .iz = 0,
        .ifacedim = -1,
        .xxvec = .{ 0, x[1], x[2], x[3] },
        .gg = undefined,
        .GG = undefined,
        .gdet = d.gdet,
        .alpha = undefined,
        .gttpert = d.gttpert,
    };
    for (0..4) |i| {
        for (0..4) |j| {
            geo.gg[i][j] = d.gcov[i][j];
            geo.GG[i][j] = d.gcon[i][j];
        }
        geo.gg[i][4] = if (i < 3) d.dlgdet[i] else d.gdet;
        geo.GG[i][4] = if (i == 3) d.gttpert else 0;
    }
    geo.alpha = @sqrt(-1.0 / geo.GG[0][0]);
    return geo;
}

pub const MetricCache = struct {
    a: std.mem.Allocator,
    grid: Grid,
    coords: Coords,
    out_coords: Coords,
    mp: MetricParams,

    /// cell centers, stride 20 (C: g, G with gSIZE=20)
    g: []f64,
    gcon: []f64,
    /// Christoffels at centers, stride 64 (C: gKr)
    kris: []f64,
    /// Jacobians MYCOORDS→OUTCOORDS and back, stride 16 (C: dxdx_my2out/out2my)
    dxdx_my2out: []f64,
    dxdx_out2my: []f64,
    /// faces, stride 20; dim d has one extra slice in direction d
    gb: [3][]f64,
    gconb: [3][]f64,

    pub const InitOpts = struct {
        coords: Coords,
        out_coords: Coords,
        mp: MetricParams,
        /// C: MODYFIKUJKRZYSIE (default on, as with GDETIN=1)
        modify_kris: bool = true,
        /// skip face metrics (postprocessing mode)
        faces: bool = true,
    };

    pub fn init(a: std.mem.Allocator, grid: Grid, opts: InitOpts) !MetricCache {
        const nc = grid.cellCount();
        var self = MetricCache{
            .a = a,
            .grid = grid,
            .coords = opts.coords,
            .out_coords = opts.out_coords,
            .mp = opts.mp,
            .g = try a.alloc(f64, nc * 20),
            .gcon = try a.alloc(f64, nc * 20),
            .kris = try a.alloc(f64, nc * 64),
            .dxdx_my2out = try a.alloc(f64, nc * 16),
            .dxdx_out2my = try a.alloc(f64, nc * 16),
            .gb = undefined,
            .gconb = undefined,
        };
        for (0..3) |d| {
            const nf = faceCount(grid, d);
            self.gb[d] = try a.alloc(f64, nf * 20);
            self.gconb[d] = try a.alloc(f64, nf * 20);
            if (!opts.faces) {
                @memset(self.gb[d], 0);
                @memset(self.gconb[d], 0);
            }
        }

        self.fillCenters(opts.modify_kris);
        if (opts.faces) self.fillFaces();
        return self;
    }

    pub fn deinit(self: *MetricCache) void {
        self.a.free(self.g);
        self.a.free(self.gcon);
        self.a.free(self.kris);
        self.a.free(self.dxdx_my2out);
        self.a.free(self.dxdx_out2my);
        for (0..3) |d| {
            self.a.free(self.gb[d]);
            self.a.free(self.gconb[d]);
        }
        self.* = undefined;
    }

    fn faceCount(grid: Grid, dim: usize) usize {
        return switch (dim) {
            0 => (grid.sx() + 1) * grid.sy() * grid.sz(),
            1 => grid.sx() * (grid.sy() + 1) * grid.sz(),
            2 => grid.sx() * grid.sy() * (grid.sz() + 1),
            else => unreachable,
        };
    }

    fn cellIndex(self: *const MetricCache, ix: i64, iy: i64, iz: i64) usize {
        const g = &self.grid;
        const jx: usize = @intCast(ix + @as(i64, @intCast(g.ngx)));
        const jy: usize = @intCast(iy + @as(i64, @intCast(g.ngy)));
        const jz: usize = @intCast(iz + @as(i64, @intCast(g.ngz)));
        std.debug.assert(jx < g.sx() and jy < g.sy() and jz < g.sz());
        return jx + jy * g.sx() + jz * g.sy() * g.sx();
    }

    /// Face `dim`-index f is the *left* face of cell f in that dimension;
    /// f ranges one past the last cell.
    fn faceIndex(self: *const MetricCache, dim: usize, ix: i64, iy: i64, iz: i64) usize {
        const g = &self.grid;
        const jx: usize = @intCast(ix + @as(i64, @intCast(g.ngx)));
        const jy: usize = @intCast(iy + @as(i64, @intCast(g.ngy)));
        const jz: usize = @intCast(iz + @as(i64, @intCast(g.ngz)));
        return switch (dim) {
            0 => blk: {
                std.debug.assert(jx <= g.sx() and jy < g.sy() and jz < g.sz());
                break :blk jx + jy * (g.sx() + 1) + jz * g.sy() * (g.sx() + 1);
            },
            1 => blk: {
                std.debug.assert(jx < g.sx() and jy <= g.sy() and jz < g.sz());
                break :blk jx + jy * g.sx() + jz * (g.sy() + 1) * g.sx();
            },
            2 => blk: {
                std.debug.assert(jx < g.sx() and jy < g.sy() and jz <= g.sz());
                break :blk jx + jy * g.sx() + jz * g.sy() * g.sx();
            },
            else => unreachable,
        };
    }

    fn storeBlocks(dst_g: []f64, dst_gcon: []f64, off: usize, d: metric.CoordData) void {
        for (0..4) |i| {
            for (0..4) |j| {
                dst_g[off + i * 5 + j] = d.gcov[i][j];
                dst_gcon[off + i * 5 + j] = d.gcon[i][j];
            }
        }
        for (0..3) |j| dst_g[off + j * 5 + 4] = d.dlgdet[j];
        dst_g[off + 3 * 5 + 4] = d.gdet;
        dst_gcon[off + 3 * 5 + 4] = d.gttpert;
    }

    fn fillCenters(self: *MetricCache, modify_kris: bool) void {
        const g = &self.grid;
        var iz: i64 = -@as(i64, @intCast(g.ngz));
        while (iz < @as(i64, @intCast(g.nz + g.ngz))) : (iz += 1) {
            var iy: i64 = -@as(i64, @intCast(g.ngy));
            while (iy < @as(i64, @intCast(g.ny + g.ngy))) : (iy += 1) {
                var ix: i64 = -@as(i64, @intCast(g.ngx));
                while (ix < @as(i64, @intCast(g.nx + g.ngx))) : (ix += 1) {
                    self.fillCell(ix, iy, iz, modify_kris);
                }
            }
        }
    }

    fn fillCell(self: *MetricCache, ix: i64, iy: i64, iz: i64, modify_kris: bool) void {
        const g = &self.grid;
        const x = [4]f64{ 0, g.xc(ix), g.yc(iy), g.zc(iz) };
        var d = metric.compute(self.coords, self.mp, x);

        if (modify_kris) applyKrisCorrection(self.coords, self.mp, g, &d, ix, iy, iz);

        const ci = self.cellIndex(ix, iy, iz);
        storeBlocks(self.g, self.gcon, ci * 20, d);

        const ki = ci * 64;
        for (0..4) |i| {
            for (0..4) |j| {
                for (0..4) |k| {
                    self.kris[ki + i * 16 + j * 4 + k] = d.kris[i][j][k];
                }
            }
        }

        // Jacobians to/from output coordinates (C: PRECOMPUTE_MY2OUT)
        const di = ci * 16;
        const j_my2out = coco.dxdx(x, self.coords, self.out_coords, self.mp);
        const x_out = coco.cocoN(x, self.coords, self.out_coords, self.mp);
        const j_out2my = coco.dxdx(x_out, self.out_coords, self.coords, self.mp);
        for (0..4) |i| {
            for (0..4) |j| {
                self.dxdx_my2out[di + i * 4 + j] = j_my2out[i][j];
                self.dxdx_out2my[di + i * 4 + j] = j_out2my[i][j];
            }
        }
    }

    // (Christoffel gdet-trace correction lives at module level below so
    // tests can apply it to a single cell without building the full cache.)

    fn fillFaces(self: *MetricCache) void {
        const g = &self.grid;
        const ngx: i64 = @intCast(g.ngx);
        const ngy: i64 = @intCast(g.ngy);
        const ngz: i64 = @intCast(g.ngz);
        const nxh: i64 = @intCast(g.nx + g.ngx);
        const nyh: i64 = @intCast(g.ny + g.ngy);
        const nzh: i64 = @intCast(g.nz + g.ngz);

        var iz: i64 = -ngz;
        while (iz <= nzh) : (iz += 1) {
            var iy: i64 = -ngy;
            while (iy <= nyh) : (iy += 1) {
                var ix: i64 = -ngx;
                while (ix <= nxh) : (ix += 1) {
                    // x-face at (xl(ix), yc, zc) exists for all cell (iy,iz)
                    if (iy < nyh and iz < nzh) {
                        const x = [4]f64{ 0, g.xl(ix), g.yc(iy), g.zc(iz) };
                        const d = metric.compute(self.coords, self.mp, x);
                        storeBlocks(self.gb[0], self.gconb[0], self.faceIndex(0, ix, iy, iz) * 20, d);
                    }
                    if (ix < nxh and iz < nzh) {
                        const x = [4]f64{ 0, g.xc(ix), g.yl(iy), g.zc(iz) };
                        const d = metric.compute(self.coords, self.mp, x);
                        storeBlocks(self.gb[1], self.gconb[1], self.faceIndex(1, ix, iy, iz) * 20, d);
                    }
                    if (ix < nxh and iy < nyh) {
                        const x = [4]f64{ 0, g.xc(ix), g.yc(iy), g.zl(iz) };
                        const d = metric.compute(self.coords, self.mp, x);
                        storeBlocks(self.gb[2], self.gconb[2], self.faceIndex(2, ix, iy, iz) * 20, d);
                    }
                }
            }
        }
    }

    // ---- accessors -------------------------------------------------------

    pub fn kr(self: *const MetricCache, i: usize, j: usize, k: usize, ix: i64, iy: i64, iz: i64) f64 {
        return self.kris[self.cellIndex(ix, iy, iz) * 64 + i * 16 + j * 4 + k];
    }

    fn geometryFromBlocks(self: *const MetricCache, src_g: []const f64, src_gcon: []const f64, off: usize, ix: i64, iy: i64, iz: i64, ifacedim: i8, xxvec: [4]f64) Geometry {
        var geo = Geometry{
            .coords = self.coords,
            .ix = ix,
            .iy = iy,
            .iz = iz,
            .ifacedim = ifacedim,
            .xxvec = xxvec,
            .gg = undefined,
            .GG = undefined,
            .gdet = undefined,
            .alpha = undefined,
            .gttpert = undefined,
        };
        for (0..4) |i| {
            for (0..5) |j| {
                geo.gg[i][j] = src_g[off + i * 5 + j];
                geo.GG[i][j] = src_gcon[off + i * 5 + j];
            }
        }
        geo.gdet = geo.gg[3][4];
        geo.gttpert = geo.GG[3][4];
        geo.alpha = @sqrt(-1.0 / geo.GG[0][0]);
        return geo;
    }

    /// C: fill_geometry (metric.c:1884).
    pub fn fillGeometry(self: *const MetricCache, ix: i64, iy: i64, iz: i64) Geometry {
        const g = &self.grid;
        return self.geometryFromBlocks(
            self.g,
            self.gcon,
            self.cellIndex(ix, iy, iz) * 20,
            ix,
            iy,
            iz,
            -1,
            .{ 0, g.xc(ix), g.yc(iy), g.zc(iz) },
        );
    }

    /// C: fill_geometry_face (metric.c:1971) — left face of cell in `dim`.
    pub fn fillGeometryFace(self: *const MetricCache, ix: i64, iy: i64, iz: i64, dim: usize) Geometry {
        const g = &self.grid;
        const xxvec: [4]f64 = switch (dim) {
            0 => .{ 0, g.xl(ix), g.yc(iy), g.zc(iz) },
            1 => .{ 0, g.xc(ix), g.yl(iy), g.zc(iz) },
            2 => .{ 0, g.xc(ix), g.yc(iy), g.zl(iz) },
            else => unreachable,
        };
        return self.geometryFromBlocks(
            self.gb[dim],
            self.gconb[dim],
            self.faceIndex(dim, ix, iy, iz) * 20,
            ix,
            iy,
            iz,
            @intCast(dim),
            xxvec,
        );
    }
};
