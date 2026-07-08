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
        // Only gdet at the two faces is needed; gdetAt runs gcovDual + det4
        // and skips the inversion + Christoffels compute() would also build
        // (bit-identical to compute().gdet — P2 #9).
        const gdet_hi = metric.gdetAt(coords, mp, x_hi);
        const gdet_lo = metric.gdetAt(coords, mp, x_lo);
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

/// BL (OUTCOORDS) geometry at a cell center — C: fill_geometry_arb(...,
/// OUTCOORDS) with OUTCOORDS == BLCOORDS. Transforms the cell-center MYCOORDS
/// position to Boyer–Lindquist and evaluates the Kerr-BL metric there. The
/// single home for this reduction, shared by the diagnostics (io/scalars),
/// the dynamo (magn/dynamo), and PUFFY init (problems/puffy) — see those
/// modules' thin wrappers.
pub fn geometryBLat(g: *const Grid, coords: Coords, mp: MetricParams, ix: i64, iy: i64, iz: i64) Geometry {
    const xx = [4]f64{ 0.0, g.xc(ix), g.yc(iy), g.zc(iz) };
    const xxbl = coco.cocoN(xx, coords, .bl, mp);
    return geometryAt(.bl, mp, xxbl);
}

pub const MetricCache = struct {
    allocator: std.mem.Allocator,
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
    /// BL (OUTCOORDS) geometry per cell — one entry per cell (incl. ghosts),
    /// filled only when `bl_cache` is set (the dynamo / radviscosity passes).
    /// Empty otherwise. Each entry is bit-identical to geometryBLat(...) and
    /// its .xxvec carries the cell's BL {r, θ}, so callers read those without
    /// re-running cocoN. Paired with out_coords == .bl, dxdx_my2out then holds
    /// the MYCOORDS→BL point Jacobian (see jacMy2Bl). Finding #1.
    bl_geom: []Geometry,

    pub const InitOpts = struct {
        coords: Coords,
        out_coords: Coords,
        mp: MetricParams,
        /// C: MODYFIKUJKRZYSIE (default on, as with GDETIN=1)
        modify_kris: bool = true,
        /// skip face metrics (postprocessing mode)
        faces: bool = true,
        /// precompute the per-cell BL (OUTCOORDS) geometry sidecar. Requires
        /// out_coords == .bl and a Kerr `coords` (the cocoN(.bl) target).
        bl_cache: bool = false,
    };

    pub fn init(allocator: std.mem.Allocator, grid: Grid, opts: InitOpts) !MetricCache {
        const nc = grid.cellCount();
        // Deliberate memory-for-simplicity trade: g/gcon (gSIZE=20) and kris
        // (64) are stored full-size even though gcov/gcon are symmetric and
        // kris is symmetric in its lower indices — matching C's flat layouts so
        // the storeBlocks/geometryFromBlocks/kr accessors are a 1:1 port. The
        // cache layout is not part of the golden contract, so if 3D memory
        // pressure appears these can be compacted to triangles behind the same
        // accessors (P5 accounting).
        var self = MetricCache{
            .allocator = allocator,
            .grid = grid,
            .coords = opts.coords,
            .out_coords = opts.out_coords,
            .mp = opts.mp,
            .g = try allocator.alloc(f64, nc * 20),
            .gcon = try allocator.alloc(f64, nc * 20),
            .kris = try allocator.alloc(f64, nc * 64),
            .dxdx_my2out = try allocator.alloc(f64, nc * 16),
            .dxdx_out2my = try allocator.alloc(f64, nc * 16),
            .gb = undefined,
            .gconb = undefined,
            .bl_geom = try allocator.alloc(Geometry, if (opts.bl_cache) nc else 0),
        };
        for (0..3) |d| {
            const nf = faceCount(grid, d);
            self.gb[d] = try allocator.alloc(f64, nf * 20);
            self.gconb[d] = try allocator.alloc(f64, nf * 20);
            if (!opts.faces) {
                @memset(self.gb[d], 0);
                @memset(self.gconb[d], 0);
            }
        }

        self.fillCenters(opts.modify_kris);
        if (opts.faces) self.fillFaces();
        if (opts.bl_cache) self.fillBl();
        return self;
    }

    pub fn deinit(self: *MetricCache) void {
        self.allocator.free(self.g);
        self.allocator.free(self.gcon);
        self.allocator.free(self.kris);
        self.allocator.free(self.dxdx_my2out);
        self.allocator.free(self.dxdx_out2my);
        self.allocator.free(self.bl_geom);
        for (0..3) |d| {
            self.allocator.free(self.gb[d]);
            self.allocator.free(self.gconb[d]);
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

    inline fn cellIndex(self: *const MetricCache, ix: i64, iy: i64, iz: i64) usize {
        const grid = &self.grid;
        const jx: usize = @intCast(ix + @as(i64, @intCast(grid.ngx)));
        const jy: usize = @intCast(iy + @as(i64, @intCast(grid.ngy)));
        const jz: usize = @intCast(iz + @as(i64, @intCast(grid.ngz)));
        std.debug.assert(jx < grid.sx() and jy < grid.sy() and jz < grid.sz());
        return jx + jy * grid.sx() + jz * grid.sy() * grid.sx();
    }

    /// Face `dim`-index f is the *left* face of cell f in that dimension;
    /// f ranges one past the last cell.
    inline fn faceIndex(self: *const MetricCache, dim: usize, ix: i64, iy: i64, iz: i64) usize {
        const grid = &self.grid;
        const jx: usize = @intCast(ix + @as(i64, @intCast(grid.ngx)));
        const jy: usize = @intCast(iy + @as(i64, @intCast(grid.ngy)));
        const jz: usize = @intCast(iz + @as(i64, @intCast(grid.ngz)));
        return switch (dim) {
            0 => blk: {
                std.debug.assert(jx <= grid.sx() and jy < grid.sy() and jz < grid.sz());
                break :blk jx + jy * (grid.sx() + 1) + jz * grid.sy() * (grid.sx() + 1);
            },
            1 => blk: {
                std.debug.assert(jx < grid.sx() and jy <= grid.sy() and jz < grid.sz());
                break :blk jx + jy * grid.sx() + jz * (grid.sy() + 1) * grid.sx();
            },
            2 => blk: {
                std.debug.assert(jx < grid.sx() and jy < grid.sy() and jz <= grid.sz());
                break :blk jx + jy * grid.sx() + jz * grid.sy() * grid.sx();
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
        // gcon column 4: row 3 holds gttpert; rows 0..2 are unused but MUST be
        // written so the whole 4×5 block is deterministic — geometryFromBlocks
        // copies all 20 slots into Geometry.GG, and the on-the-fly geometryAt
        // (line 103) explicitly sets GG[i][4] = 0 for i<3. Leaving them at
        // alloc-garbage made the cached and recomputed Geometry differ bit-for-
        // bit and tripped msan/valgrind (P1 correctness).
        for (0..3) |i| dst_gcon[off + i * 5 + 4] = 0;
        dst_gcon[off + 3 * 5 + 4] = d.gttpert;
    }

    fn fillCenters(self: *MetricCache, modify_kris: bool) void {
        const grid = &self.grid;
        var iz: i64 = -@as(i64, @intCast(grid.ngz));
        while (iz < @as(i64, @intCast(grid.nz + grid.ngz))) : (iz += 1) {
            var iy: i64 = -@as(i64, @intCast(grid.ngy));
            while (iy < @as(i64, @intCast(grid.ny + grid.ngy))) : (iy += 1) {
                var ix: i64 = -@as(i64, @intCast(grid.ngx));
                while (ix < @as(i64, @intCast(grid.nx + grid.ngx))) : (ix += 1) {
                    self.fillCell(ix, iy, iz, modify_kris);
                }
            }
        }
    }

    fn fillCell(self: *MetricCache, ix: i64, iy: i64, iz: i64, modify_kris: bool) void {
        const grid = &self.grid;
        const x = [4]f64{ 0, grid.xc(ix), grid.yc(iy), grid.zc(iz) };
        var d = metric.compute(self.coords, self.mp, x);

        if (modify_kris) applyKrisCorrection(self.coords, self.mp, grid, &d, ix, iy, iz);

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

    /// Fill the BL (OUTCOORDS) geometry sidecar over every cell incl. ghosts.
    /// Each entry equals geometryBLat(&grid, coords, mp, ix, iy, iz) exactly —
    /// same cocoN → geometryAt(.bl) chain — so cached reads are bit-identical.
    fn fillBl(self: *MetricCache) void {
        const grid = &self.grid;
        var iz: i64 = -@as(i64, @intCast(grid.ngz));
        while (iz < @as(i64, @intCast(grid.nz + grid.ngz))) : (iz += 1) {
            var iy: i64 = -@as(i64, @intCast(grid.ngy));
            while (iy < @as(i64, @intCast(grid.ny + grid.ngy))) : (iy += 1) {
                var ix: i64 = -@as(i64, @intCast(grid.ngx));
                while (ix < @as(i64, @intCast(grid.nx + grid.ngx))) : (ix += 1) {
                    const xx = [4]f64{ 0, grid.xc(ix), grid.yc(iy), grid.zc(iz) };
                    const xxbl = coco.cocoN(xx, self.coords, .bl, self.mp);
                    self.bl_geom[self.cellIndex(ix, iy, iz)] = geometryAt(.bl, self.mp, xxbl);
                }
            }
        }
    }

    fn fillFaces(self: *MetricCache) void {
        const grid = &self.grid;
        const ngx: i64 = @intCast(grid.ngx);
        const ngy: i64 = @intCast(grid.ngy);
        const ngz: i64 = @intCast(grid.ngz);
        const nxh: i64 = @intCast(grid.nx + grid.ngx);
        const nyh: i64 = @intCast(grid.ny + grid.ngy);
        const nzh: i64 = @intCast(grid.nz + grid.ngz);

        var iz: i64 = -ngz;
        while (iz <= nzh) : (iz += 1) {
            var iy: i64 = -ngy;
            while (iy <= nyh) : (iy += 1) {
                var ix: i64 = -ngx;
                while (ix <= nxh) : (ix += 1) {
                    // x-face at (xl(ix), yc, zc) exists for all cell (iy,iz)
                    if (iy < nyh and iz < nzh) {
                        const x = [4]f64{ 0, grid.xl(ix), grid.yc(iy), grid.zc(iz) };
                        const d = metric.compute(self.coords, self.mp, x);
                        storeBlocks(self.gb[0], self.gconb[0], self.faceIndex(0, ix, iy, iz) * 20, d);
                    }
                    if (ix < nxh and iz < nzh) {
                        const x = [4]f64{ 0, grid.xc(ix), grid.yl(iy), grid.zc(iz) };
                        const d = metric.compute(self.coords, self.mp, x);
                        storeBlocks(self.gb[1], self.gconb[1], self.faceIndex(1, ix, iy, iz) * 20, d);
                    }
                    if (ix < nxh and iy < nyh) {
                        const x = [4]f64{ 0, grid.xc(ix), grid.yc(iy), grid.zl(iz) };
                        const d = metric.compute(self.coords, self.mp, x);
                        storeBlocks(self.gb[2], self.gconb[2], self.faceIndex(2, ix, iy, iz) * 20, d);
                    }
                }
            }
        }
    }

    // ---- accessors -------------------------------------------------------

    pub inline fn kr(self: *const MetricCache, i: usize, j: usize, k: usize, ix: i64, iy: i64, iz: i64) f64 {
        return self.kris[self.cellIndex(ix, iy, iz) * 64 + i * 16 + j * 4 + k];
    }

    /// The full 64-entry Christoffel block Γ^i_jk (flattened i·16 + j·4 + k)
    /// at a cell center — fetched once so the metric-source and shear loops
    /// index it directly instead of recomputing cellIndex on every one of the
    /// 64/128 kr() calls per cell. Bit-identical to kr(i,j,k,ix,iy,iz).
    pub inline fn krBlock(self: *const MetricCache, ix: i64, iy: i64, iz: i64) *const [64]f64 {
        return self.kris[self.cellIndex(ix, iy, iz) * 64 ..][0..64];
    }

    /// √−g at a cell center — the block store's gdet slot (off + 3·5 + 4),
    /// identical to fillGeometry(ix,iy,iz).gdet without building a Geometry.
    /// Finding #1: the calc_BfromA_core hot path only needs this scalar.
    pub fn gdet(self: *const MetricCache, ix: i64, iy: i64, iz: i64) f64 {
        return self.g[self.cellIndex(ix, iy, iz) * 20 + 19];
    }

    /// BL (OUTCOORDS) geometry at a cell center — requires the bl_cache
    /// sidecar. Bit-identical to precompute.geometryBLat(&grid, coords, mp,
    /// ix, iy, iz); its .xxvec holds the cell's BL {r, θ}.
    pub fn blGeom(self: *const MetricCache, ix: i64, iy: i64, iz: i64) *const Geometry {
        std.debug.assert(self.bl_geom.len != 0); // bl_cache must be set
        return &self.bl_geom[self.cellIndex(ix, iy, iz)];
    }

    /// The MYCOORDS→BL point Jacobian ∂x_BL/∂x_my at a cell center — the
    /// dxdx_my2out store when out_coords == .bl. Equals
    /// coco.dxdx({0, xc, yc, zc}, coords, .bl, mp), i.e. exactly the Jacobian
    /// trans_pmhd_coco recomputes; used by transPmhdCocoJ (finding #1).
    pub fn jacMy2Bl(self: *const MetricCache, ix: i64, iy: i64, iz: i64) [4][4]f64 {
        std.debug.assert(self.out_coords == .bl);
        const di = self.cellIndex(ix, iy, iz) * 16;
        var j: [4][4]f64 = undefined;
        for (0..4) |i| {
            for (0..4) |k| j[i][k] = self.dxdx_my2out[di + i * 4 + k];
        }
        return j;
    }

    fn geometryFromBlocks(self: *const MetricCache, src_g: []const f64, src_gcon: []const f64, off: usize, xxvec: [4]f64) Geometry {
        var geo = Geometry{
            .coords = self.coords,
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
        const grid = &self.grid;
        return self.geometryFromBlocks(
            self.g,
            self.gcon,
            self.cellIndex(ix, iy, iz) * 20,
            .{ 0, grid.xc(ix), grid.yc(iy), grid.zc(iz) },
        );
    }

    /// C: fill_geometry_face (metric.c:1971) — left face of cell in `dim`.
    pub fn fillGeometryFace(self: *const MetricCache, ix: i64, iy: i64, iz: i64, dim: usize) Geometry {
        const grid = &self.grid;
        const xxvec: [4]f64 = switch (dim) {
            0 => .{ 0, grid.xl(ix), grid.yc(iy), grid.zc(iz) },
            1 => .{ 0, grid.xc(ix), grid.yl(iy), grid.zc(iz) },
            2 => .{ 0, grid.xc(ix), grid.yc(iy), grid.zl(iz) },
            else => unreachable,
        };
        return self.geometryFromBlocks(
            self.gb[dim],
            self.gconb[dim],
            self.faceIndex(dim, ix, iy, iz) * 20,
            xxvec,
        );
    }
};
