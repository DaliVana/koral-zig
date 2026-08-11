//! Boundary conditions for the evolution driver, written SimT-generic (the
//! house `fn f(comptime SimT: type, sim: *SimT)` pattern, like magn/ct.zig).
//! This is the region the planned MPI backend grows into (comm/serial.zig:
//! "the MPI backend implements the same API later") and maps directly onto C
//! finite.c:2805 (set_bc) / finite.c:3203 (2D corner filling). `sim.zig`
//! exposes a thin `Sim.setBc` method that delegates here, and re-exports
//! `BcKind`/`BcFace` so problems and tests keep referencing `sim.BcFace`.
//!
//! P1: the three ghost-face fills run band-parallel (x faces over iy rows,
//! y/z faces over ix columns) — each ghost cell is written exactly once and
//! reads only domain cells (or the user BC), so banding is bit-identical to
//! serial. The corner fill stays serial: it is ~4·(2·NG−1) cells and reads
//! the just-filled face surfaces (a natural barrier).

const p2u_mod = @import("../p2u.zig");
const threading = @import("threading.zig");

const Error = @import("../relele.zig").Error || error{OutOfMemory};

/// Boundary handling per axis (C: PERIODIC_?BC / COPY_?BC / SPECIFIC_BC).
pub const BcKind = enum { periodic, copy, specific };

/// Which boundary a ghost cell belongs to (C: XBCLO..ZBCHI).
pub const BcFace = enum { xlo, xhi, ylo, yhi, zlo, zhi };

/// C: set_bc (finite.c:2805) — ghost cells (no corners), then for MHD
/// ("MPI4CORNERS") builds the 2D corner surfaces + diagonals
/// (finite.c:3203-3403; serial, so mpi_isitBC ≡ 1).
///
/// MPI (plan §5.2, φ-only decomposition — x/y faces are ALWAYS physical):
/// when ntz>1 the z faces are interior. Mirroring C's gating exactly:
///  * z-ghost faces: p2u only, from the exchanged primitives
///    (finite.c:2862-2870, the `mpi_isitBC(BCtype)==0` branch);
///  * corner fill: x-y edge tubes still run over the FULL z-span
///    (finite.c:3865+ — both their faces are physical), x-z/y-z tubes and
///    the 8 corner cubes are skipped (forcorners(ZBC*)==0, perz==1);
///  * then C's "corners in the middle" pass (finite.c:4463+/4569+): the
///    x-BC and y-BC are applied INTO the z-ghost slices, overwriting the
///    exchange-delivered transverse ghost columns with fresh local fills.
pub fn setBc(comptime SimT: type, sim: *SimT, t: f64, ifinit: bool) Error!void {
    const g = &sim.grid;
    const Ctx = BcCtx(SimT);
    var ctx = Ctx{ .sim = sim, .t = t, .ifinit = ifinit };
    const z_physical = sim.isPhysicalBoundary(.zlo); // == .zhi (whole ring)

    // x boundaries (ghost columns, domain rows) — banded over iy
    if (g.ngx > 0) {
        try threading.parallelRangeErr(Ctx, &ctx, sim.team, 0, sim.nyi(), xFacesFn(SimT));
    }
    // y boundaries — banded over ix
    if (g.ngy > 0) {
        try threading.parallelRangeErr(Ctx, &ctx, sim.team, 0, sim.nxi(), yFacesFn(SimT));
    }
    // z boundaries — banded over ix; interior z faces get p2u-of-exchanged-p
    if (g.ngz > 0) {
        if (z_physical)
            try threading.parallelRangeErr(Ctx, &ctx, sim.team, 0, sim.nxi(), zFacesFn(SimT))
        else
            try threading.parallelRangeErr(Ctx, &ctx, sim.team, 0, sim.nxi(), zGhostP2uFn(SimT));
    }

    if (comptime SimT.Cfg.has(.mhd)) {
        if (g.ny > 1 and g.nz == 1) {
            try fillCorners2d(SimT, sim);
        } else if (g.nz > 1 and g.ny > 1) {
            try fillCorners3d(SimT, sim, z_physical);
            if (!z_physical) try fillMiddleCornersZ(SimT, sim, t, ifinit);
        } else if (g.nz > 1) {
            // x-z plane (TNY==1 && TNZ>1): no target needs it (finite.c:3406).
            @panic("Sim.setBc: x-z 2D corner filling not implemented");
        }
    }
}

fn BcCtx(comptime SimT: type) type {
    return struct { sim: *SimT, t: f64, ifinit: bool };
}

fn xFacesFn(comptime SimT: type) fn (*BcCtx(SimT), i64, i64) Error!void {
    return struct {
        fn w(c: *BcCtx(SimT), iy0: i64, iy1: i64) Error!void {
            return xFacesBand(SimT, c, iy0, iy1);
        }
    }.w;
}

fn xFacesBand(comptime SimT: type, c: *BcCtx(SimT), iy0: i64, iy1: i64) Error!void {
    const sim = c.sim;
    const nx = sim.nxi();
    const nz = sim.nzi();
    const ngx: i64 = @intCast(sim.grid.ngx);
    var iz: i64 = 0;
    while (iz < nz) : (iz += 1) {
        var iy: i64 = iy0;
        while (iy < iy1) : (iy += 1) {
            var i: i64 = 1;
            while (i <= ngx) : (i += 1) {
                try setBcCell(SimT, sim, -i, iy, iz, c.t, c.ifinit, .xlo);
                try setBcCell(SimT, sim, nx - 1 + i, iy, iz, c.t, c.ifinit, .xhi);
            }
        }
    }
}

fn yFacesFn(comptime SimT: type) fn (*BcCtx(SimT), i64, i64) Error!void {
    return struct {
        fn w(c: *BcCtx(SimT), ix0: i64, ix1: i64) Error!void {
            return yFacesBand(SimT, c, ix0, ix1);
        }
    }.w;
}

fn yFacesBand(comptime SimT: type, c: *BcCtx(SimT), ix0: i64, ix1: i64) Error!void {
    const sim = c.sim;
    const ny = sim.nyi();
    const nz = sim.nzi();
    const ngy: i64 = @intCast(sim.grid.ngy);
    var iz: i64 = 0;
    while (iz < nz) : (iz += 1) {
        var ix: i64 = ix0;
        while (ix < ix1) : (ix += 1) {
            var i: i64 = 1;
            while (i <= ngy) : (i += 1) {
                try setBcCell(SimT, sim, ix, -i, iz, c.t, c.ifinit, .ylo);
                try setBcCell(SimT, sim, ix, ny - 1 + i, iz, c.t, c.ifinit, .yhi);
            }
        }
    }
}

fn zFacesFn(comptime SimT: type) fn (*BcCtx(SimT), i64, i64) Error!void {
    return struct {
        fn w(c: *BcCtx(SimT), ix0: i64, ix1: i64) Error!void {
            return zFacesBand(SimT, c, ix0, ix1);
        }
    }.w;
}

fn zFacesBand(comptime SimT: type, c: *BcCtx(SimT), ix0: i64, ix1: i64) Error!void {
    const sim = c.sim;
    const ny = sim.nyi();
    const nz = sim.nzi();
    const ngz: i64 = @intCast(sim.grid.ngz);
    var iy: i64 = 0;
    while (iy < ny) : (iy += 1) {
        var ix: i64 = ix0;
        while (ix < ix1) : (ix += 1) {
            var i: i64 = 1;
            while (i <= ngz) : (i += 1) {
                try setBcCell(SimT, sim, ix, iy, -i, c.t, c.ifinit, .zlo);
                try setBcCell(SimT, sim, ix, iy, nz - 1 + i, c.t, c.ifinit, .zhi);
            }
        }
    }
}

/// MPI z-interior variant of the z-face pass (C finite.c:2862-2870, the
/// `mpi_isitBC(BCtype)==0` branch): the exchange already deposited fresh
/// primitives in the z-ghost planes; recompute only the conserveds there.
/// Same cell set as zFacesBand (domain ix/iy — corners are treated later).
fn zGhostP2uFn(comptime SimT: type) fn (*BcCtx(SimT), i64, i64) Error!void {
    return struct {
        fn w(c: *BcCtx(SimT), ix0: i64, ix1: i64) Error!void {
            return zGhostP2uBand(SimT, c, ix0, ix1);
        }
    }.w;
}

fn zGhostP2uBand(comptime SimT: type, c: *BcCtx(SimT), ix0: i64, ix1: i64) Error!void {
    const sim = c.sim;
    const ny = sim.nyi();
    const nz = sim.nzi();
    const ngz: i64 = @intCast(sim.grid.ngz);
    var iy: i64 = 0;
    while (iy < ny) : (iy += 1) {
        var ix: i64 = ix0;
        while (ix < ix1) : (ix += 1) {
            var i: i64 = 1;
            while (i <= ngz) : (i += 1) {
                try p2uCell(SimT, sim, ix, iy, -i);
                try p2uCell(SimT, sim, ix, iy, nz - 1 + i);
            }
        }
    }
}

fn setBcCell(comptime SimT: type, sim: *SimT, ix: i64, iy: i64, iz: i64, t: f64, ifinit: bool, face: BcFace) Error!void {
    const cfg = SimT.Cfg;
    const NV = SimT.nv;
    const kind: BcKind = switch (face) {
        .xlo, .xhi => sim.opt.bc_x,
        .ylo, .yhi => sim.opt.bc_y,
        .zlo, .zhi => sim.opt.bc_z,
    };
    var pp: [NV]f64 = undefined;
    switch (kind) {
        .specific => {
            pp = try sim.opt.specific_bc.?(sim.opt.bc_ctx, sim, ix, iy, iz, t, ifinit, face);
        },
        .periodic, .copy => {
            var iix = ix;
            var iiy = iy;
            var iiz = iz;
            const nx = sim.nxi();
            const ny = sim.nyi();
            const nz = sim.nzi();
            switch (face) {
                .xlo, .xhi => switch (kind) {
                    .periodic => {
                        if (ix < 0) iix = ix + nx;
                        if (ix > nx - 1) iix = ix - nx;
                    },
                    .copy => {
                        if (ix < 0) iix = 0;
                        if (ix > nx - 1) iix = nx - 1;
                    },
                    else => unreachable,
                },
                .ylo, .yhi => switch (kind) {
                    .periodic => {
                        if (iy < 0) iiy = iy + ny;
                        if (iy > ny - 1) iiy = iy - ny;
                        // C quirk (finite.c:2756): NY<NG pins to 0
                        if (ny < @as(i64, @intCast(sim.grid.ng))) iiy = 0;
                    },
                    .copy => {
                        if (iy < 0) iiy = 0;
                        if (iy > ny - 1) iiy = ny - 1;
                    },
                    else => unreachable,
                },
                .zlo, .zhi => switch (kind) {
                    .periodic => {
                        if (iz < 0) iiz = iz + nz;
                        if (iz > nz - 1) iiz = iz - nz;
                        if (nz < @as(i64, @intCast(sim.grid.ng))) iiz = 0;
                    },
                    .copy => {
                        if (iz < 0) iiz = 0;
                        if (iz > nz - 1) iiz = nz - 1;
                    },
                    else => unreachable,
                },
            }
            sim.p.load(iix, iiy, iiz, &pp);
        },
    }
    const geom = sim.cache.fillGeometry(ix, iy, iz);
    const uu = try p2u_mod.p2u(cfg, pp, &geom, sim.opt.gam);
    sim.p.store(ix, iy, iz, &pp);
    sim.u.store(ix, iy, iz, &uu);
}

/// p2u one ghost cell from its (already stored) primitives.
fn p2uCell(comptime SimT: type, sim: *SimT, ix: i64, iy: i64, iz: i64) Error!void {
    const cfg = SimT.Cfg;
    const NV = SimT.nv;
    var pp: [NV]f64 = undefined;
    sim.p.load(ix, iy, iz, &pp);
    const geom = sim.cache.fillGeometry(ix, iy, iz);
    const uu = try p2u_mod.p2u(cfg, pp, &geom, sim.opt.gam);
    sim.u.store(ix, iy, iz, &uu);
}

fn copyCellP(comptime SimT: type, sim: *SimT, dix: i64, diy: i64, six: i64, siy: i64) void {
    const NV = SimT.nv;
    var pp: [NV]f64 = undefined;
    sim.p.load(six, siy, 0, &pp);
    sim.p.store(dix, diy, 0, &pp);
}

fn avgCellP(comptime SimT: type, sim: *SimT, dix: i64, diy: i64, ax: i64, ay: i64, bx: i64, by: i64) void {
    const NV = SimT.nv;
    var pa: [NV]f64 = undefined;
    var pb: [NV]f64 = undefined;
    sim.p.load(ax, ay, 0, &pa);
    sim.p.load(bx, by, 0, &pb);
    var pp: [NV]f64 = undefined;
    for (0..NV) |iv| pp[iv] = 0.5 * (pa[iv] + pb[iv]);
    sim.p.store(dix, diy, 0, &pp);
}

/// finite.c:3203-3403 — 2D (TNZ==1) total-corner filling: NG−1 deep
/// one-cell surfaces copied from the adjacent domain row/column, then
/// two diagonal cells averaged (periodic runs wrap the diagonals).
fn fillCorners2d(comptime SimT: type, sim: *SimT) Error!void {
    const nx = sim.nxi();
    const ny = sim.nyi();
    const ng: i64 = @intCast(sim.grid.ng);
    const per_x = sim.opt.bc_x == .periodic;
    const per_y = sim.opt.bc_y == .periodic;

    // bottom-left
    {
        var i: i64 = 0;
        while (i < ng - 1) : (i += 1) {
            copyCellP(SimT, sim, -ng + i, -1, -ng + i, 0);
            try p2uCell(SimT, sim, -ng + i, -1, 0);
            copyCellP(SimT, sim, -1, -ng + i, 0, -ng + i);
            try p2uCell(SimT, sim, -1, -ng + i, 0);
        }
        var s1 = [4]i64{ -1, 0, 0, -1 }; // ix1,iy1,ix2,iy2
        if (per_y) s1 = .{ -1, ny - 1, -1, ny - 1 };
        if (per_x) s1 = .{ nx - 1, -1, nx - 1, -1 };
        avgCellP(SimT, sim, -1, -1, s1[0], s1[1], s1[2], s1[3]);
        try p2uCell(SimT, sim, -1, -1, 0);

        var s2 = [4]i64{ -2, -1, -1, -2 };
        if (per_y) s2 = .{ -2, ny - 2, -2, ny - 2 };
        if (per_x) s2 = .{ nx - 2, -2, nx - 2, -2 };
        avgCellP(SimT, sim, -2, -2, s2[0], s2[1], s2[2], s2[3]);
        try p2uCell(SimT, sim, -2, -2, 0);
    }
    // top-left
    {
        var i: i64 = 0;
        while (i < ng - 1) : (i += 1) {
            copyCellP(SimT, sim, -ng + i, ny, -ng + i, ny - 1);
            try p2uCell(SimT, sim, -ng + i, ny, 0);
            copyCellP(SimT, sim, -1, ny + i + 1, 0, ny + i + 1);
            try p2uCell(SimT, sim, -1, ny + i + 1, 0);
        }
        var s1 = [4]i64{ -1, ny - 1, 0, ny };
        if (per_y) s1 = .{ -1, 0, -1, 0 };
        if (per_x) s1 = .{ nx - 1, ny, nx - 1, ny };
        avgCellP(SimT, sim, -1, ny, s1[0], s1[1], s1[2], s1[3]);
        try p2uCell(SimT, sim, -1, ny, 0);

        var s2 = [4]i64{ -2, ny, -1, ny + 1 };
        if (per_y) s2 = .{ -2, 1, -2, 1 };
        if (per_x) s2 = .{ nx - 2, ny + 1, nx - 2, ny + 1 };
        avgCellP(SimT, sim, -2, ny + 1, s2[0], s2[1], s2[2], s2[3]);
        try p2uCell(SimT, sim, -2, ny + 1, 0);
    }
    // bottom-right
    {
        var i: i64 = 0;
        while (i < ng - 1) : (i += 1) {
            copyCellP(SimT, sim, nx + i + 1, -1, nx + i + 1, 0);
            try p2uCell(SimT, sim, nx + i + 1, -1, 0);
            copyCellP(SimT, sim, nx, -ng + i, nx - 1, -ng + i);
            try p2uCell(SimT, sim, nx, -ng + i, 0);
        }
        var s1 = [4]i64{ nx - 1, -1, nx, 0 };
        if (per_y) s1 = .{ nx, ny - 1, nx, ny - 1 };
        if (per_x) s1 = .{ 0, -1, 0, -1 };
        avgCellP(SimT, sim, nx, -1, s1[0], s1[1], s1[2], s1[3]);
        try p2uCell(SimT, sim, nx, -1, 0);

        var s2 = [4]i64{ nx, -2, nx + 1, -1 };
        if (per_y) s2 = .{ nx + 1, ny - 2, nx + 1, ny - 2 };
        if (per_x) s2 = .{ 1, -2, 1, -2 };
        avgCellP(SimT, sim, nx + 1, -2, s2[0], s2[1], s2[2], s2[3]);
        try p2uCell(SimT, sim, nx + 1, -2, 0);
    }
    // top-right
    {
        var i: i64 = 0;
        while (i < ng - 1) : (i += 1) {
            copyCellP(SimT, sim, nx + i + 1, ny, nx + i + 1, ny - 1);
            try p2uCell(SimT, sim, nx + i + 1, ny, 0);
            copyCellP(SimT, sim, nx, ny + i + 1, nx - 1, ny + i + 1);
            try p2uCell(SimT, sim, nx, ny + i + 1, 0);
        }
        var s1 = [4]i64{ nx - 1, ny, nx, ny - 1 };
        if (per_y) s1 = .{ nx, 0, nx, 0 };
        if (per_x) s1 = .{ 0, ny, 0, ny };
        avgCellP(SimT, sim, nx, ny, s1[0], s1[1], s1[2], s1[3]);
        try p2uCell(SimT, sim, nx, ny, 0);

        var s2 = [4]i64{ nx, ny + 1, nx + 1, ny };
        if (per_y) s2 = .{ nx + 1, 1, nx + 1, 1 };
        if (per_x) s2 = .{ 1, ny + 1, 1, ny + 1 };
        avgCellP(SimT, sim, nx + 1, ny + 1, s2[0], s2[1], s2[2], s2[3]);
        try p2uCell(SimT, sim, nx + 1, ny + 1, 0);
    }
}

// --- 3D corner/edge filling (finite.c:3673-4228, the regular non-SHEARINGBOX
// branch of `if(TNZ>1 && TNY>1)`) ------------------------------------------
//
// Serial ⇒ mpi_isitBC_forcorners()≡1 (mpi.c:2875), so every block runs. The
// order is load-bearing: the 12 elongated edge-tubes are filled first (along
// z, then y, then x), then the 8 total corner cubes OVERWRITE the shared
// corner cells last ("fills crap but overwritten below", finite.c:3867). PUFFY
// is not SHEARINGBOX, so there are no VZ no-inflow clamps here. z-periodicity
// is already baked into the z-face ghosts by setBcZ (run before corners), so
// the edge/corner fills are plain copies + diagonal averages — no wrapping.

/// Build a 3D cell index from three (comptime axis, value) pairs that must be a
/// permutation of {0,1,2}. Lets one edge-tube body serve all three orientations.
fn cell3(comptime ax: usize, av: i64, comptime bx: usize, bv: i64, comptime cx: usize, cv: i64) [3]i64 {
    var r: [3]i64 = undefined;
    r[ax] = av;
    r[bx] = bv;
    r[cx] = cv;
    return r;
}

fn copyCell3(comptime SimT: type, sim: *SimT, dst: [3]i64, src: [3]i64) void {
    const NV = SimT.nv;
    var pp: [NV]f64 = undefined;
    sim.p.load(src[0], src[1], src[2], &pp);
    sim.p.store(dst[0], dst[1], dst[2], &pp);
}

fn avgCell3(comptime SimT: type, sim: *SimT, dst: [3]i64, a: [3]i64, b: [3]i64) void {
    const NV = SimT.nv;
    var pa: [NV]f64 = undefined;
    var pb: [NV]f64 = undefined;
    sim.p.load(a[0], a[1], a[2], &pa);
    sim.p.load(b[0], b[1], b[2], &pb);
    var pp: [NV]f64 = undefined;
    for (0..NV) |iv| pp[iv] = 0.5 * (pa[iv] + pb[iv]);
    sim.p.store(dst[0], dst[1], dst[2], &pp);
}

fn p2uCell3(comptime SimT: type, sim: *SimT, c: [3]i64) Error!void {
    try p2uCell(SimT, sim, c[0], c[1], c[2]);
}

/// One elongated corner edge-tube: the 2D "corner" in the (pa,pb) plane, filled
/// identically at every slice of the through-axis `pc` over its full ghost span
/// [-NG, N+NG). Mirrors the per-corner body of finite.c's along-z/y/x blocks:
/// two 1-deep surface strips (the NG−1 outer ghosts) copied from the domain
/// edge, then the two diagonal ghosts averaged. `a_hi`/`b_hi` pick the low/high
/// side on each corner axis.
fn fillEdgeTube(
    comptime SimT: type,
    sim: *SimT,
    comptime pa: usize,
    comptime a_hi: bool,
    comptime pb: usize,
    comptime b_hi: bool,
    comptime pc: usize,
) Error!void {
    const ng: i64 = @intCast(sim.grid.ng);
    const dimsize = [3]i64{ sim.nxi(), sim.nyi(), sim.nzi() };
    const na = dimsize[pa];
    const nb = dimsize[pb];
    const nc = dimsize[pc];

    // domain-edge cell and the two inner ghosts on each corner axis
    const a_edge: i64 = if (a_hi) na - 1 else 0;
    const a_g1: i64 = if (a_hi) na else -1;
    const a_g2: i64 = if (a_hi) na + 1 else -2;
    const b_edge: i64 = if (b_hi) nb - 1 else 0;
    const b_g1: i64 = if (b_hi) nb else -1;
    const b_g2: i64 = if (b_hi) nb + 1 else -2;

    var c: i64 = -ng;
    while (c < nc + ng) : (c += 1) {
        // NG−1 outer-ghost surface strips: b-strip along a, a-strip along b
        var i: i64 = 0;
        while (i < ng - 1) : (i += 1) {
            const a_out: i64 = if (a_hi) na + i + 1 else -ng + i;
            const b_out: i64 = if (b_hi) nb + i + 1 else -ng + i;
            const db = cell3(pa, a_out, pb, b_g1, pc, c); // (a_out, b_g1) ← (a_out, b_edge)
            copyCell3(SimT, sim, db, cell3(pa, a_out, pb, b_edge, pc, c));
            try p2uCell3(SimT, sim, db);
            const da = cell3(pa, a_g1, pb, b_out, pc, c); // (a_g1, b_out) ← (a_edge, b_out)
            copyCell3(SimT, sim, da, cell3(pa, a_edge, pb, b_out, pc, c));
            try p2uCell3(SimT, sim, da);
        }
        // inner diagonal (a_g1,b_g1) = ½[(a_g1,b_edge) + (a_edge,b_g1)]
        const d1 = cell3(pa, a_g1, pb, b_g1, pc, c);
        avgCell3(SimT, sim, d1, cell3(pa, a_g1, pb, b_edge, pc, c), cell3(pa, a_edge, pb, b_g1, pc, c));
        try p2uCell3(SimT, sim, d1);
        // outer diagonal (a_g2,b_g2) = ½[(a_g2,b_g1) + (a_g1,b_g2)]
        const d2 = cell3(pa, a_g2, pb, b_g2, pc, c);
        avgCell3(SimT, sim, d2, cell3(pa, a_g2, pb, b_g1, pc, c), cell3(pa, a_g1, pb, b_g2, pc, c));
        try p2uCell3(SimT, sim, d2);
    }
}

/// One of the 8 total corner cubes (finite.c:4133-4228): an NG×NG×NG block of
/// ghosts all copied from the single nearest active domain corner cell.
fn fillCornerCube(comptime SimT: type, sim: *SimT, x_hi: bool, y_hi: bool, z_hi: bool) Error!void {
    const ng: i64 = @intCast(sim.grid.ng);
    const nx = sim.nxi();
    const ny = sim.nyi();
    const nz = sim.nzi();
    const src = [3]i64{
        if (x_hi) nx - 1 else 0,
        if (y_hi) ny - 1 else 0,
        if (z_hi) nz - 1 else 0,
    };
    const x0: i64 = if (x_hi) nx else -ng;
    const x1: i64 = if (x_hi) nx + ng else 0;
    const y0: i64 = if (y_hi) ny else -ng;
    const y1: i64 = if (y_hi) ny + ng else 0;
    const z0: i64 = if (z_hi) nz else -ng;
    const z1: i64 = if (z_hi) nz + ng else 0;

    var ix = x0;
    while (ix < x1) : (ix += 1) {
        var iy = y0;
        while (iy < y1) : (iy += 1) {
            var iz = z0;
            while (iz < z1) : (iz += 1) {
                const dst = [3]i64{ ix, iy, iz };
                copyCell3(SimT, sim, dst, src);
                try p2uCell3(SimT, sim, dst);
            }
        }
    }
}

/// finite.c:3673-4228 — full-3D (TNY>1 && TNZ>1) total-corner filling.
/// `z_physical` mirrors C's mpi_isitBC_forcorners(ZBC*) gating: with z
/// interior (ntz>1, periodic ring) only the x-y tubes run — still over the
/// FULL z ghost span, on the exchange-delivered z-ghost planes — while the
/// x-z/y-z tubes and all 8 cubes are skipped (their z face is exchanged,
/// finite.c:3956+/4046+/4134+ gates are false).
fn fillCorners3d(comptime SimT: type, sim: *SimT, z_physical: bool) Error!void {
    // elongated corners along z (x-y corners, all iz)
    try fillEdgeTube(SimT, sim, 0, false, 1, false, 2);
    try fillEdgeTube(SimT, sim, 0, false, 1, true, 2);
    try fillEdgeTube(SimT, sim, 0, true, 1, false, 2);
    try fillEdgeTube(SimT, sim, 0, true, 1, true, 2);
    if (!z_physical) return;
    // elongated corners along y (x-z corners, all iy)
    try fillEdgeTube(SimT, sim, 0, false, 2, false, 1);
    try fillEdgeTube(SimT, sim, 0, false, 2, true, 1);
    try fillEdgeTube(SimT, sim, 0, true, 2, false, 1);
    try fillEdgeTube(SimT, sim, 0, true, 2, true, 1);
    // elongated corners along x (y-z corners, all ix)
    try fillEdgeTube(SimT, sim, 1, false, 2, false, 0);
    try fillEdgeTube(SimT, sim, 1, false, 2, true, 0);
    try fillEdgeTube(SimT, sim, 1, true, 2, false, 0);
    try fillEdgeTube(SimT, sim, 1, true, 2, true, 0);
    // total corner cubes (overwrite the shared corner cells last), C order
    try fillCornerCube(SimT, sim, false, false, false);
    try fillCornerCube(SimT, sim, true, false, false);
    try fillCornerCube(SimT, sim, false, true, false);
    try fillCornerCube(SimT, sim, true, true, false);
    try fillCornerCube(SimT, sim, false, false, true);
    try fillCornerCube(SimT, sim, true, false, true);
    try fillCornerCube(SimT, sim, false, true, true);
    try fillCornerCube(SimT, sim, true, true, true);
}

/// C's 3D "corners in the middle" pass for a physical x/y face meeting the
/// exchanged z faces (finite.c:4463-4556 / 4569-4662): the x-BC is applied
/// at every (x-ghost, domain-y, z-ghost) cell and the y-BC at every
/// (domain-x, y-ghost, z-ghost) cell — overwriting the neighbor's stale
/// transverse ghost columns delivered by the exchange with a fresh local
/// fill. C's block order (all four XBC±/ZBC± first, then YBC±/ZBC±) is
/// kept; setBcCell routes to the same specific/copy/periodic logic as
/// set_bc_core with the face's BCtype.
/// Band-parallel like every other face pass: the x-blocks band over iy and
/// the y-blocks over ix, so each band writes only ghost cells carrying its
/// own index and reads only domain cells (which this pass never writes) —
/// bit-identical to serial at any thread count.
///
/// Threading this is not a micro-optimization. Its cost is
/// O((nx+ny)·ng²) per call × ~13 calls/step and, unlike every other pass,
/// does NOT shrink as φ-slabs get thinner — so left serial it becomes the
/// dominant Amdahl term of an MPI run (measured: 74 ms of a 317 ms step at
/// 3 ranks, i.e. 23%, on a 64×60×24 wedge).
fn fillMiddleCornersZ(comptime SimT: type, sim: *SimT, t: f64, ifinit: bool) Error!void {
    const Ctx = BcCtx(SimT);
    var ctx = Ctx{ .sim = sim, .t = t, .ifinit = ifinit };
    // XBCLO/XBCHI × ZBCLO/ZBCHI (finite.c:4463+): domain iy, x-ghost, z-ghost
    try threading.parallelRangeErr(Ctx, &ctx, sim.team, 0, sim.nyi(), midXFn(SimT));
    // YBCLO/YBCHI × ZBCLO/ZBCHI (finite.c:4569+): domain ix, y-ghost, z-ghost
    try threading.parallelRangeErr(Ctx, &ctx, sim.team, 0, sim.nxi(), midYFn(SimT));
}

fn midXFn(comptime SimT: type) fn (*BcCtx(SimT), i64, i64) Error!void {
    return struct {
        fn w(c: *BcCtx(SimT), iy0: i64, iy1: i64) Error!void {
            return midXBand(SimT, c, iy0, iy1);
        }
    }.w;
}

fn midXBand(comptime SimT: type, c: *BcCtx(SimT), iy0: i64, iy1: i64) Error!void {
    const sim = c.sim;
    const nx = sim.nxi();
    const nz = sim.nzi();
    const ng: i64 = @intCast(sim.grid.ng);
    var iy: i64 = iy0;
    while (iy < iy1) : (iy += 1) {
        var i: i64 = 1;
        while (i <= ng) : (i += 1) {
            var j: i64 = 1;
            while (j <= ng) : (j += 1) {
                try setBcCell(SimT, sim, -i, iy, -j, c.t, c.ifinit, .xlo);
                try setBcCell(SimT, sim, -i, iy, nz - 1 + j, c.t, c.ifinit, .xlo);
                try setBcCell(SimT, sim, nx - 1 + i, iy, -j, c.t, c.ifinit, .xhi);
                try setBcCell(SimT, sim, nx - 1 + i, iy, nz - 1 + j, c.t, c.ifinit, .xhi);
            }
        }
    }
}

fn midYFn(comptime SimT: type) fn (*BcCtx(SimT), i64, i64) Error!void {
    return struct {
        fn w(c: *BcCtx(SimT), ix0: i64, ix1: i64) Error!void {
            return midYBand(SimT, c, ix0, ix1);
        }
    }.w;
}

fn midYBand(comptime SimT: type, c: *BcCtx(SimT), ix0: i64, ix1: i64) Error!void {
    const sim = c.sim;
    const ny = sim.nyi();
    const nz = sim.nzi();
    const ng: i64 = @intCast(sim.grid.ng);
    var ix: i64 = ix0;
    while (ix < ix1) : (ix += 1) {
        var i: i64 = 1;
        while (i <= ng) : (i += 1) {
            var j: i64 = 1;
            while (j <= ng) : (j += 1) {
                try setBcCell(SimT, sim, ix, -i, -j, c.t, c.ifinit, .ylo);
                try setBcCell(SimT, sim, ix, -i, nz - 1 + j, c.t, c.ifinit, .ylo);
                try setBcCell(SimT, sim, ix, ny - 1 + i, -j, c.t, c.ifinit, .yhi);
                try setBcCell(SimT, sim, ix, ny - 1 + i, nz - 1 + j, c.t, c.ifinit, .yhi);
            }
        }
    }
}
