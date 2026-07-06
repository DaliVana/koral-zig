//! Boundary conditions for the evolution driver, written SimT-generic (the
//! house `fn f(comptime SimT: type, sim: *SimT)` pattern, like magn/ct.zig).
//! This is the region the planned MPI backend grows into (comm/serial.zig:
//! "the MPI backend implements the same API later") and maps directly onto C
//! finite.c:2805 (set_bc) / finite.c:3203 (2D corner filling). `sim.zig`
//! exposes a thin `Sim.setBc` method that delegates here, and re-exports
//! `BcKind`/`BcFace` so problems and tests keep referencing `sim.BcFace`.

const p2u_mod = @import("../p2u.zig");

const Error = @import("../relele.zig").Error || error{OutOfMemory};

/// Boundary handling per axis (C: PERIODIC_?BC / COPY_?BC / SPECIFIC_BC).
pub const BcKind = enum { periodic, copy, specific };

/// Which boundary a ghost cell belongs to (C: XBCLO..ZBCHI).
pub const BcFace = enum { xlo, xhi, ylo, yhi, zlo, zhi };

/// C: set_bc (finite.c:2805) — ghost cells (no corners), then for MHD
/// ("MPI4CORNERS") builds the 2D corner surfaces + diagonals
/// (finite.c:3203-3403; serial, so mpi_isitBC ≡ 1).
pub fn setBc(comptime SimT: type, sim: *SimT, t: f64, ifinit: bool) Error!void {
    const g = &sim.grid;
    const nx = sim.nxi();
    const ny = sim.nyi();
    const nz = sim.nzi();
    const ngx: i64 = @intCast(g.ngx);
    const ngy: i64 = @intCast(g.ngy);
    const ngz: i64 = @intCast(g.ngz);

    // x boundaries (ghost columns, domain rows)
    if (g.ngx > 0) {
        var iz: i64 = 0;
        while (iz < nz) : (iz += 1) {
            var iy: i64 = 0;
            while (iy < ny) : (iy += 1) {
                var i: i64 = 1;
                while (i <= ngx) : (i += 1) {
                    try setBcCell(SimT, sim, -i, iy, iz, t, ifinit, .xlo);
                    try setBcCell(SimT, sim, nx - 1 + i, iy, iz, t, ifinit, .xhi);
                }
            }
        }
    }
    // y boundaries
    if (g.ngy > 0) {
        var iz: i64 = 0;
        while (iz < nz) : (iz += 1) {
            var ix: i64 = 0;
            while (ix < nx) : (ix += 1) {
                var i: i64 = 1;
                while (i <= ngy) : (i += 1) {
                    try setBcCell(SimT, sim, ix, -i, iz, t, ifinit, .ylo);
                    try setBcCell(SimT, sim, ix, ny - 1 + i, iz, t, ifinit, .yhi);
                }
            }
        }
    }
    // z boundaries
    if (g.ngz > 0) {
        var iy: i64 = 0;
        while (iy < ny) : (iy += 1) {
            var ix: i64 = 0;
            while (ix < nx) : (ix += 1) {
                var i: i64 = 1;
                while (i <= ngz) : (i += 1) {
                    try setBcCell(SimT, sim, ix, iy, -i, t, ifinit, .zlo);
                    try setBcCell(SimT, sim, ix, iy, nz - 1 + i, t, ifinit, .zhi);
                }
            }
        }
    }

    if (comptime SimT.Cfg.has(.mhd)) {
        if (g.ny > 1 and g.nz == 1) {
            try fillCorners2d(SimT, sim);
        } else if (g.nz > 1) {
            // 3D / r-φ corner filling arrives with the problems that
            // need it; no M5/M6 target is 3D.
            @panic("Sim.setBc: 3D corner filling not implemented");
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
            pp = sim.opt.specific_bc.?(sim, ix, iy, iz, t, ifinit, face);
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
