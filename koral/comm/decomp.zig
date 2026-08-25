//! φ-only domain decomposition (MPI plan 2026-08-07 §5): the global grid is
//! split into NTZ contiguous z-slabs, one per rank, forming a 1D periodic
//! ring. θ and r are NEVER decomposed (every radial/polar subtlety stays
//! rank-local) and 2D (nz==1) never decomposes at all. `decompose()` is pure
//! arithmetic: unit-testable without MPI (validation ladder gate 1).

const std = @import("std");
const Grid = @import("../grid.zig").Grid;
const BcFace = @import("../sim/bc.zig").BcFace;

pub const Error = error{
    /// 2D runs are single-node by decision; one modern node beats C's
    /// 960-rank 2D setup; requesting ntz>1 with nz==1 is a config error.
    TwoDNeverDecomposes,
    /// TNZ % NTZ != 0; only uniform slabs are supported (weighted cuts are
    /// a documented later option).
    PhiNotDivisible,
    /// nz_local < NG: a rank's ghost region must be fillable from its
    /// adjacent neighbor's domain alone (single-hop nearest-neighbor
    /// exchange: C has the identical per-tile ≥ NG constraint).
    SlabThinnerThanGhosts,
    /// tk must be in [0, ntz).
    BadRingRank,
};

/// The decomposition record a rank carries: its slab, its ring position, and
/// which faces are physical boundaries (C: TOI/TOJ/TOK + mpi_isitBC).
/// TOI and TOJ are identically 0; r and θ whole.
pub const Decomp = struct {
    /// Global dims/extents (what the params file describes).
    global: Grid,
    /// This rank's slab: nz = TNZ/NTZ, z coordinates offset via izoff. See
    /// Grid.initLocal (dz copied, never recomputed).
    local: Grid,
    /// φ-slab origin: global z-index of local z-cell 0 (C: TOK).
    tok: i64,
    /// Ring position (cart-communicator rank; C: TK).
    tk: usize,
    /// Ring size (C: NTZ). 1 ⇒ no decomposition, all faces physical.
    ntz: usize,

    /// C: mpi_isitBC. Is this face a physical boundary of this rank's slab?
    /// x/y faces: always (r/θ never split). z faces: only when the ring is
    /// trivial: with ntz>1 the periodic-φ wrap becomes the interior
    /// exchange and setBc's z fill reduces to p2u of exchanged primitives.
    pub fn isPhysical(d: *const Decomp, face: BcFace) bool {
        return switch (face) {
            .xlo, .xhi, .ylo, .yhi => true,
            .zlo, .zhi => d.ntz == 1,
        };
    }

    /// The trivial single-rank decomposition (serial semantics).
    pub fn serial(g: Grid) Decomp {
        return .{ .global = g, .local = g, .tok = 0, .tk = 0, .ntz = 1 };
    }
};

/// Split `global` over an ntz-rank periodic φ ring; `tk` is this rank's ring
/// position. Enforces the plan's split rules with clear errors.
pub fn decompose(global: Grid, ntz: usize, tk: usize) Error!Decomp {
    std.debug.assert(ntz >= 1);
    if (tk >= ntz) return Error.BadRingRank;
    if (ntz == 1) return Decomp.serial(global);
    if (global.nz == 1) return Error.TwoDNeverDecomposes;
    if (global.nz % ntz != 0) return Error.PhiNotDivisible;
    const nz_local = global.nz / ntz;
    if (nz_local < global.ng) return Error.SlabThinnerThanGhosts;
    const tok = tk * nz_local;
    return .{
        .global = global,
        .local = Grid.initLocal(global, tok, nz_local),
        .tok = @intCast(tok),
        .tk = tk,
        .ntz = ntz,
    };
}

//
// ---- gate 1 tests (no MPI needed) -----------------------------------------
//

test "decompose: slab origins, sizes, dz bit-equality, isPhysical" {
    const g = Grid.init(.{ .nx = 8, .ny = 6, .nz = 12, .ng = 3, .minx = 0, .maxx = 1, .minz = -std.math.pi / 4.0, .maxz = std.math.pi / 4.0 });
    for (0..4) |tk| {
        const d = try decompose(g, 4, tk);
        try std.testing.expectEqual(@as(usize, 3), d.local.nz);
        try std.testing.expectEqual(@as(i64, @intCast(tk * 3)), d.tok);
        try std.testing.expectEqual(g.dz, d.local.dz); // copied, never recomputed
        try std.testing.expectEqual(g.dx, d.local.dx);
        try std.testing.expect(d.isPhysical(.xlo) and d.isPhysical(.xhi));
        try std.testing.expect(d.isPhysical(.ylo) and d.isPhysical(.yhi));
        try std.testing.expect(!d.isPhysical(.zlo) and !d.isPhysical(.zhi));
        // slab coordinates == global coordinates for the same physical cell
        try std.testing.expectEqual(g.zc(d.tok), d.local.zc(0));
        try std.testing.expectEqual(g.zc(d.tok + 2), d.local.zc(2));
    }
    // slabs tile the ring exactly
    const d0 = try decompose(g, 2, 0);
    const d1 = try decompose(g, 2, 1);
    try std.testing.expectEqual(@as(i64, 0), d0.tok);
    try std.testing.expectEqual(@as(i64, 6), d1.tok);
}

test "decompose: ntz=1 is the trivial serial decomposition" {
    const g = Grid.init(.{ .nx = 8, .ny = 6, .nz = 12, .ng = 3, .minx = 0, .maxx = 1 });
    const d = try decompose(g, 1, 0);
    try std.testing.expectEqual(g, d.local);
    try std.testing.expect(d.isPhysical(.zlo) and d.isPhysical(.zhi));
}

test "decompose: rejects 2D splits, indivisible φ, too-thin slabs, bad rank" {
    const g2d = Grid.init(.{ .nx = 8, .ny = 6, .nz = 1, .ng = 3, .minx = 0, .maxx = 1 });
    try std.testing.expectError(Error.TwoDNeverDecomposes, decompose(g2d, 2, 0));

    const g = Grid.init(.{ .nx = 8, .ny = 6, .nz = 12, .ng = 3, .minx = 0, .maxx = 1 });
    try std.testing.expectError(Error.PhiNotDivisible, decompose(g, 5, 0));
    // 12/6 = 2 < NG=3: the hard slab floor (the "why not 32 nodes" rule)
    try std.testing.expectError(Error.SlabThinnerThanGhosts, decompose(g, 6, 0));
    try std.testing.expectError(Error.BadRingRank, decompose(g, 4, 4));
}
