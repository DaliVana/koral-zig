//! The explicit operator's transport passes (C: op_explicit, finite.c:633),
//! generic over CoreT and driven through `Ctx` (the Core, the operator's
//! own face stores, the optional viscosity state, and the sub-step dt):
//!
//!   sweep(dim):      finite.c:708-1181 — reconstruct cell → face states,
//!                    floor them, one-sided fluxes into pb/fl face arrays.
//!   fluxesAtFaces:   finite.c:1461 f_calc_fluxes_at_faces (LAXF / HLL).
//!   update:          the conserved update from the flux divergence plus
//!                    the metric source (physics.c:789
//!                    f_metric_source_term_arb, GDETIN == 1 branch).
//!
//! `Sim.opExplicit` composes them in C's order (wavespeeds → sweeps →
//! fluxes → flux-CT → update → calcU2p). The x-innermost band loop orders,
//! the `inline` face kernels and every expression shape are unchanged from
//! the original sim.zig bodies, so the result stays bit-identical.

const std = @import("std");
const relele = @import("../relele.zig");
const hydro = @import("../physics/hydro.zig");
const flux_mod = @import("../physics/flux.zig");
const radiation = @import("../physics/radiation.zig");
const recon = @import("../fv/recon.zig");
const laxf_mod = @import("../fv/laxf.zig");
const invert = @import("../solve/invert.zig");
const p2u_mod = @import("../p2u.zig");
const rijvisc_mod = @import("rijvisc.zig");
const threading = @import("../threading.zig");
const storage = @import("storage.zig");
const Grid = @import("../grid.zig").Grid;

const Error = relele.Error || error{OutOfMemory};

/// The face stores the explicit operator owns (pass-owned scratch):
/// face-interpolated primitives and one-sided fluxes per side
/// (C: pbLx.., flLx..) and the combined fluxes (flbx/flby/flbz). Lives at
/// `sim.faces`; flux-CT (sim/ct.zig) rewrites the B rows of `flb` in place.
pub fn Faces(comptime NV: usize) type {
    return struct {
        const Self = @This();
        pub const FaceT = storage.FaceStore(NV);

        pb_l: [3]FaceT,
        pb_r: [3]FaceT,
        fl_l: [3]FaceT,
        fl_r: [3]FaceT,
        flb: [3]FaceT,

        pub fn init(allocator: std.mem.Allocator, g: Grid, team: ?*threading.Team) !Self {
            var self: Self = undefined;
            var n: usize = 0;
            errdefer self.deinitFirst(n);
            for (0..3) |d| {
                inline for (.{ "pb_l", "pb_r", "fl_l", "fl_r", "flb" }) |name| {
                    @field(self, name)[d] = try FaceT.initUninitialized(allocator, g, d);
                    n += 1;
                    // Zeroed THROUGH THE TEAM (P4b NUMA first-touch); zeros are zeros.
                    threading.parallelZero(team, @field(self, name)[d].data);
                }
            }
            return self;
        }

        /// Release the first `n` stores in init's allocation order.
        fn deinitFirst(self: *Self, n: usize) void {
            var k: usize = 0;
            for (0..3) |d| {
                inline for (.{ "pb_l", "pb_r", "fl_l", "fl_r", "flb" }) |name| {
                    if (k < n) @field(self, name)[d].deinit();
                    k += 1;
                }
            }
        }

        pub fn deinit(self: *Self) void {
            self.deinitFirst(15);
            self.* = undefined;
        }
    };
}

/// What one explicit operator sees: the Core, its own face stores, the
/// viscosity state when radiative viscosity is on, and the sub-step dt.
pub fn Ctx(comptime CoreT: type) type {
    return struct {
        core: *CoreT,
        faces: *Faces(CoreT.nv),
        visc: ?*const rijvisc_mod.State,
        dt: f64 = 0,
    };
}

// ---- source terms -----------------------------------------------------

/// C: f_metric_source_term_arb (physics.c:789), GDETIN == 1: only the
/// Christoffel contraction rows survive; f_general_source_term is
/// zero for every M5/M6 problem (no artificial heating/cooling).
fn metricSource(comptime CoreT: type, self: *CoreT, ix: i64, iy: i64, iz: i64) Error![CoreT.nv]f64 {
    const cfg = CoreT.Cfg;
    const L = CoreT.Layout;
    const NV = CoreT.nv;
    const geom = self.cache.fillGeometry(ix, iy, iz);
    var pp: [NV]f64 = undefined;
    self.p.load(ix, iy, iz, &pp);

    const gdetu = geom.gdet;
    var ss: [NV]f64 = @splat(0);

    const tij = try hydro.calcTij(cfg, pp, &geom, self.phys.gam);
    const t = relele.lowerSecond(tij, &geom);

    const kr_blk = self.cache.krBlock(ix, iy, iz);
    const rows = [4]usize{ L.index(.uu), L.index(.vx), L.index(.vy), L.index(.vz) };
    if (comptime L.hasVar(.ee)) {
        const rij_up = try radiation.calcRij(cfg, pp, &geom);
        const rij = relele.lowerSecond(rij_up, &geom);
        const rrows = [4]usize{ L.index(.ee), L.index(.fx), L.index(.fy), L.index(.fz) };
        for (0..4) |k| {
            for (0..4) |l| {
                for (0..4) |nu| {
                    ss[rows[nu]] += gdetu * t[k][l] * kr_blk[l * 16 + nu * 4 + k];
                    ss[rrows[nu]] += gdetu * rij[k][l] * kr_blk[l * 16 + nu * 4 + k];
                }
            }
        }
    } else {
        for (0..4) |k| {
            for (0..4) |l| {
                for (0..4) |nu| {
                    ss[rows[nu]] += gdetu * t[k][l] * kr_blk[l * 16 + nu * 4 + k];
                }
            }
        }
    }
    return ss;
}

// ---- sweep bounds -----------------------------------------------------

/// Sweep bounds for the cross directions of a sweep/face loop:
/// narrow = domain; wide (MHD) = ±1 where that dimension is active.
fn crossLo(comptime CoreT: type, self: *const CoreT, dim: usize) i64 {
    if (comptime CoreT.wide) {
        return if (self.nDim(dim) > 1) -1 else 0;
    }
    return 0;
}
fn crossHi(comptime CoreT: type, self: *const CoreT, dim: usize) i64 {
    if (comptime CoreT.wide) {
        return if (self.nDim(dim) > 1) self.nDim(dim) + 1 else self.nDim(dim);
    }
    return self.nDim(dim);
}

/// The two cross directions of a sweep/face pass in dimension `dim`.
/// Band-parallelism runs over cross[0]; the largest transverse axis
/// in 2D (y for the x-sweep, x for the y-sweep).
fn crossDims(comptime dim: usize) [2]usize {
    return switch (dim) {
        0 => .{ 1, 2 },
        1 => .{ 0, 2 },
        2 => .{ 0, 1 },
        else => unreachable,
    };
}

// ---- the interpolation sweep --------------------------------------------

/// One direction of op_explicit's interpolation sweep
/// (finite.c:708-1181): reconstruct cell → face states, floor them,
/// compute one-sided fluxes, stash into pb/fl face arrays. Skips a
/// collapsed dimension. The caller owns the timer.
pub fn sweep(comptime CoreT: type, c: *Ctx(CoreT), comptime dim: usize) Error!void {
    const self = c.core;
    const n = self.nDim(dim);
    if (n <= 1) return;
    const cross = comptime crossDims(dim);
    try threading.parallelRangeErr(Ctx(CoreT), c, self.team, crossLo(CoreT, self, cross[0]), crossHi(CoreT, self, cross[0]), sweepBandFn(CoreT, dim));
}

fn sweepBandFn(comptime CoreT: type, comptime dim: usize) fn (*Ctx(CoreT), i64, i64) Error!void {
    return struct {
        fn w(c: *Ctx(CoreT), c0_lo: i64, c0_hi: i64) Error!void {
            return sweepBand(CoreT, c, dim, c0_lo, c0_hi);
        }
    }.w;
}

/// One direction of the interpolation sweep for cross[0] ∈
/// [c0_lo, c0_hi): every (c0, c1) line's faces are written only by its
/// own band, so the banding is bit-identical to serial. Loop order keeps
/// the contiguous x index innermost; for dim==0 that is the sweep
/// direction itself; for dim!=0 x is cross[0] (the band range this
/// worker owns), so we iterate it innermost with the sweep direction in
/// the middle, turning the five-point stencil into contiguous x streams
/// instead of iy/iz strides (out of L2 on the 2D grid, a DRAM/TLB cliff
/// in 3D). Each (i,c0,c1) writes only its own face slots, so reordering
/// the iterations is bit-identical (P2 #5).
fn sweepBand(comptime CoreT: type, c: *Ctx(CoreT), comptime dim: usize, c0_lo: i64, c0_hi: i64) Error!void {
    const self = c.core;
    const n = self.nDim(dim);
    const cross = comptime crossDims(dim);
    const c1_lo = crossLo(CoreT, self, cross[1]);
    const c1_hi = crossHi(CoreT, self, cross[1]);

    if (comptime dim == 0) {
        var c0 = c0_lo;
        while (c0 < c0_hi) : (c0 += 1) {
            var c1 = c1_lo;
            while (c1 < c1_hi) : (c1 += 1) {
                var i: i64 = -1;
                while (i < n + 1) : (i += 1) {
                    try sweepFace(CoreT, c, dim, i, c0, c1, dx5At(CoreT, self, dim, i));
                }
            }
        }
    } else {
        var c1 = c1_lo;
        while (c1 < c1_hi) : (c1 += 1) {
            var i: i64 = -1;
            while (i < n + 1) : (i += 1) {
                const dx5 = dx5At(CoreT, self, dim, i); // invariant over the inner (x) loop — hoisted (#6)
                var c0 = c0_lo;
                while (c0 < c0_hi) : (c0 += 1) {
                    try sweepFace(CoreT, c, dim, i, c0, c1, dx5);
                }
            }
        }
    }
}

/// The ±2 sweep-direction cell sizes at slot i (C: get_size_x). Function
/// of (i,dim) only, so it lifts out of the cross loops.
inline fn dx5At(comptime CoreT: type, self: *const CoreT, comptime dim: usize, i: i64) [5]f64 {
    return .{
        self.grid.cellSize(i - 2, dim),
        self.grid.cellSize(i - 1, dim),
        self.grid.cellSize(i, dim),
        self.grid.cellSize(i + 1, dim),
        self.grid.cellSize(i + 2, dim),
    };
}

/// Reconstruct + one-sided fluxes for the face pair bracketing the cell
/// at (i along dim, c0=cross[0], c1=cross[1]); dx5 is the sweep-direction
/// stencil spacing. `inline` so the body stays fused into sweepBand's
/// loops (Debug too). All writes land in this cell's own face slots.
inline fn sweepFace(comptime CoreT: type, c: *Ctx(CoreT), comptime dim: usize, i: i64, c0: i64, c1: i64, dx5: [5]f64) Error!void {
    const self = c.core;
    const cfg = CoreT.Cfg;
    const L = CoreT.Layout;
    const NV = CoreT.nv;
    const base_order = CoreT.base_order;
    const n = self.nDim(dim);
    const cross = comptime crossDims(dim);
    var cell = [3]i64{ 0, 0, 0 };
    cell[dim] = i;
    cell[cross[0]] = c0;
    cell[cross[1]] = c1;

    const dol = i >= 0;
    const dor = i < n;

    var pm2: [NV]f64 = @splat(0);
    var pm1: [NV]f64 = undefined;
    var p0: [NV]f64 = undefined;
    var pp1: [NV]f64 = undefined;
    var pp2: [NV]f64 = @splat(0);
    var cc = cell;
    cc[dim] = i - 1;
    self.p.load(cc[0], cc[1], cc[2], &pm1);
    cc[dim] = i;
    self.p.load(cc[0], cc[1], cc[2], &p0);
    cc[dim] = i + 1;
    self.p.load(cc[0], cc[1], cc[2], &pp1);
    // C guards the ±2 stencil with INT_ORDER>1 — with
    // linear reconstruction there are only 2 ghost cells.
    if (comptime base_order > 1) {
        cc[dim] = i - 2;
        self.p.load(cc[0], cc[1], cc[2], &pm2);
        cc[dim] = i + 2;
        self.p.load(cc[0], cc[1], cc[2], &pp2);
    }

    // C: REDUCEORDERWHENNEEDED + REDUCEORDERATBH — drop the order by one
    // for cells whose center is inside the BL horizon (reduce_order_check,
    // finite.c:14). cell[0] is the radial index. REDUCEMINMODTHETA is
    // still off.
    var reduce: u8 = if (cell[0] <= self.reduce_order_ix_max) 1 else 0;
    // koral_lite_puffy REDUCEORDERAFTERFIXUP: the flags hold whatever
    // the most recent u2p/implicit pass wrote — within a step that is
    // the calcU2p/opImplicit that ran right before this opExplicit
    // (C's RK2IMEX stage order is identical). Ghost cells read 0
    // (flags are ghost-allocated, memset, never set there).
    if (self.num.reduceorderafterfixup and reduce == 0 and
        (self.getFlag(.hd_fixup, cell[0], cell[1], cell[2]) != 0 or
            self.getFlag(.rad_fixup, cell[0], cell[1], cell[2]) != 0 or
            self.getFlag(.radimp_fixup, cell[0], cell[1], cell[2]) != 0))
    {
        reduce = 1;
    }
    const scheme: recon.Scheme = switch (base_order -| reduce) {
        0 => .donor,
        1 => .{ .linear = .{ .theta = self.num.minmod_theta } },
        else => .ppm,
    };
    const f = recon.reconstructN(NV, scheme, .{ pm2, pm1, p0, pp1, pp2 }, dx5);
    var pl = f.left;
    var pr = f.right;

    var ffl: [NV]f64 = undefined;
    var ffr: [NV]f64 = undefined;
    if (dol) {
        const geom = self.cache.fillGeometryFace(cell[0], cell[1], cell[2], dim);
        _ = try invert.checkFloorsMhd(cfg, &pl, &geom, self.phys.gam, self.phys.floors);
        ffl = try flux_mod.fFluxPrime(cfg, pl, dim, &geom, self.phys.gam);
        // radiative viscosity: face i is between cells i−1 and i
        if (c.visc != null and comptime L.hasVar(.ee)) {
            const rv = rijvisc_mod.faceAvg(c.visc.?, dim, cell[0], cell[1], cell[2]);
            try rijvisc_mod.addRadViscFlux(CoreT, self, c.visc.?, &ffl, &pl, &geom, dim, &rv);
        }
    }
    if (dor) {
        var cp = cell;
        cp[dim] = i + 1;
        const geom = self.cache.fillGeometryFace(cp[0], cp[1], cp[2], dim);
        _ = try invert.checkFloorsMhd(cfg, &pr, &geom, self.phys.gam, self.phys.floors);
        ffr = try flux_mod.fFluxPrime(cfg, pr, dim, &geom, self.phys.gam);
        // radiative viscosity: face i+1 is between cells i and i+1
        if (c.visc != null and comptime L.hasVar(.ee)) {
            const rv = rijvisc_mod.faceAvg(c.visc.?, dim, cp[0], cp[1], cp[2]);
            try rijvisc_mod.addRadViscFlux(CoreT, self, c.visc.?, &ffr, &pr, &geom, dim, &rv);
        }
    }

    // pbR[dim] at face i ← pl; pbL[dim] at face i+1 ← pr
    var cp = cell;
    c.faces.pb_r[dim].store(cp[0], cp[1], cp[2], &pl);
    if (dol) c.faces.fl_r[dim].store(cp[0], cp[1], cp[2], &ffl);
    cp[dim] = i + 1;
    c.faces.pb_l[dim].store(cp[0], cp[1], cp[2], &pr);
    if (dor) c.faces.fl_l[dim].store(cp[0], cp[1], cp[2], &ffr);
}

// ---- flux combination ---------------------------------------------------

/// C: f_calc_fluxes_at_faces (finite.c:1461). For every face,
/// combine the one-sided fluxes with LAXF or HLL using the saved
/// cell wavespeeds. Hydro and radiation rows use separate speeds.
/// Band-parallel over cross[0] per dimension, like the sweep. The caller
/// owns the timer.
pub fn fluxesAtFaces(comptime CoreT: type, c: *Ctx(CoreT)) Error!void {
    const self = c.core;
    for (0..3) |d| threading.parallelZero(self.team, c.faces.flb[d].data);

    inline for (0..3) |dim| {
        if (self.nDim(dim) > 1) {
            const cross = comptime crossDims(dim);
            try threading.parallelRangeErr(Ctx(CoreT), c, self.team, crossLo(CoreT, self, cross[0]), crossHi(CoreT, self, cross[0]), fluxesBandFn(CoreT, dim));
        }
    }
}

fn fluxesBandFn(comptime CoreT: type, comptime dim: usize) fn (*Ctx(CoreT), i64, i64) Error!void {
    return struct {
        fn w(c: *Ctx(CoreT), c0_lo: i64, c0_hi: i64) Error!void {
            return fluxesBand(CoreT, c, dim, c0_lo, c0_hi);
        }
    }.w;
}

/// Band-parallel flux combination over cross[0] ∈ [c0_lo, c0_hi). Same
/// x-innermost loop order as sweepBand (P2 #5): dim==0 has the face index
/// i=x innermost already; for dim!=0 the band range x=cross[0] iterates
/// innermost with the face index in the middle, so the per-face reads and
/// the flb store stream x. Each face writes only its own flb[dim] slot →
/// order-independent, bit-identical.
fn fluxesBand(comptime CoreT: type, c: *Ctx(CoreT), comptime dim: usize, c0_lo: i64, c0_hi: i64) Error!void {
    const self = c.core;
    const n = self.nDim(dim);
    const cross = comptime crossDims(dim);
    const c1_lo = crossLo(CoreT, self, cross[1]);
    const c1_hi = crossHi(CoreT, self, cross[1]);

    if (comptime dim == 0) {
        var c0 = c0_lo;
        while (c0 < c0_hi) : (c0 += 1) {
            var c1 = c1_lo;
            while (c1 < c1_hi) : (c1 += 1) {
                var i: i64 = 0;
                while (i <= n) : (i += 1) try fluxFace(CoreT, c, dim, i, c0, c1);
            }
        }
    } else {
        var c1 = c1_lo;
        while (c1 < c1_hi) : (c1 += 1) {
            var i: i64 = 0;
            while (i <= n) : (i += 1) {
                var c0 = c0_lo;
                while (c0 < c0_hi) : (c0 += 1) try fluxFace(CoreT, c, dim, i, c0, c1);
            }
        }
    }
}

/// Combine the one-sided fluxes at the single face (i along dim,
/// c0=cross[0], c1=cross[1]) into flb[dim] via LAXF/HLL. `inline` to stay
/// fused into fluxesBand's loops; writes only this face's own flb slot.
inline fn fluxFace(comptime CoreT: type, c: *Ctx(CoreT), comptime dim: usize, i: i64, c0: i64, c1: i64) Error!void {
    const self = c.core;
    const cfg = CoreT.Cfg;
    const L = CoreT.Layout;
    const NV = CoreT.nv;
    const cross = comptime crossDims(dim);
    var cf = [3]i64{ 0, 0, 0 };
    cf[dim] = i;
    cf[cross[0]] = c0;
    cf[cross[1]] = c1;
    var cm = cf;
    cm[dim] = i - 1;

    const ap1l = self.scGet(storage.ahd_l[dim], cf[0], cf[1], cf[2]);
    const ap1r = self.scGet(storage.ahd_r[dim], cf[0], cf[1], cf[2]);
    const ap1 = self.scGet(storage.ahd_m[dim], cf[0], cf[1], cf[2]);
    const am1l = self.scGet(storage.ahd_l[dim], cm[0], cm[1], cm[2]);
    const am1r = self.scGet(storage.ahd_r[dim], cm[0], cm[1], cm[2]);
    const am1 = self.scGet(storage.ahd_m[dim], cm[0], cm[1], cm[2]);

    var pLl: [NV]f64 = undefined;
    var pRl: [NV]f64 = undefined;
    c.faces.pb_l[dim].load(cf[0], cf[1], cf[2], &pLl);
    c.faces.pb_r[dim].load(cf[0], cf[1], cf[2], &pRl);

    const geom = self.cache.fillGeometryFace(cf[0], cf[1], cf[2], dim);
    const uLl = try p2u_mod.p2u(cfg, pLl, &geom, self.phys.gam);
    const uRl = try p2u_mod.p2u(cfg, pRl, &geom, self.phys.gam);

    // hydro and radiation are two separate systems,
    // each combined with its own speeds (finite.c:1549)
    const ag = @max(ap1, am1);
    const al = @min(ap1l, am1l);
    const ar = @max(ap1r, am1r);

    var agr: f64 = 0;
    var alr: f64 = 0;
    var arr: f64 = 0;
    if (comptime L.hasVar(.ee)) {
        const rp1l = self.scGet(storage.arad_l[dim], cf[0], cf[1], cf[2]);
        const rp1r = self.scGet(storage.arad_r[dim], cf[0], cf[1], cf[2]);
        const rp1 = self.scGet(storage.arad_m[dim], cf[0], cf[1], cf[2]);
        const rm1l = self.scGet(storage.arad_l[dim], cm[0], cm[1], cm[2]);
        const rm1r = self.scGet(storage.arad_r[dim], cm[0], cm[1], cm[2]);
        const rm1 = self.scGet(storage.arad_m[dim], cm[0], cm[1], cm[2]);
        agr = @max(rp1, rm1);
        alr = @min(rp1l, rm1l);
        arr = @max(rp1r, rm1r);
    }

    // load the two one-sided flux vectors once, combine into a stack
    // buffer, store the face once — vs per-iv get/set offset recompute
    // (P2 #4). uLl/uRl are already whole-cell arrays.
    var fll: [NV]f64 = undefined;
    var flr: [NV]f64 = undefined;
    c.faces.fl_l[dim].load(cf[0], cf[1], cf[2], &fll);
    c.faces.fl_r[dim].load(cf[0], cf[1], cf[2], &flr);
    var fstar: [NV]f64 = undefined;
    for (0..NV) |iv| {
        // C: i < NVMHD → hydro block, else radiation
        const is_rad = if (comptime L.hasVar(.ee)) iv >= L.index(.ee) else false;
        const ag_sel = if (is_rad) agr else ag;
        const al_sel = if (is_rad) alr else al;
        const ar_sel = if (is_rad) arr else ar;
        fstar[iv] = switch (comptime cfg.flux) {
            .laxf => laxf_mod.laxf(fll[iv], flr[iv], uLl[iv], uRl[iv], ag_sel),
            .hll => laxf_mod.hll(fll[iv], flr[iv], uLl[iv], uRl[iv], al_sel, ar_sel),
        };
    }
    c.faces.flb[dim].store(cf[0], cf[1], cf[2], &fstar);
}

// ---- the conserved update -----------------------------------------------

/// The conserved update over the domain: du from the flux divergence plus
/// the metric source, band-parallel over iy, with `c.dt` the sub-step dt.
/// The caller owns the timer.
pub fn update(comptime CoreT: type, c: *Ctx(CoreT)) Error!void {
    const self = c.core;
    try threading.parallelRangeErr(Ctx(CoreT), c, self.team, 0, self.nyi(), updateRowsBand(CoreT));
}

fn updateRowsBand(comptime CoreT: type) fn (*Ctx(CoreT), i64, i64) Error!void {
    return struct {
        fn w(c: *Ctx(CoreT), iy0: i64, iy1: i64) Error!void {
            return updateRows(CoreT, c, iy0, iy1);
        }
    }.w;
}

/// The conserved update (flux divergence + metric source) for
/// iy ∈ [iy0, iy1); writes only its own cells' u.
fn updateRows(comptime CoreT: type, c: *Ctx(CoreT), iy0: i64, iy1: i64) Error!void {
    const self = c.core;
    const NV = CoreT.nv;
    const dtin = c.dt;
    var iz: i64 = 0;
    while (iz < self.nzi()) : (iz += 1) {
        var iy: i64 = iy0;
        while (iy < iy1) : (iy += 1) {
            var ix: i64 = 0;
            while (ix < self.nxi()) : (ix += 1) {
                const ms = try metricSource(CoreT, self, ix, iy, iz);
                const dx = self.grid.cellSize(ix, 0);
                const dy = self.grid.cellSize(iy, 1);
                const dz = self.grid.cellSize(iz, 2);
                const dt = dtin;

                // Load each accessed face/cell vector once into a stack
                // buffer (single offset per cell) and index the buffers
                // per iv — vs re-deriving the offset NV× per FaceStore.get
                // (P2 #4). Per-component arithmetic unchanged → bit-identical.
                var flxl: [NV]f64 = undefined;
                var flxr: [NV]f64 = undefined;
                var flyl: [NV]f64 = undefined;
                var flyr: [NV]f64 = undefined;
                var flzl: [NV]f64 = undefined;
                var flzr: [NV]f64 = undefined;
                c.faces.flb[0].load(ix, iy, iz, &flxl);
                c.faces.flb[0].load(ix + 1, iy, iz, &flxr);
                c.faces.flb[1].load(ix, iy, iz, &flyl);
                c.faces.flb[1].load(ix, iy + 1, iz, &flyr);
                c.faces.flb[2].load(ix, iy, iz, &flzl);
                c.faces.flb[2].load(ix, iy, iz + 1, &flzr);
                var uu: [NV]f64 = undefined;
                self.u.load(ix, iy, iz, &uu);

                for (0..NV) |iv| {
                    const du = -(flxr[iv] - flxl[iv]) * dt / dx - (flyr[iv] - flyl[iv]) * dt / dy - (flzr[iv] - flzl[iv]) * dt / dz;
                    uu[iv] = uu[iv] + du + ms[iv] * dt;
                }
                self.u.store(ix, iy, iz, &uu);
            }
        }
    }
}
