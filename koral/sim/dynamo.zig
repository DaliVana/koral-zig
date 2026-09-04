//! Mean-field "mimic" dynamo (PUFFY's MIMICDYNAMO), transcribed from KORAL:
//!
//!   apply_dynamo    finite.c:1370: calc_avgs_throughout (scale height) →
//!                                    set_bc → mimic_dynamo → calc_u2p.
//!   mimic_dynamo    magn.c:1003: per cell (domain + 1 ring incl. corners),
//!                                    build an extra toroidal vector potential
//!                                    ΔA_φ ∝ α_dyn · (dt/P_K) · B^φ, weighted
//!                                    weighted by radius, field angle, and height;
//!                                    apply DAMPBETA azimuthal-field damping;
//!                                    run calc_BfromA(ΔA_φ) → curl; superimpose the
//!                                    new poloidal B on the domain + p2u.
//!   calc_avgs_throughout  mpi.c:3163; the CALCHRONTHEGO density-weighted RMS
//!                                    scale height scaleth_otg[radius].
//!
//! PUFFY switches (define.h:66-74): MIMICDYNAMO, CALCHRONTHEGO, THETAANGLE
//! 0.25, ALPHAFLIPSSIGN, ALPHADYNAMO 2·0.314, DAMPBETA, BETASATURATED 0.1,
//! ALPHABETA 2·6.28, EXPECTEDHR 0.3 (only used without CALCHRONTHEGO). No
//! MAXRADIUS4DYNAMO. The dynamo runs after each explicit sub-step
//! (problem.c:210,326).
//!
//! Fidelity notes:
//!  * calc_ucon_ucov_from_prims at magn.c:1034 and the facmag1/facmag2 step
//!    functions are dead (unused by the ΔA_φ / DAMPBETA expressions); skipped.
//!  * scaleth_otg[0] keeps the *unnormalized* raw sum (the sqrt loop skips
//!    gix==0, mpi.c:3225); a C quirk transcribed verbatim; the innermost
//!    column's faczH collapses to 0 there anyway.
//!  * pow(1-zH², 1) == 1-zH² (libm special case) → written as a plain sub.
//!  * P1: the three per-cell passes (scale-height columns, ΔA_φ ring,
//!    superpose+p2u) run band-parallel via threading.zig; every band
//!    writes only its own cells/columns, so the result is bit-identical to
//!    serial at any nthreads (pinned by the dynamo threading test).

const std = @import("std");
const field_mod = @import("../field.zig");
const Grid = @import("../grid.zig").Grid;
const threading_mod = @import("../threading.zig");

/// The dynamo's scratch, owned by a Sim only when `opt.dynamo` is on
/// (redesign step 3: pass-owned scratch). Lives at `core.dynamo`.
pub const State = struct {
    allocator: std.mem.Allocator,
    /// PUFFY dynamo parameters (define.h:66-74).
    params: Params,
    /// C: ptemp1 — the dynamo's cell-centered ΔA_φ scratch (slot 2 = φ).
    dyn_a: field_mod.Field(3),
    /// C: scaleth_otg — the per-radius density-weighted scale height.
    scaleth: []f64,
    /// MPI plan §7.2: per-radius partial sums (Σρ√g, Σρ√g·Δθ²) for the
    /// scale-height ring reduction — the pre-√ Allreduce(SUM) operands.
    /// Only the ntz>1 path touches them; tiny (2·nx doubles).
    scaleth_sig: []f64,
    scaleth_sth: []f64,

    pub fn init(allocator: std.mem.Allocator, g: Grid, team: ?*threading_mod.Team, params: Params) !State {
        var dyn_a = try field_mod.Field(3).initUninitialized(allocator, g);
        errdefer dyn_a.deinit();
        threading_mod.parallelZero(team, dyn_a.data);
        const scaleth = try allocator.alloc(f64, g.nx);
        errdefer allocator.free(scaleth);
        @memset(scaleth, 0);
        const sig = try allocator.alloc(f64, g.nx);
        errdefer allocator.free(sig);
        @memset(sig, 0);
        const sth = try allocator.alloc(f64, g.nx);
        @memset(sth, 0);
        return .{ .allocator = allocator, .params = params, .dyn_a = dyn_a, .scaleth = scaleth, .scaleth_sig = sig, .scaleth_sth = sth };
    }

    pub fn deinit(self: *State) void {
        self.dyn_a.deinit();
        self.allocator.free(self.scaleth);
        self.allocator.free(self.scaleth_sig);
        self.allocator.free(self.scaleth_sth);
        self.* = undefined;
    }
};
const config = @import("../config.zig");
const layout = @import("../layout.zig");
const relele = @import("../relele.zig");
const mhd = @import("../physics/bfield.zig");
const radiation = @import("../physics/radiation.zig");
const frames = @import("../frames.zig");
const metric = @import("../metric/metric.zig");
const misc = @import("../math/misc.zig");
const p2u_mod = @import("../p2u.zig");
const ct = @import("ct.zig");
const threading = @import("../threading.zig");
const Geometry = @import("../geometry.zig").Geometry;

const pi = std.math.pi;

/// setBc / calcU2p can allocate; the rest is relele-only.
const Error = relele.Error || error{OutOfMemory};

/// PUFFY dynamo parameters (define.h:66-74).
pub const Params = struct {
    alphadynamo: f64 = 2.0 * 0.314, // ALPHADYNAMO
    thetaangle: f64 = 0.25, // THETAANGLE
    alphabeta: f64 = 2.0 * 6.28, // ALPHABETA
    betasaturated: f64 = 0.1, // BETASATURATED
    expectedhr: f64 = 0.3, // EXPECTEDHR (only if !calchronthego)
    alphaflipssign: bool = true, // ALPHAFLIPSSIGN
    dampbeta: bool = true, // DAMPBETA
    calchronthego: bool = true, // CALCHRONTHEGO
};

/// C: the DAMPBETA azimuthal-field damping (magn.c:1144-1151). dB^φ opposes
/// B^φ, grows with (β−β_sat), and is clamped so it never overshoots zero, so
/// |B^φ| is non-increasing and its sign is preserved (saturation toward 0
/// once β exceeds BETASATURATED). Pure, so it can be unit-tested directly.
pub fn dampBphi(alphabeta: f64, facradius: f64, faczh: f64, dt_over_pk: f64, beta: f64, betasat: f64, bphi: f64) f64 {
    var dbphi = -alphabeta * facradius * faczh * dt_over_pk *
        @max(0.0, beta - betasat) / betasat * bphi;
    if ((dbphi + bphi) * bphi < 0.0) dbphi = -bphi; // no overshoot past zero
    return dbphi;
}

/// The per-cell state mimic_dynamo consumes after the geometry / frame gather;
/// everything the ΔA_φ law reads that isn't a fixed parameter.
pub const DynamoCell = struct {
    r: f64, // BL radius
    th: f64, // BL colatitude θ
    bphi: f64, // p[B3] (toroidal field)
    bsq: f64, // b·b in BL (calc_angle_brbphibsq)
    angle: f64, // field pitch −b^r b^φ √(g_rr g_φφ)/b² (fieldAngle)
    prermhd: f64, // (γ−1)u (+ Ê/3 with radiation) — the pressure feeding β
    gg33: f64, // MKS2 g_φφ at the cell
    scaleth: f64, // core.dynamo.?.scaleth[clamp(ix)] — raw density-weighted scale height
};

/// What the dynamo produces for one cell.
pub const DynamoOut = struct {
    aphi: f64, // ΔA_φ toroidal vector potential (dyn_a φ slot)
    bphi: f64, // B^φ after DAMPBETA (== input bphi when damping is off)
    skip: bool, // r inside 1.0001·r_horizon → the cell contributes nothing
};

/// C: the per-cell body of mimic_dynamo (magn.c:1042-1151). The ΔA_φ toroidal
/// vector potential and the DAMPBETA azimuthal-field damping, as a pure function
/// of the gathered cell state. Extracted from deltaARows so the dynamo *law* (the
/// α sign flip across the midplane, the horizon/ISCO radial cutoffs, the faczh
/// height window, the β-saturation damping) is unit-testable without a Sim, like
/// dampBphi. `a` is the BH spin, `dt` the sub-step, rhor/risco the BL horizon /
/// ISCO radii. deltaARows is the thin wrapper that gathers the metric, field
/// angle and pressure per cell, then writes aphi/bphi back.
pub fn dynamoDeltaA(dp: Params, a: f64, dt: f64, rhor: f64, risco: f64, c: DynamoCell) DynamoOut {
    if (c.r < 1.0001 * rhor) return .{ .aphi = 0, .bphi = c.bphi, .skip = true }; // avoid the BH

    const omk = 1.0 / (a + @sqrt(c.r * c.r * c.r));
    const pk = 2.0 * pi / omk;

    var angle = c.angle;
    var facangle: f64 = 0;
    if (std.math.isFinite(angle)) {
        if (angle < -1.0) angle = -1.0;
        facangle = @max(0.0, (dp.thetaangle - angle) / dp.thetaangle);
    }

    const facradius = misc.stepFunction(c.r - 1.0 * risco, 0.1 * risco);

    const beta = 0.5 * c.bsq / c.prermhd;

    // CALCHRONTHEGO scale height (clamped to avoid the axis)
    var hrdtheta: f64 = undefined;
    if (dp.calchronthego) {
        hrdtheta = c.scaleth;
        if (hrdtheta > 0.9 * pi / 2.0) hrdtheta = 0.9 * pi / 2.0;
    } else {
        hrdtheta = dp.expectedhr * pi / 2.0;
    }

    const zh = (pi / 2.0 - c.th) / hrdtheta;
    const faczh = @max(0.0, 1.0 - zh * zh); // pow(·,1)
    const facmagnetization = faczh;

    var effalpha = dp.alphadynamo;
    if (dp.alphaflipssign) {
        effalpha = -(pi / 2.0 - c.th) / (hrdtheta / 2.0) * dp.alphadynamo;
    }

    const bphi = c.bphi;
    const aphi = effalpha * (hrdtheta / (pi / 2.0)) / 0.4 *
        dt / pk * c.r * c.gg33 * bphi *
        facradius * facmagnetization * facangle;

    var new_bphi = bphi;
    if (dp.dampbeta) {
        const dbphi = dampBphi(dp.alphabeta, facradius, faczh, dt / pk, beta, dp.betasaturated, bphi);
        new_bphi = bphi + dbphi;
    }

    return .{ .aphi = aphi, .bphi = new_bphi, .skip = false };
}

/// C: calc_avgs_throughout / CALCHRONTHEGO (mpi.c:3168). Density-weighted RMS
/// angular scale height at each radius, sqrt(Σρ√g(π/2−θ)² / Σρ√g) over the
/// θ(,φ) column. Fills core.dynamo.?.scaleth[0..nx). Single rank (TOI=0): gix==0 stays
/// the raw (unnormalized) sum, matching C's `if(gix>0 && gix<TNX)` guard.
/// Band-parallel over ix: each worker owns whole columns, so every column's
/// θ-sum accumulates in serial order; bit-identical at any nthreads.
/// The scale-height passes' view: the Core plus the dynamo state.
fn ShCtx(comptime CoreT: type) type {
    return struct { core: *CoreT, dy: *State };
}

pub fn calcScaleHeight(comptime CoreT: type, core: *CoreT, dy: *State) void {
    // MPI plan §7.2: with ntz>1 the θ,φ column spans all ranks — take the
    // ring-reduced path (partial sums → Allreduce(SUM) → √). The world IS
    // the φ-ring (φ-only decomposition), matching C's world-Allreduce in
    // calc_avgs_throughout (mpi.c:3168).
    if (core.decomp.ntz > 1) {
        calcScaleHeightRing(CoreT, core, dy);
        return;
    }
    var ctx = ShCtx(CoreT){ .core = core, .dy = dy };
    const W = struct {
        fn cols(c: *ShCtx(CoreT), ix0: i64, ix1: i64, res: *threading.ChunkResult) void {
            _ = res;
            scaleHeightCols(CoreT, c, ix0, ix1);
        }
    };
    // the column worker is infallible — no result to check
    _ = threading.parallelRange(ShCtx(CoreT), &ctx, core.team, 0, core.nxi(), W.cols);
}

/// The ntz>1 scale height: per-radius partial sums over the LOCAL φ wedge
/// (same iy-outer/iz-inner accumulation order as scaleHeightAtIx), two
/// pre-√ Allreduce(SUM)s of the nx-length arrays, then the C finalize;
/// ix==0 keeps the raw (now-global) sum (toi==0 identically, so the local
/// index test IS C's `gix>0` global test).
fn calcScaleHeightRing(comptime CoreT: type, core: *CoreT, dy: *State) void {
    var ctx = ShCtx(CoreT){ .core = core, .dy = dy };
    const W = struct {
        fn cols(c: *ShCtx(CoreT), ix0: i64, ix1: i64, res: *threading.ChunkResult) void {
            _ = res;
            partialSumCols(CoreT, c, ix0, ix1);
        }
    };
    _ = threading.parallelRange(ShCtx(CoreT), &ctx, core.team, 0, core.nxi(), W.cols);
    const c = core.comm.?; // ntz>1 implies a backend (Sim.init check 6)
    core.timers.begin(.collect);
    c.allreduceSum(dy.scaleth_sig);
    c.allreduceSum(dy.scaleth_sth);
    core.timers.end();
    for (dy.scaleth, dy.scaleth_sig, dy.scaleth_sth, 0..) |*out, sig, sth, ix| {
        out.* = if (ix > 0) @sqrt(sth / sig) else sth;
    }
}

fn partialSumCols(comptime CoreT: type, c: *ShCtx(CoreT), ix0: i64, ix1: i64) void {
    const core = c.core;
    const dy = c.dy;
    const L = CoreT.Layout;
    const rho_i = comptime L.index(.rho);
    const ny = core.nyi();
    const nz = core.nzi();
    var ix: i64 = ix0;
    while (ix < ix1) : (ix += 1) {
        var sigma: f64 = 0;
        var sth: f64 = 0;
        var iy: i64 = 0;
        while (iy < ny) : (iy += 1) {
            var iz: i64 = 0;
            while (iz < nz) : (iz += 1) {
                const gd = core.cache.gdet(ix, iy, iz);
                const th = core.cache.blGeom(ix, iy, iz).xxvec[2];
                const rho = core.p.get(rho_i, ix, iy, iz);
                const dth = pi / 2.0 - th;
                sigma += rho * gd;
                sth += rho * gd * dth * dth;
            }
        }
        dy.scaleth_sig[@intCast(ix)] = sigma;
        dy.scaleth_sth[@intCast(ix)] = sth;
    }
}

/// The per-column scale-height body for ix ∈ [ix0, ix1).
fn scaleHeightCols(comptime CoreT: type, c: *ShCtx(CoreT), ix0: i64, ix1: i64) void {
    var ix: i64 = ix0;
    while (ix < ix1) : (ix += 1) {
        c.dy.scaleth[@intCast(ix)] = scaleHeightAtIx(CoreT, c.core, ix);
    }
}

/// The pre-√ column sums (Σρ√g, Σρ√gΔθ²) at one radial index over THIS
/// rank's θ(,φ) column; the fold operands of the scale-height reduction.
/// Split out so a globally-folded diagnostic (io/scalars.zig under MPI)
/// can Allreduce(SUM) them before the finalize; the accumulation order is
/// the load-bearing part (iy outer, iz inner; matches partialSumCols).
pub fn scaleHeightPartsAtIx(comptime CoreT: type, core: *const CoreT, ix: i64) struct { sig: f64, sth: f64 } {
    const L = CoreT.Layout;
    const rho_i = comptime L.index(.rho);
    const ny = core.nyi();
    const nz = core.nzi();

    var sigma: f64 = 0;
    var scaleth: f64 = 0;
    var iy: i64 = 0;
    while (iy < ny) : (iy += 1) {
        var iz: i64 = 0;
        while (iz < nz) : (iz += 1) {
            // √−g and BL θ from the caches — no per-cell fill_geometry /
            // cocoN (finding #1); bit-identical to the recomputed values.
            const gd = core.cache.gdet(ix, iy, iz);
            const th = core.cache.blGeom(ix, iy, iz).xxvec[2];
            const rho = core.p.get(rho_i, ix, iy, iz);
            const dth = pi / 2.0 - th;
            sigma += rho * gd;
            scaleth += rho * gd * dth * dth;
        }
    }
    return .{ .sig = sigma, .sth = scaleth };
}

/// The C finalize: √(Σρ√gΔθ²/Σρ√g), except the innermost radial index is
/// left as the raw sum (C's `gix>0` quirk; toi==0 identically, so the
/// local test IS the global one).
pub fn scaleHeightFinalize(ix: i64, sig: f64, sth: f64) f64 {
    return if (ix > 0) @sqrt(sth / sig) else sth;
}

/// The density-weighted scale height at a single radial index; the per-column
/// body of calcScaleHeight as a pure read (no core.dynamo.?.scaleth write), so a
/// diagnostic can query one radius without mutating the Sim or filling the
/// whole grid. C's ix==0 quirk (raw, unnormalized sum) is preserved. Both
/// paths route through here, so calcScaleHeight stays bit-identical.
pub fn scaleHeightAtIx(comptime CoreT: type, core: *const CoreT, ix: i64) f64 {
    const parts = scaleHeightPartsAtIx(CoreT, core, ix);
    return scaleHeightFinalize(ix, parts.sig, parts.sth);
}

/// C: calc_angle_brbphibsq (magn.c:804), non-avg path. The field pitch
/// −b^r b^φ √(g_rr g_φφ) / b² evaluated in BL. Returns {angle, bsq}. The
/// MKS2/BL geometries and the MKS2→BL Jacobian are supplied precomputed
/// (finding #1): the BL sidecar (MetricCache.blGeom/jacMy2Bl) is bit-identical
/// to the per-cell fill_geometry_arb(BL) + coco.dxdx this used to do inline.
fn fieldAngle(comptime CoreT: type, geomMKS2: *const Geometry, geomBL: *const Geometry, jac: [4][4]f64, pp: *const [CoreT.nv]f64) relele.Error!struct { angle: f64, bsq: f64 } {
    const cfg = CoreT.Cfg;
    const L = CoreT.Layout;

    const ppbl = try frames.transPmhdCocoJ(cfg, pp.*, geomMKS2, geomBL, jac);
    const u = try relele.uconUcovFromPrims(
        .{ ppbl[L.index(.vx)], ppbl[L.index(.vy)], ppbl[L.index(.vz)] },
        geomBL,
    );
    const b = mhd.bconBcovBsqFrom4vel(
        .{ ppbl[L.index(.b1)], ppbl[L.index(.b2)], ppbl[L.index(.b3)] },
        u.con,
        u.cov,
        geomBL,
    );
    const brbphi = @sqrt(geomBL.gg[1][1] * geomBL.gg[3][3]) * b.bcon[1] * b.bcon[3];
    return .{ .angle = -brbphi / b.bsq, .bsq = b.bsq };
}

/// C: mimic_dynamo (magn.c:1003). Build ΔA_φ into the dynamo scratch, apply
/// the DAMPBETA azimuthal damping to p[B3], curl ΔA_φ, and superimpose the
/// resulting poloidal B on the domain (+ p2u). `dt` is the sub-step dt.
/// The two per-cell passes (ΔA_φ and superpose+p2u) are row-parallel; every
/// cell writes only its own ΔA_φ / B³ / B+u slots, so the result is
/// bit-identical to serial; the cheap curl stencil between them stays serial
/// (it reads the finished ΔA_φ of neighbour rows; a natural barrier).
/// mimic_dynamo's view: the Core, the dynamo state, the CT work field.
fn DynCtx(comptime CoreT: type) type {
    return struct { core: *CoreT, dy: *State, cts: *ct.Scratch, dt: f64 };
}

pub fn mimicDynamo(comptime CoreT: type, core: *CoreT, dy: *State, cts: *ct.Scratch, dt: f64) Error!void {
    const L = CoreT.Layout;
    if (comptime !L.hasVar(.b1)) return;

    const ny = core.nyi();
    const ylim: i64 = if (ny > 1) 1 else 0;

    threading.parallelZero(core.team, dy.dyn_a.data);

    const Ctx = DynCtx(CoreT);
    var ctx = Ctx{ .core = core, .dy = dy, .cts = cts, .dt = dt };
    const W = struct {
        fn deltaA(c: *Ctx, iy0: i64, iy1: i64, res: *threading.ChunkResult) void {
            deltaARows(CoreT, c, iy0, iy1) catch |e| {
                res.err = e;
            };
        }
        fn superpose(c: *Ctx, iy0: i64, iy1: i64, res: *threading.ChunkResult) void {
            superposeRows(CoreT, c, iy0, iy1) catch |e| {
                res.err = e;
            };
        }
    };

    // ---- ΔA_φ over the domain + one ring including corners (Nloop_6) ----
    {
        const res = threading.parallelRange(Ctx, &ctx, core.team, -ylim, ny + ylim, W.deltaA);
        if (res.err) |e| return e;
    }

    // ---- curl ΔA_φ → B^i in the scratch slots 3..5, superimpose on domain ----
    // curlFromA clobbers &core.ct.vecpot 0..2 (corner A) and returns it with B in
    // 3..5; the superpose pass below reads those slots from core.vecpot.
    _ = ct.curlFromA(CoreT, core, &cts.vecpot, &dy.dyn_a, 0);

    {
        const res = threading.parallelRange(Ctx, &ctx, core.team, 0, ny, W.superpose);
        if (res.err) |e| return e;
    }
}

/// The ΔA_φ + DAMPBETA body for iy ∈ [iy0, iy1) (all iz, ix incl. the ring).
fn deltaARows(comptime CoreT: type, c: *DynCtx(CoreT), iy0: i64, iy1: i64) relele.Error!void {
    const core = c.core;
    const dy = c.dy;
    const dt = c.dt;
    const cfg = CoreT.Cfg;
    const L = CoreT.Layout;
    const dp = &dy.params;

    const b3 = comptime L.index(.b3);
    const uu_i = comptime L.index(.uu);

    const nx = core.nxi();
    const nz = core.nzi();
    const rhor = metric.rHorizonBL(core.phys.mp.a);
    const risco = metric.rIscoBL(core.phys.mp.a);

    const xlim: i64 = if (nx > 1) 1 else 0;
    const zlim: i64 = if (nz > 1) 1 else 0;

    var iz: i64 = -zlim;
    while (iz < nz + zlim) : (iz += 1) {
        var iy: i64 = iy0;
        while (iy < iy1) : (iy += 1) {
            var ix: i64 = -xlim;
            while (ix < nx + xlim) : (ix += 1) {
                const geom = core.cache.fillGeometry(ix, iy, iz);
                const geomBL = core.cache.blGeom(ix, iy, iz);
                const jac = core.cache.jacMy2Bl(ix, iy, iz);
                var pp: [CoreT.nv]f64 = undefined;
                core.p.load(ix, iy, iz, &pp);

                // dynamo scratch starts at zero (memset above)
                const fa = try fieldAngle(CoreT, &geom, geomBL, jac, &pp);

                // BL {r, θ} come from the cached BL geometry's position vector
                // (bit-identical to coco.cocoN(geom.xxvec, coords, .bl, mp)). The
                // horizon cutoff stays in the wrapper so the calc_ff_Rtt closure
                // work below runs on exactly the cells C's mimic_dynamo reaches
                // (fieldAngle already ran for all cells, matching C's ordering).
                const r = geomBL.xxvec[1];
                const th = geomBL.xxvec[2];
                if (r < 1.0001 * rhor) continue; // avoid the BH

                // pressure feeding β: gas (+ radiation Ê/3 with the M1 closure)
                const gamma = core.phys.gam; // no CONSISTENTGAMMA
                var prermhd = (gamma - 1.0) * pp[uu_i];
                if (comptime L.hasVar(.ee)) {
                    const ff = try radiation.calcFfRtt(cfg, pp, &geom);
                    const ehat = -ff.rtt;
                    prermhd += ehat / 3.0;
                }

                // the raw density-weighted scale height for this radial column
                const gix = std.math.clamp(ix, 0, nx - 1);

                // pure dynamo law over the gathered cell state
                const out = dynamoDeltaA(dp.*, core.phys.mp.a, dt, rhor, risco, .{
                    .r = r,
                    .th = th,
                    .bphi = pp[b3],
                    .bsq = fa.bsq,
                    .angle = fa.angle,
                    .prermhd = prermhd,
                    .gg33 = geom.gg[3][3],
                    .scaleth = dy.scaleth[@intCast(gix)],
                });

                dy.dyn_a.set(2, ix, iy, iz, out.aphi); // φ component (B3 slot)
                if (dp.dampbeta) core.p.set(b3, ix, iy, iz, out.bphi);
            }
        }
    }
}

/// The superpose body for iy ∈ [iy0, iy1): add the curled ΔA_φ field
/// (vecpot 3..5) to the domain B and refresh the magnetic conserveds.
fn superposeRows(comptime CoreT: type, c: *DynCtx(CoreT), iy0: i64, iy1: i64) relele.Error!void {
    const core = c.core;
    const cfg = CoreT.Cfg;
    const L = CoreT.Layout;
    const b1 = comptime L.index(.b1);

    const nx = core.nxi();
    const nz = core.nzi();

    var jz: i64 = 0;
    while (jz < nz) : (jz += 1) {
        var jy: i64 = iy0;
        while (jy < iy1) : (jy += 1) {
            var jx: i64 = 0;
            while (jx < nx) : (jx += 1) {
                var pp: [CoreT.nv]f64 = undefined;
                core.p.load(jx, jy, jz, &pp);
                pp[b1] += c.cts.vecpot.get(3, jx, jy, jz);
                pp[b1 + 1] += c.cts.vecpot.get(4, jx, jy, jz);
                pp[b1 + 2] += c.cts.vecpot.get(5, jx, jy, jz);
                const geom = core.cache.fillGeometry(jx, jy, jz);
                const uu = try p2u_mod.p2u(cfg, pp, &geom, core.phys.gam);
                core.p.store(jx, jy, jz, &pp);
                core.u.set(b1, jx, jy, jz, uu[b1]);
                core.u.set(b1 + 1, jx, jy, jz, uu[b1 + 1]);
                core.u.set(b1 + 2, jx, jy, jz, uu[b1 + 2]);
            }
        }
    }
}

/// C: apply_dynamo (finite.c:1370). The full per-sub-step sequence,
/// including its MPI order: mpi_exchangedata → calc_avgs_throughout →
/// set_bc → mimic_dynamo → calc_u2p (the third canonical exchange site,
/// MPI plan §6.1). Operator-level: takes the whole Sim because it composes
/// the Sim's setBc / calcU2p with the kernels above.
pub fn applyDynamo(comptime SimT: type, sim: *SimT, t: f64, dt: f64) Error!void {
    const L = SimT.Layout;
    if (comptime !L.hasVar(.b1)) return;
    const dy = &(sim.dynamo.?);
    sim.core.exchangeHalos();
    calcScaleHeight(SimT.CoreT, &sim.core, dy);
    try sim.setBc(t, false);
    try mimicDynamo(SimT.CoreT, &sim.core, dy, &sim.ct, dt);
    try sim.calcU2p(t);
}
