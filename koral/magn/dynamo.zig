//! Mean-field "mimic" dynamo (PUFFY's MIMICDYNAMO), transcribed from KORAL:
//!
//!   apply_dynamo    finite.c:1370  — calc_avgs_throughout (scale height) →
//!                                    set_bc → mimic_dynamo → calc_u2p.
//!   mimic_dynamo    magn.c:1003    — per cell (domain + 1 ring incl. corners)
//!                                    build an extra toroidal vector potential
//!                                    ΔA_φ ∝ α_dyn · (dt/P_K) · B^φ, weighted
//!                                    by radius / field-angle / height; the
//!                                    DAMPBETA azimuthal-field damping;
//!                                    calc_BfromA(ΔA_φ) → curl; superimpose the
//!                                    new poloidal B on the domain + p2u.
//!   calc_avgs_throughout  mpi.c:3163 — the CALCHRONTHEGO density-weighted RMS
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
//!    functions are dead (unused by the ΔA_φ / DAMPBETA expressions) — skipped.
//!  * scaleth_otg[0] keeps the *unnormalized* raw sum (the sqrt loop skips
//!    gix==0, mpi.c:3225) — a C quirk transcribed verbatim; the innermost
//!    column's faczH collapses to 0 there anyway.
//!  * pow(1-zH², 1) == 1-zH² (libm special case) → written as a plain sub.

const std = @import("std");
const config = @import("../config.zig");
const layout = @import("../layout.zig");
const relele = @import("../relele.zig");
const mhd = @import("../physics/mhd.zig");
const radiation = @import("../physics/radiation.zig");
const radvisc = @import("../physics/radvisc.zig");
const frames = @import("../frames.zig");
const coco = @import("../metric/coco.zig");
const precompute = @import("../metric/precompute.zig");
const p2u_mod = @import("../p2u.zig");
const ct = @import("ct.zig");
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
/// B^φ, grows with (β−β_sat), and is clamped so it never overshoots zero — so
/// |B^φ| is non-increasing and its sign is preserved (saturation toward 0
/// once β exceeds BETASATURATED). Pure, so it can be unit-tested directly.
pub fn dampBphi(alphabeta: f64, facradius: f64, faczh: f64, dt_over_pk: f64, beta: f64, betasat: f64, bphi: f64) f64 {
    var dbphi = -alphabeta * facradius * faczh * dt_over_pk *
        @max(0.0, beta - betasat) / betasat * bphi;
    if ((dbphi + bphi) * bphi < 0.0) dbphi = -bphi; // no overshoot past zero
    return dbphi;
}

/// C: calc_avgs_throughout / CALCHRONTHEGO (mpi.c:3168) — density-weighted RMS
/// angular scale height at each radius, sqrt(Σρ√g(π/2−θ)² / Σρ√g) over the
/// θ(,φ) column. Fills sim.scaleth[0..nx). Serial (TOI=0): gix==0 stays the
/// raw (unnormalized) sum, matching C's `if(gix>0 && gix<TNX)` guard.
pub fn calcScaleHeight(comptime SimT: type, sim: *SimT) void {
    const cfg = SimT.Cfg;
    const L = SimT.Layout;
    const rho_i = comptime L.index(.rho);

    const nx = sim.nxi();
    const ny = sim.nyi();
    const nz = sim.nzi();

    var ix: i64 = 0;
    while (ix < nx) : (ix += 1) {
        var sigma: f64 = 0;
        var scaleth: f64 = 0;
        var iy: i64 = 0;
        while (iy < ny) : (iy += 1) {
            var iz: i64 = 0;
            while (iz < nz) : (iz += 1) {
                const geom = sim.cache.fillGeometry(ix, iy, iz);
                const xxbl = coco.cocoN(geom.xxvec, cfg.coords, .bl, sim.opt.mp);
                const rho = sim.p.get(rho_i, ix, iy, iz);
                const dth = pi / 2.0 - xxbl[2];
                sigma += rho * geom.gdet;
                scaleth += rho * geom.gdet * dth * dth;
            }
        }
        // C quirk: the innermost radial index is left unnormalized
        sim.scaleth[@intCast(ix)] = if (ix > 0) @sqrt(scaleth / sigma) else scaleth;
    }
}

/// BL geometry at a cell center (C: fill_geometry_arb KERRCOORDS).
fn geomBLat(comptime SimT: type, sim: *const SimT, ix: i64, iy: i64, iz: i64) Geometry {
    const cfg = SimT.Cfg;
    const g = &sim.grid;
    const xx = [4]f64{ 0.0, g.xc(ix), g.yc(iy), g.zc(iz) };
    const xxbl = coco.cocoN(xx, cfg.coords, .bl, sim.opt.mp);
    var geom = precompute.geometryAt(.bl, sim.opt.mp, xxbl);
    geom.ix = ix;
    geom.iy = iy;
    geom.iz = iz;
    return geom;
}

/// C: calc_angle_brbphibsq (magn.c:804), non-avg path — the field pitch
/// −b^r b^φ √(g_rr g_φφ) / b² evaluated in BL. Returns {angle, bsq}.
fn fieldAngle(comptime SimT: type, sim: *const SimT, ix: i64, iy: i64, iz: i64, pp: *const [SimT.nv]f64) relele.Error!struct { angle: f64, bsq: f64 } {
    const cfg = SimT.Cfg;
    const L = SimT.Layout;
    const geomMKS2 = sim.cache.fillGeometry(ix, iy, iz);
    const geomBL = geomBLat(SimT, sim, ix, iy, iz);

    const ppbl = try frames.transPmhdCoco(cfg, pp.*, &geomMKS2, &geomBL, sim.opt.mp);
    const u = try relele.uconUcovFromPrims(
        .{ ppbl[L.index(.vx)], ppbl[L.index(.vy)], ppbl[L.index(.vz)] },
        &geomBL,
    );
    const b = mhd.bconBcovBsqFrom4vel(
        .{ ppbl[L.index(.b1)], ppbl[L.index(.b2)], ppbl[L.index(.b3)] },
        u.con,
        u.cov,
        &geomBL.gg,
    );
    const brbphi = @sqrt(geomBL.gg[1][1] * geomBL.gg[3][3]) * b.bcon[1] * b.bcon[3];
    return .{ .angle = -brbphi / b.bsq, .bsq = b.bsq };
}

/// C: mimic_dynamo (magn.c:1003) — build ΔA_φ into the dynamo scratch, apply
/// the DAMPBETA azimuthal damping to p[B3], curl ΔA_φ, and superimpose the
/// resulting poloidal B on the domain (+ p2u). `dt` is the sub-step dt.
pub fn mimicDynamo(comptime SimT: type, sim: *SimT, dt: f64) relele.Error!void {
    const cfg = SimT.Cfg;
    const L = SimT.Layout;
    if (comptime !L.hasVar(.b1)) return;
    const dp = &sim.opt.dynamo_params;

    const b1 = comptime L.index(.b1);
    const b3 = comptime L.index(.b3);
    const uu_i = comptime L.index(.uu);

    const nx = sim.nxi();
    const ny = sim.nyi();
    const nz = sim.nzi();
    const rhor = radvisc.rHorizonBL(sim.opt.mp.a);
    const risco = radvisc.rIscoBL(sim.opt.mp.a);

    @memset(sim.dynA.data, 0);

    // ---- ΔA_φ over the domain + one ring including corners (Nloop_6) ----
    const xlim: i64 = if (nx > 1) 1 else 0;
    const ylim: i64 = if (ny > 1) 1 else 0;
    const zlim: i64 = if (nz > 1) 1 else 0;

    var iz: i64 = -zlim;
    while (iz < nz + zlim) : (iz += 1) {
        var iy: i64 = -ylim;
        while (iy < ny + ylim) : (iy += 1) {
            var ix: i64 = -xlim;
            while (ix < nx + xlim) : (ix += 1) {
                const geom = sim.cache.fillGeometry(ix, iy, iz);
                var pp: [SimT.nv]f64 = undefined;
                sim.p.load(ix, iy, iz, &pp);

                // dynamo scratch starts at zero (memset above)
                const fa = try fieldAngle(SimT, sim, ix, iy, iz, &pp);
                var angle = fa.angle;
                const bsq = fa.bsq;

                const xxbl = coco.cocoN(geom.xxvec, cfg.coords, .bl, sim.opt.mp);
                const r = xxbl[1];
                const th = xxbl[2];
                if (r < 1.0001 * rhor) continue; // avoid the BH

                const omk = 1.0 / (sim.opt.mp.a + @sqrt(r * r * r));
                const pk = 2.0 * pi / omk;

                var facangle: f64 = 0;
                if (std.math.isFinite(angle)) {
                    if (angle < -1.0) angle = -1.0;
                    facangle = @max(0.0, (dp.thetaangle - angle) / dp.thetaangle);
                }

                const facradius = radvisc.stepFunction(r - 1.0 * risco, 0.1 * risco);

                const gamma = sim.opt.gam; // no CONSISTENTGAMMA
                var prermhd = (gamma - 1.0) * pp[uu_i];
                if (comptime L.hasVar(.ee)) {
                    const ff = try radiation.calcFfRtt(cfg, pp, &geom);
                    const ehat = -ff.rtt;
                    prermhd += ehat / 3.0;
                }

                const beta = 0.5 * bsq / prermhd;

                // CALCHRONTHEGO scale height (clamped to avoid the axis)
                var hrdtheta: f64 = undefined;
                if (dp.calchronthego) {
                    const gix = std.math.clamp(ix, 0, nx - 1);
                    hrdtheta = sim.scaleth[@intCast(gix)];
                    if (hrdtheta > 0.9 * pi / 2.0) hrdtheta = 0.9 * pi / 2.0;
                } else {
                    hrdtheta = dp.expectedhr * pi / 2.0;
                }

                const zh = (pi / 2.0 - th) / hrdtheta;
                const faczh = @max(0.0, 1.0 - zh * zh); // pow(·,1)
                const facmagnetization = faczh;

                var effalpha = dp.alphadynamo;
                if (dp.alphaflipssign) {
                    effalpha = -(pi / 2.0 - th) / (hrdtheta / 2.0) * dp.alphadynamo;
                }

                const bphi = pp[b3];
                const aphi = effalpha * (hrdtheta / (pi / 2.0)) / 0.4 *
                    dt / pk * r * geom.gg[3][3] * bphi *
                    facradius * facmagnetization * facangle;

                sim.dynA.set(2, ix, iy, iz, aphi); // φ component (B3 slot)

                if (dp.dampbeta) {
                    const dbphi = dampBphi(dp.alphabeta, facradius, faczh, dt / pk, beta, dp.betasaturated, bphi);
                    sim.p.set(b3, ix, iy, iz, bphi + dbphi);
                }
            }
        }
    }

    // ---- curl ΔA_φ → B^i (vecpot 3..5), superimpose on the domain ----
    ct.curlFromA(SimT, sim, &sim.dynA, 0);

    var jz: i64 = 0;
    while (jz < nz) : (jz += 1) {
        var jy: i64 = 0;
        while (jy < ny) : (jy += 1) {
            var jx: i64 = 0;
            while (jx < nx) : (jx += 1) {
                var pp: [SimT.nv]f64 = undefined;
                sim.p.load(jx, jy, jz, &pp);
                pp[b1] += sim.vecpot.get(3, jx, jy, jz);
                pp[b1 + 1] += sim.vecpot.get(4, jx, jy, jz);
                pp[b1 + 2] += sim.vecpot.get(5, jx, jy, jz);
                const geom = sim.cache.fillGeometry(jx, jy, jz);
                const uu = try p2u_mod.p2u(cfg, pp, &geom, sim.opt.gam);
                sim.p.store(jx, jy, jz, &pp);
                sim.u.set(b1, jx, jy, jz, uu[b1]);
                sim.u.set(b1 + 1, jx, jy, jz, uu[b1 + 1]);
                sim.u.set(b1 + 2, jx, jy, jz, uu[b1 + 2]);
            }
        }
    }
}

/// C: apply_dynamo (finite.c:1370) — the full per-sub-step sequence.
pub fn applyDynamo(comptime SimT: type, sim: *SimT, t: f64, dt: f64) Error!void {
    const L = SimT.Layout;
    if (comptime !L.hasVar(.b1)) return;
    calcScaleHeight(SimT, sim);
    try sim.setBc(t, false);
    try mimicDynamo(SimT, sim, dt);
    try sim.calcU2p();
}
