//! Theory gates for evolution in curved spacetime (Kerr–Schild, a = 0)
//! through the spherical machinery the Minkowski gates (evolution_tests.zig,
//! mhd_evolution_tests.zig) never touch: the θ-direction metric sources, the
//! polar-axis band (CORRECT_POLARAXIS, sim/polaraxis.zig), the stock polar
//! reflection BC (problems/common/bcs.zig polarReflect, bc.c:150-190) and
//! flux-CT with a curved-space √−g (sim/ct.zig).
//!
//!  * 2D axisymmetric Bondi/Michel accretion (Michel 1972) on a full-θ KS
//!    grid r ∈ [1.5, 20], θ ∈ [0.001π, 0.999π]: the spherically symmetric transonic
//!    solution is an exact stationary solution of the 2D equations, so the
//!    L1 drift of ρ is pure truncation error and must converge ~2nd order
//!    with N (it matches the 1D gate's numbers). The profile must stay
//!    θ-uniform to ROUNDOFF: the θ-momentum flux term ∂_θ(√−g p)/√−g and
//!    the source p·Γ^λ_θλ = p ∂_θ ln √−g cancel exactly on the grid because
//!    the Christoffel trace is built from finite differences of √−g (the
//!    correction's defining property, metric_tests.zig "precompute:
//!    Christoffel trace equals the FD gdet gradient"), and a = 0 leaves no
//!    other θ source; the cot θ singularity at the axis is what the polar
//!    band absorbs.
//!  * magnetized Bondi (Gammie, McKinney & Tóth 2003, §5.3; the C problem
//!    PROBLEMS/FFBONDI/init.c:44 seeds the same A_φ = bfac (1 − cos θ)): a
//!    split monopole B^r = B0/r² has ∇_μ F^μν = 0, so J = 0, the Lorentz
//!    force vanishes and the hydro solution is stationary at any β. The
//!    drift and the generated B^θ, B^φ must converge with N; corner divB
//!    stays at machine zero throughout (flux-CT), and the discrete curl of
//!    the potential is itself an exact monopole, B^r r² = const to roundoff
//!    (corner averaging of a θ-only A_φ, center √−g = r² sin θ).

const std = @import("std");
const config = @import("../config.zig");
const grid_mod = @import("../grid.zig");
const sim_mod = @import("../sim.zig");
const relele = @import("../relele.zig");
const hydro = @import("../physics/hydro.zig");
const bcs = @import("../problems/common/bcs.zig");
const ev = @import("evolution_tests.zig");

const Grid = grid_mod.Grid;
const expect = std.testing.expect;
const Michel = ev.Michel;
const michel_gam = ev.michel_gam;

const cfg_hd = config.Config{
    .modules = &.{.hydro},
    .reconstruction = .ppm,
    .flux = .laxf,
    .timestepping = .rk2imex,
    .coords = .ks,
};
const cfg_mhd = config.Config{
    .modules = &.{ .hydro, .mhd },
    .reconstruction = .ppm,
    .flux = .laxf,
    .timestepping = .rk2imex,
    .coords = .ks,
};

/// sonic radius (M = 1)
const rc: f64 = 8.0;

fn Harness(comptime cfg: config.Config) type {
    return struct {
        const SimT = sim_mod.Sim(cfg);
        const CoreT = SimT.CoreT;
        const L = SimT.Layout;
        const NV = SimT.nv;
        const has_b = L.hasVar(.b1);

        const Ctx = struct {
            michel: Michel,
            /// monopole strength, B^r = b0 / r² (0 ≡ unmagnetized)
            b0: f64,
            /// what the B slots carry: the vector potential A_i until
            /// calcBfromA has run (ko.c: init → set_bc → calc_BfromA, so the
            /// ghost cells must hold A too), the field afterwards
            phase: enum { potential, field } = .field,
        };

        /// Michel primitives at cell (ix, iy); ghost ix allowed (the x faces
        /// are held at the exact solution, inside the horizon included, as
        /// the 1D gate does).
        fn prims(c: *const Ctx, core: *const CoreT, ix: i64, iy: i64) [NV]f64 {
            const r = core.grid.xc(ix);
            const m = c.michel.solve(r, rc);
            const p = c.michel.k * std.math.pow(f64, m.rho, michel_gam);
            const uint = p / (michel_gam - 1.0);

            const geom = core.cache.fillGeometry(ix, iy, 0);
            var ucon = [4]f64{ 0, -m.u, 0, 0 };
            ucon[0] = relele.utFromSpatialUcon(ucon, &geom) catch unreachable;
            const vr = relele.convert(ucon, .vel4, .velr, &geom, .recompute_ut) catch unreachable;

            var pp: [NV]f64 = @splat(0);
            pp[L.index(.rho)] = m.rho;
            pp[L.index(.uu)] = uint;
            pp[L.index(.vx)] = vr[1];
            pp[L.index(.vy)] = vr[2];
            pp[L.index(.vz)] = vr[3];
            pp[L.index(.entr)] = hydro.sFromU(m.rho, uint, michel_gam);
            if (comptime has_b) {
                switch (c.phase) {
                    // A_i in the B slots (C: FFBONDI init.c:44-50)
                    .potential => pp[L.index(.b3)] = c.b0 * (1.0 - @cos(core.grid.yc(iy))),
                    .field => pp[L.index(.b1)] = c.b0 / (r * r),
                }
            }
            return pp;
        }

        fn bc(ctx: ?*const anyopaque, core: *const CoreT, ix: i64, iy: i64, iz: i64, t: f64, ifinit: bool, face: sim_mod.BcFace) relele.Error![NV]f64 {
            _ = t;
            _ = ifinit;
            const c: *const Ctx = @ptrCast(@alignCast(ctx.?));
            // the potential is analytic on every face (its θ-mirror across the
            // first face is not the potential at the ghost θ); the field
            // uses the stock polar reflection
            if (comptime has_b) {
                if (c.phase == .potential) return prims(c, core, ix, iy);
            }
            return switch (face) {
                .xlo, .xhi => prims(c, core, ix, iy),
                .ylo, .yhi => bcs.polarReflect(CoreT, core, ix, iy, iz, face),
                .zlo, .zhi => unreachable,
            };
        }

        const Result = struct {
            /// L1(ρ(t_end) − ρ(0)) / L1(ρ(0)) over the domain
            drift: f64,
            /// max over cells of |ρ − ⟨ρ⟩_θ| / ⟨ρ⟩_θ at t_end
            theta_nonuni: f64,
            /// max |B^r r² / b0 − 1| at t = 0 (monopole exactness); interior
            mono_err: f64,
            /// max relative change of B^r over the run; interior
            db_r: f64,
            /// max orthonormal |B^θ̂|, |B^φ̂| relative to |B^r| at t_end; interior
            db_t: f64,
            /// max corner |divB| · Δr / max|B^r| at t = 0 and t_end; interior
            divb0: f64,
            divb1: f64,
        };

        fn idx(ix: i64, iy: i64, nx: usize) usize {
            return @as(usize, @intCast(iy)) * nx + @as(usize, @intCast(ix));
        }

        /// The field diagnostics are taken on a fixed physical region away
        /// from every boundary, r ∈ [3, 18], θ ∈ [0.5, π−0.5]. Two set_bc
        /// properties (C's, reproduced verbatim) make the boundary layer
        /// unphysical for a 1/r² field and are not the solver's: the corner
        /// ghosts are diagonal averages of two ghosts at different r
        /// (finite.c:3203), so the four domain-corner cells see a
        /// non-monopole neighbour in both the A-curl and the CT EMF (the
        /// B^θ this seeds is then carried a few cells along the edge rows);
        /// and the reflected θ-ghosts carry −B^θ, so once an edge row holds
        /// any B^θ its cell-centered corner divergence is nonzero by
        /// construction. A fixed region (not a fixed cell count) keeps the
        /// measurement comparable across resolutions.
        fn interior(r: f64, th: f64) bool {
            return r >= 3.0 and r <= 18.0 and th >= 0.5 and th <= std.math.pi - 0.5;
        }

        fn divbScaled(s: *const SimT, dr: f64) f64 {
            var worst: f64 = 0;
            var bmax: f64 = 0;
            const g = &s.core.grid;
            var iy: i64 = 0;
            while (iy < s.nyi()) : (iy += 1) {
                var ix: i64 = 0;
                while (ix < s.nxi()) : (ix += 1) {
                    if (!interior(g.xc(ix), g.yc(iy))) continue;
                    worst = @max(worst, @abs(s.calcDivB(ix, iy, 0)));
                    bmax = @max(bmax, @abs(s.core.p.get(L.index(.b1), ix, iy, 0)));
                }
            }
            return worst * dr / bmax;
        }

        fn run(a: std.mem.Allocator, nx: usize, ny: usize, tend: f64, beta_c: ?f64) !Result {
            const g = Grid.init(.{
                .nx = nx,
                .ny = ny,
                .ng = 3,
                .minx = 1.5,
                .maxx = 20.0,
                // C: MINY = 0.001 — the axis face itself is excluded (g^φφ = 1/sin²θ
                // diverges on it); the polar band and the reflection BC act on
                // the first face, exactly as in production PUFFY.
                .miny = 0.001 * std.math.pi,
                .maxy = 0.999 * std.math.pi,
            });
            var ctx = Ctx{ .michel = Michel.init(rc), .b0 = 0 };
            if (beta_c) |beta| {
                // β ≡ p_gas / (B²/2) at the sonic point, where ρ_c ≡ 1 and
                // p_c = K: B0² / (2 rc⁴) = K / β
                ctx.b0 = rc * rc * @sqrt(2.0 * ctx.michel.k / beta);
                ctx.phase = .potential;
            }
            var s = try SimT.init(a, g, .{
                .phys = .{ .coords = .ks, .gam = michel_gam },
                .num = .{ .polaraxis = .{ .ncells = 2 } },
                .bc = .{
                    .x = .{ .specific = .{ .f = &bc, .ctx = &ctx } },
                    .y = .{ .specific = .{ .f = &bc, .ctx = &ctx } },
                },
            });
            defer s.deinit();

            var iy: i64 = 0;
            while (iy < s.nyi()) : (iy += 1) {
                var ix: i64 = 0;
                while (ix < s.nxi()) : (ix += 1) {
                    try s.initCell(ix, iy, 0, prims(&ctx, &s.core, ix, iy));
                }
            }
            var res: Result = .{ .drift = 0, .theta_nonuni = 0, .mono_err = 0, .db_r = 0, .db_t = 0, .divb0 = 0, .divb1 = 0 };
            if (comptime has_b) {
                try s.setBc(0.0, true);
                try s.calcBfromA(true);
                // the corner-difference curl carries a sin Δθ/Δθ factor
                // relative to B0: hold the ghost cells at the discrete
                // monopole so the field is ONE monopole across the x faces
                const rm = g.xc(@intCast(nx / 2));
                ctx.b0 = s.core.p.get(L.index(.b1), @intCast(nx / 2), @intCast(ny / 2), 0) * rm * rm;
                ctx.phase = .field;
            }
            try s.finishInit();

            const n = nx * ny;
            const rho0 = try a.alloc(f64, n);
            defer a.free(rho0);
            const br0 = try a.alloc(f64, n);
            defer a.free(br0);
            iy = 0;
            while (iy < s.nyi()) : (iy += 1) {
                var ix: i64 = 0;
                while (ix < s.nxi()) : (ix += 1) {
                    const k = idx(ix, iy, nx);
                    rho0[k] = s.core.p.get(L.index(.rho), ix, iy, 0);
                    if (comptime has_b) {
                        const r = g.xc(ix);
                        br0[k] = s.core.p.get(L.index(.b1), ix, iy, 0);
                        if (interior(r, g.yc(iy))) res.mono_err = @max(res.mono_err, @abs(br0[k] * r * r / ctx.b0 - 1.0));
                    }
                }
            }
            if (comptime has_b) res.divb0 = divbScaled(&s, g.dx);
            while (s.t < tend) {
                var dt: ?f64 = null;
                if (s.t + 1.0 / s.core.tstepdenmax > tend) dt = tend - s.t;
                s.step(dt) catch |e| {
                    std.debug.print("step failed at t={d:.4} nstep={d}: {s}\n", .{ s.t, s.nstep, @errorName(e) });
                    return e;
                };
            }

            var l1: f64 = 0;
            var norm: f64 = 0;
            var ix: i64 = 0;
            while (ix < s.nxi()) : (ix += 1) {
                const r = g.xc(ix);
                var mean: f64 = 0;
                iy = 0;
                while (iy < s.nyi()) : (iy += 1) mean += s.core.p.get(L.index(.rho), ix, iy, 0);
                mean /= @as(f64, @floatFromInt(ny));
                iy = 0;
                while (iy < s.nyi()) : (iy += 1) {
                    const k = idx(ix, iy, nx);
                    const rho = s.core.p.get(L.index(.rho), ix, iy, 0);
                    l1 += @abs(rho - rho0[k]);
                    norm += @abs(rho0[k]);
                    res.theta_nonuni = @max(res.theta_nonuni, @abs(rho - mean) / mean);
                    if (comptime has_b) {
                        if (!interior(r, g.yc(iy))) continue;
                        const br = s.core.p.get(L.index(.b1), ix, iy, 0);
                        res.db_r = @max(res.db_r, @abs(br - br0[k]) / @abs(br0[k]));
                        const bth = r * @abs(s.core.p.get(L.index(.b2), ix, iy, 0));
                        const bph = r * @sin(g.yc(iy)) * @abs(s.core.p.get(L.index(.b3), ix, iy, 0));
                        res.db_t = @max(res.db_t, @max(bth, bph) / @abs(br0[k]));
                    }
                }
            }
            res.drift = l1 / norm;
            if (comptime has_b) res.divb1 = divbScaled(&s, g.dx);
            return res;
        }
    };
}

/// Default-suite run length; the slow set repeats at t = 10 (the 1D gate's).
const tend_fast: f64 = 3.0;

fn gateHydro(tend: f64) !void {
    const a = std.testing.allocator;
    const H = Harness(cfg_hd);
    const c = try H.run(a, 32, 16, tend, null);
    const f = try H.run(a, 64, 32, tend, null);
    const order = std.math.log2(c.drift / f.drift);
    std.debug.print("bondi2d t={d}: drift 32x16 {e:.3} 64x32 {e:.3} order {d:.2}; theta-nonuni {e:.3} {e:.3}\n", .{ tend, c.drift, f.drift, order, c.theta_nonuni, f.theta_nonuni });
    try expect(f.drift < 2e-2);
    try expect(order >= 1.5);
    // exact discrete θ-balance (see the header): roundoff, not truncation
    try expect(c.theta_nonuni < 1e-12);
    try expect(f.theta_nonuni < 1e-12);
}

test "KS 2D: axisymmetric Bondi/Michel through the polar axis is stationary, θ-uniform to roundoff, ~2nd order" {
    try gateHydro(tend_fast);
}

fn gateMhd(tend: f64) !void {
    const a = std.testing.allocator;
    const H = Harness(cfg_mhd);
    const c = try H.run(a, 32, 16, tend, 10.0);
    const f = try H.run(a, 64, 32, tend, 10.0);
    const order = std.math.log2(c.drift / f.drift);
    std.debug.print("magbondi t={d}: drift 32x16 {e:.3} 64x32 {e:.3} order {d:.2}; dB^r {e:.3} {e:.3}; B^t,B^p/B^r {e:.3} {e:.3}; mono {e:.3}; divB {e:.3}->{e:.3}\n", .{ tend, c.drift, f.drift, order, c.db_r, f.db_r, c.db_t, f.db_t, f.mono_err, f.divb0, f.divb1 });
    // discrete curl of A_φ = B0 (1 − cos θ) is an exact monopole
    try expect(c.mono_err < 1e-12);
    try expect(f.mono_err < 1e-12);
    // flux-CT: corner divB at machine zero before and after
    try expect(c.divb0 < 1e-13 and c.divb1 < 1e-13);
    try expect(f.divb0 < 1e-13 and f.divb1 < 1e-13);
    // stationarity, converging; the field itself stays a monopole to
    // ~1e-7 (measured 9e-8 at 64×32, t = 3) and that error converges too
    try expect(f.drift < 2e-2);
    try expect(order >= 1.5);
    try expect(f.db_r < 1e-5 and f.db_r < c.db_r);
    try expect(f.db_t < 1e-5 and f.db_t < c.db_t);
}

test "KS 2D: magnetized Bondi (split monopole) — exact monopole, divB machine zero, flow and B stationary to truncation" {
    try gateMhd(tend_fast);
}

test "KS 2D slow: both gates at t = 10" {
    if (!@import("build_options").slow_tests) return error.SkipZigTest;
    try gateHydro(10.0);
    try gateMhd(10.0);
}
