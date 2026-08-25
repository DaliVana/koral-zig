//! M10 theory battery: the Farris et al. (2008) / Sądowski et al. (2013)
//! radiative shock tubes evolved to a stationary state and validated
//! against the semi-analytic stationary system itself:
//!
//!   (a) stationarity: the profile stops evolving,
//!   (b) plateaus: the near-boundary states match the published
//!                       asymptotic (jump-condition + LTE) states,
//!   (c) flux constancy; the ANALYTIC total fluxes (ρu^x, T^{xν}+R^{xν})
//!                       evaluated from the numerical primitives are
//!                       constant across the domain (they are the first
//!                       integrals of the stationary ODE system; also the
//!                       Rankine-Hugoniot conditions across the embedded
//!                       gas shock of tubes 1/2),
//!   (d) ODE residual: for smooth tubes, ∂x R^{xν} = −G^ν pointwise
//!                       along the profile (the remaining two equations of
//!                       the stationary system).
//!
//! (b)+(c)+(d) together state that the computed profile IS a solution of
//! the semi-analytic stationary equations with the correct boundary
//! values: the same content as integrating the ODEs and comparing in L1,
//! without the shooting fragility. Each run finishes with a dt/5 "polish"
//! phase: the RK2IMEX fixed point carries a dt-proportional splitting bias
//! in the pointwise balance (F and G are evaluated at intermediate states
//! offset by γ·dt·G) that the polish shrinks ~3×; the ~0.3 residual floor
//! is this bias at dt/5, resolution-independent at fixed CFL fraction.
//!
//! Tubes 3a and 4a run in the default suite; 1 (non-relativistic: flush
//! time ~3200), 2 (contact residue flushes at v ≈ 0.08) and 4b, plus
//! hi-res 3a/4a, run under `zig build test -Dslow-tests`. Tube 3b (κ = 25)
//! is excluded. See the note in the slow set.

const std = @import("std");
const config = @import("../config.zig");
const layout = @import("../layout.zig");
const grid_mod = @import("../grid.zig");
const sim_mod = @import("../sim.zig");
const hydro = @import("../physics/hydro.zig");
const radiation = @import("../physics/radiation.zig");
const radforce = @import("../physics/radforce.zig");
const relele = @import("../relele.zig");
const invert = @import("../solve/invert.zig");
const invert_rad = @import("../solve/invert_rad.zig");
const implicit = @import("../solve/implicit.zig");
const tubes = @import("../testing/tubes.zig");

const cfg = tubes.cfg_rad;
const SimT = sim_mod.Sim(cfg);
const L = SimT.Layout;
const NV = SimT.nv;
const Grid = grid_mod.Grid;

fn tubeBc(
    ctx: ?*const anyopaque,
    s: *const SimT,
    ix: i64,
    iy: i64,
    iz: i64,
    t: f64,
    ifinit: bool,
    face: sim_mod.BcFace,
) relele.Error![NV]f64 {
    _ = s;
    _ = iy;
    _ = iz;
    _ = t;
    _ = ifinit;
    _ = face;
    const tube: *const tubes.Tube = @ptrCast(@alignCast(ctx.?));
    return tubes.ppSide(cfg, if (ix < 0) tube.left else tube.right, tube.gam);
}

/// analytic total fluxes (ρu^x, T^{xx}+R^{xx}, T^{tx}+R^{tx}) from prims
fn totalFluxes(pp: [NV]f64, geom: anytype, gam: f64) ![3]f64 {
    const tij = try hydro.calcTij(cfg, pp, geom, gam);
    const rij = try radiation.calcRij(cfg, pp, geom);
    const u = try relele.uconUcovFromPrims(
        .{ pp[L.index(.vx)], pp[L.index(.vy)], pp[L.index(.vz)] },
        geom,
    );
    const rhoux = pp[L.index(.rho)] * u.con[1];
    return .{ rhoux, tij[1][1] + rij[1][1], tij[0][1] + rij[0][1] };
}

const Metrics = struct {
    stat: f64, // max profile change over the last 15% of the run
    plat_l: f64, // worst plateau deviation, left
    plat_r: f64, // worst plateau deviation, right
    flux: f64, // worst analytic-flux deviation (shock cells excluded)
    ode: f64, // L1(∂x R^{xν} + G^ν) / L1(G^ν)
};

fn runTube(a: std.mem.Allocator, tube: *const tubes.Tube, nx: usize, t_end: f64) !Metrics {
    // choices.h's default UURHORATIOMAX = 1e2 would chop the Γ = 2 tubes'
    // downstream u/ρ ≈ 290 — the tubes need wide hydro ratio floors
    var floors = invert.FloorParams.cdefault;
    floors.uurhoratiomin = 1.0e-30;
    floors.uurhoratiomax = 1.0e30;
    const g = Grid.init(.{ .nx = nx, .ng = 3, .minx = -20, .maxx = 20 });
    var s = try SimT.init(a, g, .{
        .coords = .mink,
        .gam = tube.gam,
        .floors = floors,
        .rad = invert_rad.RadParams.cdefault,
        .opac = tube.params(),
        .implicit = implicit.ImplicitParams.puffy,
        .bc_x = .specific,
        .specific_bc = &tubeBc,
        .bc_ctx = tube, // which tube the BC samples (was a module-level var)
    });
    defer s.deinit();

    var ix: i64 = 0;
    while (ix < s.nxi()) : (ix += 1) {
        const side = if (g.xc(ix) < 0) tube.left else tube.right;
        try s.initCell(ix, 0, 0, tubes.ppSide(cfg, side, tube.gam));
    }
    try s.finishInit();

    // Evolve to t_end at CFL dt, then POLISH at dt/5: the RK2IMEX
    // stationary fixed point evaluates F and G at intermediate states
    // offset by γ·dt·G, so the pointwise flux/source balance carries an
    // O(γ·CFL) splitting bias that refining dt removes — the discrete
    // fixed point converges to the stationary ODE solution. The
    // stationarity snapshot is taken mid-polish, after the re-convergence
    // transient.
    var steps: usize = 0;
    while (s.t < t_end and steps < 50000) : (steps += 1) try s.step(null);
    var tp = s.t + 0.05 * t_end;
    while (s.t < tp and steps < 80000) : (steps += 1) try s.step(0.2 / s.tstepdenmax);

    const snap = try a.alloc(f64, nx * NV);
    defer a.free(snap);
    ix = 0;
    while (ix < s.nxi()) : (ix += 1) {
        var pp: [NV]f64 = undefined;
        s.p.load(ix, 0, 0, &pp);
        @memcpy(snap[@as(usize, @intCast(ix)) * NV ..][0..NV], &pp);
    }

    tp = s.t + 0.05 * t_end;
    while (s.t < tp and steps < 80000) : (steps += 1) try s.step(0.2 / s.tstepdenmax);

    // per-variable scales for normalization
    var scale: [NV]f64 = @splat(1e-30);
    ix = 0;
    while (ix < s.nxi()) : (ix += 1) {
        var pp: [NV]f64 = undefined;
        s.p.load(ix, 0, 0, &pp);
        for (0..NV) |iv| scale[iv] = @max(scale[iv], @abs(pp[iv]));
    }

    var m = Metrics{ .stat = 0, .plat_l = 0, .plat_r = 0, .flux = 0, .ode = 0 };
    var stat_ix: i64 = 0;
    var stat_iv: usize = 0;

    // locate the steepest gas-density jump (the embedded shock of tubes
    // 1/2). The table states are 3-digit-rounded, so the tiny residual flux
    // imbalance makes the jump drift slowly through the grid — cells at the
    // jump are excluded from the pointwise gates (Farris/Sądowski shift
    // profiles before comparing for the same reason).
    var ish: i64 = 0;
    var dmax: f64 = 0;
    ix = 1;
    while (ix < s.nxi()) : (ix += 1) {
        const d = @abs(s.p.get(L.index(.rho), ix, 0, 0) - s.p.get(L.index(.rho), ix - 1, 0, 0));
        if (d > dmax) {
            dmax = d;
            ish = ix;
        }
    }

    // (a) stationarity, gradient-masked: the whole structure translates
    // with the slow drift, so any cell with a local gradient changes by
    // drift·gradient without the profile evolving. Cells whose local
    // cell-to-cell variation exceeds 5% of the row scale — and the ±8-cell
    // neighborhood of the steepest jump (slowly-moving captured fronts
    // breathe at low amplitude in FV schemes) — are excluded; the
    // structured region is validated translation-invariantly by the
    // flux/ODE gates instead. Transients or ringing in the gentle regions
    // (most of the domain) still trip this.
    ix = 1;
    while (ix < s.nxi() - 1) : (ix += 1) {
        if (@abs(ix - ish) <= 8) continue;
        var pp: [NV]f64 = undefined;
        var ppm: [NV]f64 = undefined;
        var ppp: [NV]f64 = undefined;
        s.p.load(ix, 0, 0, &pp);
        s.p.load(ix - 1, 0, 0, &ppm);
        s.p.load(ix + 1, 0, 0, &ppp);
        for (0..NV) |iv| {
            const grad = @max(@abs(pp[iv] - ppm[iv]), @abs(ppp[iv] - pp[iv])) / scale[iv];
            if (grad > 0.05) continue;
            const d = @abs(pp[iv] - snap[@as(usize, @intCast(ix)) * NV + iv]) / scale[iv];
            if (d > m.stat) {
                m.stat = d;
                stat_ix = ix;
                stat_iv = iv;
            }
        }
    }

    // (b) plateaus over the 5 cells nearest each boundary. Rad-velocity
    // rows are excluded: the table does not specify them, and the
    // precursor flux only decays to zero as x → ±∞.
    const pl = tubes.ppSide(cfg, tube.left, tube.gam);
    const pr = tubes.ppSide(cfg, tube.right, tube.gam);
    const fx0 = L.index(.fx);
    const fz0 = L.index(.fz);
    for (0..5) |i| {
        var pp: [NV]f64 = undefined;
        s.p.load(@intCast(i), 0, 0, &pp);
        for (0..NV) |iv| {
            if (iv >= fx0 and iv <= fz0) continue;
            m.plat_l = @max(m.plat_l, @abs(pp[iv] - pl[iv]) / scale[iv]);
        }
        s.p.load(@as(i64, @intCast(nx - 1 - i)), 0, 0, &pp);
        for (0..NV) |iv| {
            if (iv >= fx0 and iv <= fz0) continue;
            m.plat_r = @max(m.plat_r, @abs(pp[iv] - pr[iv]) / scale[iv]);
        }
    }

    // (c) analytic total-flux constancy (vs the left state), excluding
    // ±8 cells around the steepest jump (LAXF-smeared intermediate states,
    // widened by the drift)
    const geom_l = s.cache.fillGeometry(0, 0, 0);
    const f_l = try totalFluxes(pl, &geom_l, tube.gam);
    var flux_ix: i64 = 0;
    ix = 0;
    while (ix < s.nxi()) : (ix += 1) {
        if (@abs(ix - ish) <= 8) continue;
        var pp: [NV]f64 = undefined;
        s.p.load(ix, 0, 0, &pp);
        const geom = s.cache.fillGeometry(ix, 0, 0);
        const f = try totalFluxes(pp, &geom, tube.gam);
        for (0..3) |c| {
            const d = @abs(f[c] - f_l[c]) / @abs(f_l[c]);
            if (d > m.flux) {
                m.flux = d;
                flux_ix = ix;
            }
        }
    }

    // (d) stationary rad ODE residual ∂x R^{xν} + G^ν (smooth tubes),
    // restricted to the exchanging core: cells with |G^ν| ≥ 5% of its
    // maximum. Outside it (the precursor foot) the exponentially small
    // tail is shaped by the scheme's O(dx) dissipation rather than κ, and
    // the ±8-cell jump window is excluded like everywhere else (the gas
    // transition can be far sharper than the radiative mean free path).
    // Even so the balance only closes after the dt-polish above — at CFL
    // dt the IMEX splitting bias dominates it.
    if (tube.smooth) {
        const p = tube.params();
        var gmax: [2]f64 = @splat(0);
        var res: [2]f64 = @splat(0);
        var gsum: [2]f64 = @splat(0);
        for (0..2) |pass| {
            ix = 2;
            while (ix < s.nxi() - 2) : (ix += 1) {
                if (@abs(ix - ish) <= 8) continue;
                var ppm: [NV]f64 = undefined;
                var pp0: [NV]f64 = undefined;
                var ppp: [NV]f64 = undefined;
                s.p.load(ix - 1, 0, 0, &ppm);
                s.p.load(ix, 0, 0, &pp0);
                s.p.load(ix + 1, 0, 0, &ppp);
                const geom_m = s.cache.fillGeometry(ix - 1, 0, 0);
                const geom_0 = s.cache.fillGeometry(ix, 0, 0);
                const geom_p = s.cache.fillGeometry(ix + 1, 0, 0);
                const rm = try radiation.calcRij(cfg, ppm, &geom_m);
                const rp = try radiation.calcRij(cfg, ppp, &geom_p);
                const gi = try radforce.calcGi(cfg, pp0, &geom_0, tube.gam, &p);
                const dx2 = g.xc(ix + 1) - g.xc(ix - 1);
                // ν = t (index 0) and ν = x (index 1)
                for (0..2) |nu| {
                    const ag = @abs(gi.lab[nu]);
                    if (pass == 0) {
                        gmax[nu] = @max(gmax[nu], ag);
                    } else if (ag >= 0.05 * gmax[nu]) {
                        const drdx = (rp[1][nu] - rm[1][nu]) / dx2;
                        res[nu] += @abs(drdx + gi.lab[nu]);
                        gsum[nu] += ag;
                    }
                }
            }
        }
        m.ode = @max(res[0] / gsum[0], res[1] / gsum[1]);
    }

    std.debug.print(
        "tube {s} (nx={d}, t={d:.0}, {d} steps): stat {e:.2}, plat L/R {e:.2}/{e:.2}, flux {e:.2}, ode {e:.2}\n",
        .{ tube.name, nx, s.t, steps, m.stat, m.plat_l, m.plat_r, m.flux, m.ode },
    );
    if (std.c.getenv("KORAL_TUBE_DEBUG") != null) {
        inline for ([_]layout.VarTag{ .rho, .uu, .vx, .ee, .fx }) |tag| {
            const iv = comptime L.index(tag);
            std.debug.print("  {s}: L={e:.4} [{e:.4} {e:.4}] .. [{e:.4} {e:.4}] R={e:.4}\n", .{
                @tagName(tag),                 pl[iv],
                s.p.get(iv, 0, 0, 0),          s.p.get(iv, 2, 0, 0),
                s.p.get(iv, @as(i64, @intCast(nx - 3)), 0, 0), s.p.get(iv, @as(i64, @intCast(nx - 1)), 0, 0),
                pr[iv],
            });
        }
        std.debug.print("  shock at ix={d} (x={d:.2}); worst flux at ix={d} (x={d:.2}); worst stat at ix={d} iv={d}\n", .{ ish, g.xc(ish), flux_ix, g.xc(flux_ix), stat_ix, stat_iv });
        var mn: [NV]f64 = @splat(1e300);
        var mx: [NV]f64 = @splat(-1e300);
        ix = 0;
        while (ix < s.nxi()) : (ix += 1) {
            var ppd: [NV]f64 = undefined;
            s.p.load(ix, 0, 0, &ppd);
            for (0..NV) |iv| {
                mn[iv] = @min(mn[iv], ppd[iv]);
                mx[iv] = @max(mx[iv], ppd[iv]);
            }
        }
        for (0..NV) |iv| std.debug.print("  iv={d}: [{e:.2}, {e:.2}]\n", .{ iv, mn[iv], mx[iv] });
    }
    return m;
}

test "radiative shock tube battery (tubes 3a, 4a)" {
    const a = std.testing.allocator;

    // Run both tubes first, assert at the end — a failed 3a gate must not
    // hide 4a (the more expensive tube, and the only default-suite
    // ODE-residual gate). Mirrors the slow set below.
    const m3a = try runTube(a, &tubes.tubes[2], 192, 60.0);
    const m4a = try runTube(a, &tubes.tubes[4], 128, 300.0);

    var ok = true;
    // tube 3a: highly relativistic wave (γ ≈ 10). "Smooth" analytically,
    // but the downstream mean free path 1/(κρ) ≈ 0.4 is ~2 cells here:
    // effectively a captured front that breathes at low amplitude (the
    // classic slowly-moving-shock behavior) — integral gates only
    // (measured after polish: stat 9.0e-3, plat 1.1e-5/2.7e-3, flux 4.7e-2)
    if (!(m3a.stat <= 2.0e-2 and m3a.plat_l <= 2.5e-2 and m3a.plat_r <= 2.5e-2 and m3a.flux <= 8.0e-2)) {
        std.debug.print("tube 3a FAILED its gates\n", .{});
        ok = false;
    }
    // tube 4a: radiation-pressure-dominated smooth wave — resolved
    // (mfp ≈ 11 cells), so the pointwise stationary-ODE gate applies too.
    // Its measured core residual 0.32 is the IMEX splitting bias at polish
    // dt/5 (resolution-independent: 0.29 at nx = 384) — a sign/factor
    // error in G or R^{μν} would put it at ≥ 1. (measured: stat 7.6e-3,
    // plat 4.8e-8/1.4e-3, flux 3.8e-3, ode 0.32)
    if (!(m4a.stat <= 1.5e-2 and m4a.plat_l <= 2.5e-2 and m4a.plat_r <= 2.5e-2 and m4a.flux <= 2.0e-2 and m4a.ode <= 4.0e-1)) {
        std.debug.print("tube 4a FAILED its gates\n", .{});
        ok = false;
    }
    try std.testing.expect(ok);
}

test "radiative shock tube battery, slow set (tubes 1, 2, 3b, 4b, hi-res)" {
    // zig build test -Dslow-tests
    if (!@import("build_options").slow_tests) return error.SkipZigTest;
    const a = std.testing.allocator;

    // Run everything first, assert at the end — a single failed gate must
    // not hide the other tubes' measurements. The shocked/slow tubes'
    // stationarity gates are loose: their diffusive precursor feet keep
    // deepening long after the structure and plateaus are converged
    // (tube 1: until t ≳ 1e4 at v ~ 0.015) — the plateau + flux gates
    // carry the convergence statement.
    const Case = struct {
        tube: usize,
        nx: usize,
        t_end: f64,
        stat: f64,
        plat: f64,
        flux: f64,
        ode: f64, // 0 = not gated
    };
    const cases = [_]Case{
        // tube 2: contact residue of the rounded right state flushes at
        // v ≈ 0.08; the fx precursor foot still deepens slowly at t = 700
        // (measured: stat 3.3e-2, plat 7.0e-4/2.5e-3, flux 3.2e-2)
        .{ .tube = 1, .nx = 128, .t_end = 700.0, .stat = 5.0e-2, .plat = 2.5e-2, .flux = 5.0e-2, .ode = 0 },
        // tube 3a hi-res: the thinner captured front breathes HARDER at
        // nx = 384 (measured: stat 5.0e-2, plat 1.1e-5/3.4e-3, flux 4.2e-2)
        .{ .tube = 2, .nx = 384, .t_end = 60.0, .stat = 7.0e-2, .plat = 2.5e-2, .flux = 6.0e-2, .ode = 0 },
        // tube 4a hi-res (measured: stat 2.6e-3, plat 7.7e-8/1.5e-3, flux
        // 3.5e-3, ode 2.9e-1 — the ODE residual does NOT shrink with dx:
        // it is the dt-proportional IMEX splitting bias at polish dt/5)
        .{ .tube = 4, .nx = 384, .t_end = 300.0, .stat = 1.5e-2, .plat = 2.5e-2, .flux = 2.0e-2, .ode = 4.0e-1 },
        // tube 4b (measured: stat 2.6e-3, plat 4.7e-8/1.4e-3, flux 5.6e-3)
        .{ .tube = 5, .nx = 192, .t_end = 120.0, .stat = 3.0e-2, .plat = 2.5e-2, .flux = 1.0e-1, .ode = 0 },
        // tube 1 (longest): non-relativistic strong shock; the diffusive
        // precursor foot deepens until t ≳ 1e4 at v ~ 0.015 — plateau +
        // flux gates carry the convergence statement (measured: stat
        // 5.2e-2, plat 1.4e-4/5.9e-3, flux 4.8e-2)
        .{ .tube = 0, .nx = 96, .t_end = 4000.0, .stat = 1.0e-1, .plat = 2.5e-2, .flux = 1.0e-1, .ode = 0 },
        // tube 3b (κ = 25) is EXCLUDED: at test-affordable resolutions the
        // initial discontinuity carries κρΔx ≈ 40 per cell and the run
        // collapses — in production C KORAL identically (verified by
        // rebuilding the C oracle with tube-3b parameters: Ê 1140 → 1e2,
        // Tgas → 1e15 K with C's own "UNPHYSICALLY HIGH TEMPERATURE!"
        // diagnostics, γ_rad pinned at GAMMAMAXRAD). The κ-stiffness
        // regime is covered by the LTE pulse (τ_cell ≈ 16) and tube 4b.





    };

    var ms: [cases.len]Metrics = undefined;
    for (&cases, 0..) |*c, i| {
        ms[i] = try runTube(a, &tubes.tubes[c.tube], c.nx, c.t_end);
    }
    var ok = true;
    for (&cases, 0..) |*c, i| {
        const m = &ms[i];
        const pass = m.stat <= c.stat and m.plat_l <= c.plat and m.plat_r <= c.plat and
            m.flux <= c.flux and (c.ode == 0 or m.ode <= c.ode);
        if (!pass) {
            std.debug.print("slow tube {s} FAILED its gates\n", .{tubes.tubes[c.tube].name});
            ok = false;
        }
    }
    try std.testing.expect(ok);
}
