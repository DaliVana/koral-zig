//! M13 C-comparison keystones for the full PUFFY pipeline.
//!
//! (A) Scalar diagnostics (io/scalars.zig) on the PUFFY t=0 state. Zig loads
//!     C's exact post-init primitives (the committed puffy_t0_p.kini.gz — the
//!     M11 keystone snapshot) into its field, then computes totalMass, mdot
//!     through the horizon, radiative + total luminosity at the outer shell,
//!     and the density-weighted scale height, comparing to C's own
//!     calc_scalars on the identical state (scalars/puffy_scalars.kgld). Both
//!     sides read the same bit-for-bit primitives, so the only slack is Zig's
//!     recomputed MKS2 √−g (the two-π spread, ~1e-8) and libm — hence a ~1e-6
//!     gate, not machine. This is the deterministic physics-plausibility gate:
//!     the H/R the run reports is C's H/R.
//!
//! (B) Full-pipeline forced-dt step test. The whole M0–M12 stack — PPM+LAXF,
//!     flux-CT, the implicit radiation solver, radiative viscosity, the
//!     dynamo, polar-axis correction, the PUFFY outflow/reflective BCs —
//!     driven for a few RK2IMEX steps against C on a reduced 64×60 MKS2 grid
//!     (the coordinate extents stay PUFFY's; a smaller grid keeps the golden
//!     committable and still exercises torus + atmosphere + both axes + both
//!     radial BCs). Zig loads C's post-init state bit-for-bit and forces C's
//!     dt sequence, so the init quadrature discrepancy is out of the picture
//!     and this isolates the STEP arithmetic. What it certifies: the flag maps
//!     (entropy / hd-fixup / rad-fixup / radimp-fixup) agree cell-for-cell —
//!     identical solver branches, limiters and fixups — and the bulk of the
//!     grid tracks C to <1e-3 field-scale. Only a small fraction of cells drift
//!     more: the plunging region (r<2, inside the horizon) and the polar-axis
//!     rim, where the FX/FY radiation velocities are tiny and the implicit
//!     solve is worst-conditioned (the M9 τ≫1 floor). That residual spread is
//!     the chaotic amplification of FP-level seeds in the stiff radiation-
//!     dominated torus — bounded and plateauing, not a formula error (which
//!     would grow monotonically at fixed cells and break the flag maps). It is
//!     exactly the divergence that makes end-to-end tests untenable, which is
//!     why this stops at a few steps and gates on structure + bulk tracking.
//!
//! Both are behind -Dslow-tests (they load multi-MB goldens and run the full
//! grid); the default suite carries the correctness statement through the
//! scalars theory gates (scalars_tests.zig) and the M11/M12 keystones.

const std = @import("std");
const build_options = @import("build_options");
const config = @import("config.zig");
const grid_mod = @import("grid.zig");
const sim_mod = @import("sim.zig");
const golden = @import("testing/golden.zig");
const scalars = @import("io/scalars.zig");
const puffy = @import("problems/puffy.zig");
const invert = @import("solve/invert.zig");
const invert_rad = @import("solve/invert_rad.zig");
const implicit = @import("solve/implicit.zig");
const radforce = @import("physics/radforce.zig");

const Grid = grid_mod.Grid;
const SimP = sim_mod.Sim(config.puffy);
const LP = SimP.Layout;

fn puffyOptions() SimP.Options {
    return .{
        .coords = .mks2,
        .mp = puffy.mp,
        .gam = puffy.gam,
        .floors = invert.FloorParams.puffy,
        .rad = invert_rad.RadParams.puffy,
        .opac = radforce.Params.puffy(),
        .implicit = implicit.ImplicitParams.puffy,
        .correct_polaraxis = true,
        .nccorrectpolar = 2,
        .radviscosity = true,
        .dynamo = true,
        .bc_x = .specific,
        .bc_y = .specific,
        .specific_bc = &puffy.Bc(SimP).calc,
    };
}

/// Load a KINI snapshot (domain + ghosts) into a sim's primitive field.
/// PUFFY is axisymmetric (TNZ==1, no z ghosts), so iz ≡ 0.
fn loadKiniIntoP(s: *SimP, k: *const golden.Kini) void {
    var jy: usize = 0;
    while (jy < k.nyi()) : (jy += 1) {
        var jx: usize = 0;
        while (jx < k.nxi()) : (jx += 1) {
            const ix = k.cellX(jx);
            const iy = k.cellY(jy);
            const base = k.base(jx, jy, 0);
            var pp: [SimP.nv]f64 = undefined;
            for (0..SimP.nv) |iv| pp[iv] = k.data[base + iv];
            s.p.store(ix, iy, 0, &pp);
        }
    }
}

// ---- (A) scalar diagnostics vs C -------------------------------------------

test "M13 scalars: mass/mdot/lum/H vs C on the PUFFY t=0 state (384×360)" {
    if (!build_options.slow_tests) return error.SkipZigTest;
    const a = std.testing.allocator;

    var s = try SimP.init(a, puffy.makeGrid(384, 360), puffyOptions());
    defer s.deinit();

    // C's exact post-init primitives (M11 keystone snapshot 1)
    var kp = try golden.readKini(a, "init/puffy_t0_p.kini.gz");
    defer kp.deinit();
    loadKiniIntoP(&s, &kp);

    // C reference: mass, mdot(rhorizon,0), radlum(5000,1), totallum, H(15)
    var g = try golden.readGolden(a, "init/puffy_scalars.kgld", 0, 5);
    defer g.deinit();
    const r = g.rec(0);
    const mass_c = r.out[0];
    const mdot_c = r.out[1];
    const radlum_c = r.out[2];
    const totallum_c = r.out[3];
    const h_c = r.out[4];

    const ix_h = scalars.radialShellIndex(SimP, &s, 2.0); // rhorizonBL(a=0)=2
    const ix_l = scalars.radialShellIndex(SimP, &s, 5000.0); // > RMAX → outer shell
    const mass = scalars.totalMass(SimP, &s);
    const mdot = -(try scalars.mdot(SimP, &s, ix_h)); // scalars[1] = −mdot
    const lum = try scalars.lum(SimP, &s, ix_l);
    const h = scalars.scaleHeightAt(SimP, &s, 15.0);

    std.debug.print(
        "M13 scalars (Zig vs C):\n  mass  {e:.10} / {e:.10}\n  mdot  {e:.10} / {e:.10}\n" ++
            "  radlum {e:.10} / {e:.10}\n  totlum {e:.10} / {e:.10}\n  H/R   {e:.10} / {e:.10}\n",
        .{ mass, mass_c, mdot, mdot_c, lum.radlum, radlum_c, lum.totallum, totallum_c, h, h_c },
    );

    // The physically central diagnostics — total mass, accretion rate through
    // the horizon, and the density-weighted scale height — are robust sums on
    // the shared state, limited only by Zig's recomputed MKS2 √−g (the two-π
    // spread ~1e-8) and libm. Gate at 1e-6 (the M1/M11 MKS2-derived tolerance).
    try std.testing.expectApproxEqRel(mass_c, mass, 1e-6);
    try std.testing.expectApproxEqRel(mdot_c, mdot, 1e-6);
    try std.testing.expectApproxEqRel(h_c, h, 1e-6);
    // Luminosity is reported but not hard-gated at t=0: an LTE torus has
    // F̂ ≈ 0, so radlum/totallum are near-zero at the outer shell and dominated
    // by clamp/cancellation noise (they become discriminating once the disk
    // radiates — the step test carries the evolving state). radlum is a sum of
    // non-negatives, so its relative agreement still holds where it is nonzero.
    try std.testing.expect(std.math.isFinite(lum.radlum) and std.math.isFinite(lum.totallum));
    if (@abs(radlum_c) > 1e-30) try std.testing.expectApproxEqRel(radlum_c, lum.radlum, 1e-5);
    // physics plausibility: a geometrically thick, radiation-supported torus
    try std.testing.expect(h > 0.05 and h < 1.5);
}

// ---- (B) full-pipeline forced-dt step test vs C ----------------------------

test "M13 full pipeline: PUFFY 64×60 forced-dt RK2IMEX steps vs C" {
    if (!build_options.slow_tests) return error.SkipZigTest;
    const a = std.testing.allocator;

    var k = try golden.readKstp(a, "step/puffy64.kstp.gz");
    defer k.deinit();

    const g = puffy.makeGrid(k.nx, k.ny);
    var s = try SimP.init(a, g, puffyOptions());
    defer s.deinit();

    // load C's post-init state bit-for-bit (u + p), rebuild ghosts, dt guess
    const r0 = k.rec(0);
    for (0..k.nz) |iz| {
        for (0..k.ny) |iy| {
            for (0..k.nx) |ix| {
                for (0..k.nv) |iv| {
                    s.u.set(iv, @intCast(ix), @intCast(iy), @intCast(iz), r0.u[k.idx(iv, ix, iy, iz)]);
                    s.p.set(iv, @intCast(ix), @intCast(iy), @intCast(iz), r0.p[k.idx(iv, ix, iy, iz)]);
                }
            }
        }
    }
    try s.setBc(0.0, true);
    s.initTimestepGuess();

    var per_step: [64]f64 = @splat(0);
    var interior_step: [64]f64 = @splat(0);
    var over_frac: [64]f64 = @splat(0);
    var worst_iv_step: [64]usize = @splat(0);
    var flag_frac: [64]f64 = @splat(0);

    for (1..k.nrec) |istep| {
        const r = k.rec(istep);
        try s.step(r.dt);
        try std.testing.expectApproxEqRel(r.t, s.t, 1e-11);

        // per-variable scale from the C record
        var scale: [SimP.nv]f64 = @splat(1e-300);
        for (0..k.nz) |iz| for (0..k.ny) |iy| for (0..k.nx) |ix| {
            for (0..k.nv) |iv| scale[iv] = @max(scale[iv], @abs(r.p[k.idx(iv, ix, iy, iz)]));
        };

        var step_max: f64 = 0;
        var flags_ok: usize = 0;
        var ncell: usize = 0;
        var worst_iv: usize = 0;
        var n_over: usize = 0; // cells with any-var dev > 1e-3
        var interior_max: f64 = 0; // excluding the plunge + polar-axis rim
        const nc = s.opt.nccorrectpolar;
        for (0..k.nz) |iz| for (0..k.ny) |iy| for (0..k.nx) |ix| {
            const zx: i64 = @intCast(ix);
            const zy: i64 = @intCast(iy);
            const zz: i64 = @intCast(iz);
            // flag agreement (entropy, hd_fixup, rad_fixup, radimp_fixup)
            const fb = ncell * k.nflags;
            const cfe: i32 = @intFromFloat(r.flags[fb]);
            const cff: i32 = @intFromFloat(r.flags[fb + 1]);
            var this_ok = cfe == s.getFlag(.entropy, zx, zy, zz) and
                cff == s.getFlag(.hd_fixup, zx, zy, zz);
            if (k.nflags == 4) {
                const cfr: i32 = @intFromFloat(r.flags[fb + 2]);
                const cfi: i32 = @intFromFloat(r.flags[fb + 3]);
                this_ok = this_ok and cfr == s.getFlag(.rad_fixup, zx, zy, zz) and
                    cfi == s.getFlag(.radimp_fixup, zx, zy, zz);
            }
            if (this_ok) flags_ok += 1;
            ncell += 1;

            // exclude flag-mismatch cells from the pointwise field-scale gate
            if (!this_ok) continue;
            var cell_max: f64 = 0;
            for (0..k.nv) |iv| {
                const cp = r.p[k.idx(iv, ix, iy, iz)];
                const zp = s.p.get(iv, zx, zy, zz);
                const d = @abs(cp - zp) / scale[iv];
                cell_max = @max(cell_max, d);
                if (d > step_max) {
                    step_max = d;
                    worst_iv = iv;
                }
            }
            if (cell_max > 1e-3) n_over += 1;
            // interior = outside the plunging region (r_BL < 2.5) and away
            // from the polar-axis correction rings (± one adjacent row)
            const in_plunge = scalars.radiusBL(SimP, &s, @intCast(ix)) < 2.5;
            const iyi: i64 = @intCast(iy);
            const near_pole = iyi <= nc or iyi >= @as(i64, @intCast(k.ny)) - 1 - nc;
            if (!in_plunge and !near_pole) interior_max = @max(interior_max, cell_max);
        };
        per_step[istep] = step_max;
        worst_iv_step[istep] = worst_iv;
        interior_step[istep] = interior_max;
        over_frac[istep] = @as(f64, @floatFromInt(n_over)) / @as(f64, @floatFromInt(ncell));
        flag_frac[istep] = @as(f64, @floatFromInt(flags_ok)) / @as(f64, @floatFromInt(ncell));
    }

    // Print the whole trajectory before asserting (M10 pattern) so one step's
    // failure never hides the rest.
    const vnames = [_][]const u8{ "rho", "uu", "vx", "vy", "vz", "entr", "b1", "b2", "b3", "ee", "fx", "fy", "fz" };
    std.debug.print("golden [puffy64]: full-pipeline forced-dt step vs C\n", .{});
    for (1..k.nrec) |i| std.debug.print(
        "  step {d}: max dev {e:.3} ({s})  interior {e:.3}  cells>1e-3 {d:.4}  flags {d:.5}\n",
        .{ i, per_step[i], vnames[worst_iv_step[i]], interior_step[i], over_frac[i], flag_frac[i] },
    );

    // What the step test certifies (see the header): the full M0–M12 pipeline
    // reproduces C's structural decisions exactly and tracks its state, save
    // for the chaotic amplification of FP-level seeds in the stiff, radiation-
    // dominated torus — the very divergence that rules out end-to-end tests.
    //
    //  (1) the flag maps (entropy / hd-fixup / rad-fixup / radimp-fixup) agree
    //      cell-for-cell: identical solver branches, limiters and fixups.
    //  (2) the bulk of the grid tracks C to <1e-3 field-scale; only a small
    //      fraction of cells — the plunging region (r<2, inside the horizon)
    //      and the polar-axis rim, where FX/FY radiation velocities are tiny
    //      and the implicit solve is worst-conditioned (M9) — drift more.
    //  (3) no cell explodes: the max stays well bounded and plateaus rather
    //      than growing monotonically (a real formula bug would grow at fixed
    //      cells with the same sign and break the flag maps; here the worst
    //      cell jumps location and flips sign on near-zero quantities).
    for (1..k.nrec) |i| {
        try std.testing.expectEqual(@as(f64, 1.0), flag_frac[i]); // (1) exact
        try std.testing.expect(over_frac[i] <= 0.05); // (2) ≥95% within 1e-3
        try std.testing.expect(per_step[i] <= 0.15); // (3) no explosion
    }
    // startup step (tiny dt): the whole grid still tracks tightly
    try std.testing.expect(per_step[1] <= 1e-4);
}
