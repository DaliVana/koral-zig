//! M11 keystone: the PUFFY t=0 initial state, cell-by-cell against the C
//! oracle (oracle/harness_init.c → tests/golden/init/*.kini.gz).
//!
//! The full ko.c:140-263 init sequence runs on the production 384×360 MKS2
//! grid, compared at three stages over the domain + ghost layers:
//!   snapshot 0 (puffy_t0_A)      — A_φ in the B slots after set_bc
//!   snapshot 1 (puffy_t0_p)      — all 13 primitives after calc_BfromA +
//!                                  set_bc (THE keystone)
//!   snapshot 2 (puffy_t0_pfinal) — β-normalized B slots after postinit
//!
//! What the numbers say (field-scale = max|C−zig| / max|C| over the class):
//!
//!  * the velocities and radiation-frame velocities (VX..VZ, FX..FZ) come
//!    from the analytic angular-momentum profile ℓ(λ) and the metric — no
//!    integral — and match to ~1e-13 (machine). In an LTE torus F̂ = 0, so
//!    prad_ff2lab returns the gas velocity, keeping FX..FZ analytic too.
//!  * ρ/u/ENTR/Ê and the derived poloidal B (B1,B2, via A_φ ∝ ρ) inherit
//!    the ln f quadrature. Against production C's gsl_qags at epsrel 1e-8
//!    the field-scale deviation is ~1e-3 — NOT machine, and NOT Zig's
//!    error: C's qags is only ~1e-3-accurate near the ℓ(λ)-break kink
//!    (its error estimate is unreliable at a C¹ discontinuity). Zig's GK21
//!    matches an independent tanh-sinh to 1e-10 (puffy_tests.zig), so Zig
//!    is the accurate side. The puffy_t0_eps12 golden (C re-run at epsrel
//!    1e-12) collapses that torus deviation to ~5e-7, pinning the
//!    quadrature as the sole source. B3 is the toroidal field: zero in 2D
//!    axisymmetry (poloidal A_φ only), so it is checked as absolute ~0.
//!
//! Cost: the Zig init runs the limotorus quadrature per cell over the full
//! grid, so the cell-by-cell comparison + attribution are behind
//! -Dslow-tests; the default suite runs the pipeline on a coarse grid as a
//! smoke check (the theory gates in puffy_tests.zig carry the default-suite
//! correctness statement: β ≡ 1/20, divB = 0, the bc.c contract, GK21 vs
//! tanh-sinh).

const std = @import("std");
const build_options = @import("build_options");
const config = @import("../../config.zig");
const grid_mod = @import("../../grid.zig");
const sim_mod = @import("../../sim.zig");
const golden = @import("../../testing/golden.zig");
const puffy = @import("../../problems/puffy/puffy.zig");
const ct = @import("../../magn/ct.zig");
const invert = @import("../../solve/invert.zig");
const invert_rad = @import("../../solve/invert_rad.zig");

const Grid = grid_mod.Grid;
const SimP = sim_mod.Sim(config.puffy);
const LP = SimP.Layout;

/// Field-scale deviation: max|c−z| over a class of cells, normalized by the
/// class field magnitude max|c|. The physically meaningful keystone metric —
/// a per-cell *relative* deviation would amplify the razor-thin torus
/// surface (ρ ∝ ε³ → 0) by 3/ε and drown the field-scale agreement.
const FieldDev = golden.FieldDev;

const var_names = [_][]const u8{ "rho", "uu", "vx", "vy", "vz", "entr", "b1", "b2", "b3", "ee", "fx", "fy", "fz" };

// classification of the 13 primitives by how they agree with C at t=0
const kinematic = [_]usize{ 2, 3, 4, 10, 11, 12 }; // vx vy vz fx fy fz — analytic
const quadrature = [_]usize{ 0, 1, 5, 6, 7, 9 }; // rho uu entr b1 b2 ee — via ln f integral
const toroidal = 8; // b3 — zero in 2D axisymmetry

fn puffyOptions() SimP.Options {
    return .{
        .coords = .mks2,
        .mp = puffy.mp,
        .gam = puffy.gam,
        .floors = invert.FloorParams.puffy,
        .rad = invert_rad.RadParams.puffy,
        .correct_polaraxis = true,
        .nccorrectpolar = 2,
        .bc_x = .specific,
        .bc_y = .specific,
        .specific_bc = &puffy.Bc(SimP).calc,
    };
}

/// Compare the B slots (B1,B2,B3) of the current state against a 3-variable
/// KINI snapshot. `is_field[i]` picks whether slot i is a real field
/// (→ field-scale ≤ tol) or an identically-zero component (→ absolute ≤
/// 1e-20). The vector potential is A_φ only, so snapshot 0 has field mask
/// {f,f,t}; the poloidal B field with a zero toroidal part is {t,t,f}.
fn compareBslots(s: *SimP, k: *const golden.Kini, tol: f64, is_field: [3]bool, label: []const u8) !void {
    var dev = [_]FieldDev{ .{}, .{}, .{} };
    var absdiff = [_]f64{ 0, 0, 0 };
    const bslots = [_]usize{ LP.index(.b1), LP.index(.b2), LP.index(.b3) };
    var jy: usize = 0;
    while (jy < k.nSamplesY()) : (jy += 1) {
        var jx: usize = 0;
        while (jx < k.nSamplesX()) : (jx += 1) {
            const ix = k.cellX(jx);
            const iy = k.cellY(jy);
            const base = k.base(jx, jy, 0);
            var pp: [SimP.nv]f64 = undefined;
            s.p.load(ix, iy, 0, &pp);
            for (0..3) |ib| {
                dev[ib].addLoc(k.data[base + ib], pp[bslots[ib]], ix, iy);
                absdiff[ib] = @max(absdiff[ib], @abs(k.data[base + ib] - pp[bslots[ib]]));
            }
        }
    }
    std.debug.print("{s}:\n", .{label});
    for (0..3) |ib| {
        dev[ib].report(var_names[bslots[ib]]);
        if (is_field[ib])
            try std.testing.expect(dev[ib].dev() <= tol)
        else
            try std.testing.expect(absdiff[ib] <= 1e-20);
    }
}

/// The full keystone: run the init sequence, comparing every snapshot.
fn runKeystone(a: std.mem.Allocator) !void {
    var s = try SimP.init(a, puffy.makeGrid(384, 360), puffyOptions());
    defer s.deinit();

    // -- stage 0: prepinit + init + set_bc → A_φ in the B slots -----------
    try puffy.prepInitDomain(SimP, &s);
    try s.setBc(0.0, true);
    {
        var ka = try golden.readKini(a, "init/puffy_t0_A.kini.gz");
        defer ka.deinit();
        // A_φ = f(ρ) near its (X²−0.02) loop-edge cutoff amplifies the
        // torus quadrature; field-scale ~2e-3 (C-qags-limited)
        try compareBslots(&s, &ka, 5e-3, .{ false, false, true }, "snapshot 0 (A_φ in B slots)");
    }

    // -- stage 1: calc_BfromA + set_bc → all primitives (keystone) --------
    try ct.calcBfromA(SimP, &s, true);
    try s.setBc(0.0, true);
    {
        var kp = try golden.readKini(a, "init/puffy_t0_p.kini.gz");
        defer kp.deinit();
        var dev: [13]FieldDev = undefined;
        for (0..13) |iv| dev[iv] = .{};
        var b3_absdiff: f64 = 0;
        var jy: usize = 0;
        while (jy < kp.nSamplesY()) : (jy += 1) {
            var jx: usize = 0;
            while (jx < kp.nSamplesX()) : (jx += 1) {
                const ix = kp.cellX(jx);
                const iy = kp.cellY(jy);
                const base = kp.base(jx, jy, 0);
                var pp: [SimP.nv]f64 = undefined;
                s.p.load(ix, iy, 0, &pp);
                for (0..13) |iv| dev[iv].addLoc(kp.data[base + iv], pp[iv], ix, iy);
                b3_absdiff = @max(b3_absdiff, @abs(kp.data[base + toroidal] - pp[toroidal]));
            }
        }
        std.debug.print("snapshot 1 (all primitives, keystone):\n", .{});
        for (0..13) |iv| dev[iv].report(var_names[iv]);

        // analytic kinematics — machine precision
        for (kinematic) |iv| try std.testing.expect(dev[iv].dev() <= 1e-11);
        // quadrature-inherited — C's production qags-1e-8 kink error
        // (the eps12 attribution test confirms Zig is the accurate side)
        for (quadrature) |iv| try std.testing.expect(dev[iv].dev() <= 3e-3);
        // toroidal B3 ~ 0 in 2D axisymmetry
        try std.testing.expect(b3_absdiff <= 1e-20);
    }

    // -- stage 2: postinit → β-normalized B -------------------------------
    _ = try puffy.postinit(SimP, &s);
    {
        var kf = try golden.readKini(a, "init/puffy_t0_pfinal.kini.gz");
        defer kf.deinit();
        // domain B scaled by fac (β = 1/20); fac inherits the torus
        // quadrature through the peak-β cell, so the scaled B carries it too
        try compareBslots(&s, &kf, 3e-3, .{ true, true, false }, "snapshot 2 (β-normalized B)");
    }
}

test "M11 keystone: PUFFY t=0 pipeline runs (default smoke; full diff is slow)" {
    if (build_options.slow_tests) return error.SkipZigTest; // slow test covers it
    // the theory gates in puffy_tests.zig carry the default-suite
    // correctness statement; here just confirm the full pipeline runs and
    // produces a finite β-normalization on a coarse grid
    const a = std.testing.allocator;
    var s = try SimP.init(a, puffy.makeGrid(96, 90), puffyOptions());
    defer s.deinit();
    const fac = try puffy.initAll(SimP, &s);
    try std.testing.expect(fac > 0 and std.math.isFinite(fac));
}

test "M11 keystone: PUFFY t=0 vs C (full 384×360 cell-by-cell)" {
    if (!build_options.slow_tests) return error.SkipZigTest;
    try runKeystone(std.testing.allocator);
}

test "M11 keystone: qags attribution — epsrel 1e-12 collapses torus deviation" {
    if (!build_options.slow_tests) return error.SkipZigTest;
    const a = std.testing.allocator;
    var ke = try golden.readKini(a, "init/puffy_t0_eps12.kini.gz");
    defer ke.deinit();
    // eps12 is snapshot-1-equivalent (post-BfromA, pre-postinit): the
    // full-grid Zig init up to that stage vs the C run with qags epsrel
    // 1e-12 on a stride-4 domain sample. The torus ρ/u field-scale
    // deviation must drop far below the ~1e-3 seen against the default
    // epsrel-1e-8 golden — pinning the quadrature as the source.
    var s = try SimP.init(a, puffy.makeGrid(384, 360), puffyOptions());
    defer s.deinit();
    try puffy.prepInitDomain(SimP, &s);
    try s.setBc(0.0, true);
    try ct.calcBfromA(SimP, &s, true);
    try s.setBc(0.0, true);

    var dev_rho: FieldDev = .{};
    var dev_uu: FieldDev = .{};
    var jy: usize = 0;
    while (jy < ke.nSamplesY()) : (jy += 1) {
        var jx: usize = 0;
        while (jx < ke.nSamplesX()) : (jx += 1) {
            const ix = ke.cellX(jx);
            const iy = ke.cellY(jy);
            const base = ke.base(jx, jy, 0);
            var pp: [SimP.nv]f64 = undefined;
            s.p.load(ix, iy, 0, &pp);
            if (ke.data[base + LP.index(.rho)] > 1e-22) { // torus cells
                dev_rho.addLoc(ke.data[base + LP.index(.rho)], pp[LP.index(.rho)], ix, iy);
                dev_uu.addLoc(ke.data[base + LP.index(.uu)], pp[LP.index(.uu)], ix, iy);
            }
        }
    }
    std.debug.print("eps12 attribution (torus cells, field-scale):\n", .{});
    dev_rho.report("rho");
    dev_uu.report("uu");
    // ~5e-7 measured (near-surface ρ ∝ ε³ floor) ≪ the ~1e-3 default
    try std.testing.expect(dev_rho.dev() <= 1e-6);
    try std.testing.expect(dev_uu.dev() <= 1e-6);
}
