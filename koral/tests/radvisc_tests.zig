//! M12 theory gates for radiative shear viscosity that the C golden does not
//! reach: the RADVISCNUDAMP diffusion cap (calcRadViscCoeff) and the
//! RADVISCMAXVELDAMP characteristic-velocity cap (addRadViscFlux). The shear
//! tensor, the coefficient, and the assembled R^i_j are validated numerically
//! against C in golden_visc_test.zig; here we pin the two limiter properties.

const std = @import("std");
const config = @import("../config.zig");
const sim_mod = @import("../sim.zig");
const puffy = @import("../problems/puffy/puffy.zig");
const radvisc = @import("../physics/radvisc.zig");
const invert = @import("../solve/invert.zig");
const invert_rad = @import("../solve/invert_rad.zig");
const radforce = @import("../physics/radforce.zig");
const relele = @import("../relele.zig");

const SimP = sim_mod.Sim(config.puffy);
const LP = SimP.Layout;

fn viscOptions() SimP.Options {
    return .{
        .coords = .mks2,
        .mp = puffy.mp,
        .gam = puffy.gam,
        .floors = invert.FloorParams.puffy,
        .rad = invert_rad.RadParams.puffy,
        .opac = radforce.Params.puffy(),
        .correct_polaraxis = true,
        .nccorrectpolar = 2,
        .radviscosity = true,
        .bc_x = .specific,
        .bc_y = .specific,
        .specific_bc = &puffy.Bc(SimP).calc,
    };
}

/// Find a torus cell (ρ above the atmosphere) at a mid radius.
fn torusCell(s: *const SimP) ?[2]i64 {
    const nx = s.nxi();
    const ny = s.nyi();
    var iy: i64 = @divTrunc(ny, 2);
    while (iy < ny) : (iy += 1) {
        var ix: i64 = @divTrunc(nx, 3);
        while (ix < nx) : (ix += 1) {
            if (s.p.get(LP.index(.rho), ix, iy, 0) > 1e-20) return .{ ix, iy };
        }
    }
    return null;
}

fn mindxAt(s: *const SimP, ix: i64, iy: i64) f64 {
    const geom = s.cache.fillGeometry(ix, iy, 0);
    const dx0 = s.grid.cellSize(ix, 0) * @sqrt(geom.gg[1][1]);
    const dx1 = s.grid.cellSize(iy, 1) * @sqrt(geom.gg[2][2]);
    return @min(dx0, dx1); // NZ==1
}

test "M12 radviscosity: RADVISCNUDAMP caps ν at mindx²/(4 global_dt)" {
    const a = std.testing.allocator;
    var s = try SimP.init(a, puffy.makeGrid(32, 40), viscOptions());
    defer s.deinit();
    _ = try puffy.initAll(SimP, &s);

    const cell = torusCell(&s) orelse return error.NoTorusCellFound;
    const ix = cell[0];
    const iy = cell[1];
    var pp: [SimP.nv]f64 = undefined;
    s.p.load(ix, iy, 0, &pp);
    const geom = s.cache.fillGeometry(ix, iy, 0);

    // tiny dt → nulimit huge → ν = ALPHARADVISC·mfp (uncapped)
    const nu_free = try radvisc.calcRadViscCoeff(SimP, &s, ix, iy, 0, &pp, &geom, 1e-30);
    // huge dt → nulimit tiny → ν clamped to nulimit
    const big_dt: f64 = 1e10;
    const nu_capped = try radvisc.calcRadViscCoeff(SimP, &s, ix, iy, 0, &pp, &geom, big_dt);

    const mindx = mindxAt(&s, ix, iy);
    const nulimit = mindx * mindx / 2.0 / big_dt / 2.0;

    try std.testing.expect(nu_free > 0);
    try std.testing.expect(nu_capped < nu_free); // the cap actually bit
    try std.testing.expectApproxEqRel(nulimit, nu_capped, 1e-12);
}

test "M12 radviscosity: RADVISCMAXVELDAMP caps the viscous flux above MAXRADVISCVEL" {
    const a = std.testing.allocator;
    var s = try SimP.init(a, puffy.makeGrid(32, 40), viscOptions());
    defer s.deinit();
    _ = try puffy.initAll(SimP, &s);

    const cell = torusCell(&s) orelse return error.NoTorusCellFound;
    const ix = cell[0];
    const iy = cell[1];
    var pp: [SimP.nv]f64 = undefined;
    s.p.load(ix, iy, 0, &pp);
    const dim: usize = 0; // x-face
    const geom = s.cache.fillGeometryFace(ix, iy, 0, dim);
    const ee0 = LP.index(.ee);

    // a viscous tensor whose implied velocity clearly exceeds MAXRADVISCVEL,
    // driven through the R^{dim+1}_i components
    var rv: [4][4]f64 = @splat(@splat(0));
    for (1..4) |i| rv[dim + 1][i] = 1.0; // large relative to the rad conserveds

    var ff1: [SimP.nv]f64 = @splat(0);
    try radvisc.addRadViscFlux(SimP, &s, &ff1, &pp, &geom, dim, &rv);

    // scale the tensor by 10 — already past the cap, so the *damped* flux must
    // be (nearly) unchanged (dampfac ∝ 1/|R|), not 10× larger
    var rv10: [4][4]f64 = @splat(@splat(0));
    for (1..4) |i| rv10[dim + 1][i] = 10.0;
    var ff10: [SimP.nv]f64 = @splat(0);
    try radvisc.addRadViscFlux(SimP, &s, &ff10, &pp, &geom, dim, &rv10);

    // the velocity-limited flux saturates: ff10 ≈ ff1 (not 10×)
    var any_nonzero = false;
    for (0..4) |nu| {
        const v1 = ff1[ee0 + nu];
        const v10 = ff10[ee0 + nu];
        if (@abs(v1) > 0) {
            any_nonzero = true;
            try std.testing.expectApproxEqRel(v1, v10, 1e-10);
        }
    }
    try std.testing.expect(any_nonzero);
}

// The C golden (visc_golden_tests.zig) pins σ^ij numerically against KORAL, but
// a byte match can't tell whether the tensor is a *kinematic shear* — that it is
// symmetric and orthogonal to the frame 4-velocity. calc_shear_lab builds those
// two properties in by construction (explicit symmetrisation of the spatial
// block; the time row/column solved from u^μ σ_μν = 0), so they must hold to
// roundoff at any state. This gate survives a change to the physics — it asks
// "is the output still a shear tensor?", not "does it still match C?" — so it is
// the net a scientist editing the viscosity wants under them.
test "M12 radviscosity: σ_μν is symmetric and u-orthogonal (kinematic shear invariants)" {
    const a = std.testing.allocator;
    var s = try SimP.init(a, puffy.makeGrid(32, 40), viscOptions());
    defer s.deinit();
    _ = try puffy.initAll(SimP, &s);

    // an interior torus cell — its ±1 neighbours are interior too, so the shear
    // FD stencil reads real data (no ghost dependence) and σ is non-degenerate.
    const cell = torusCell(&s) orelse return error.NoTorusCellFound;
    const ix = cell[0];
    const iy = cell[1];
    var pp: [SimP.nv]f64 = undefined;
    s.p.load(ix, iy, 0, &pp);
    const geom = s.cache.fillGeometry(ix, iy, 0);

    // σ_μν (both indices lowered) in the radiation frame (FX..FZ, VELR).
    const sh = try radvisc.calcShearLab(SimP, &s, ix, iy, 0, comptime LP.index(.fx), .velr);
    const sigma = sh.s;

    // the radiation-frame 4-velocity the shear is projected around.
    const u = try relele.convertBoth(
        .{ 0, pp[LP.index(.fx)], pp[LP.index(.fy)], pp[LP.index(.fz)] },
        .velr,
        &geom,
    );

    // scale for the tolerances, and a guard that the shear isn't trivially zero
    // (a rotating torus cell shears — the invariants must have something to bite).
    var smag: f64 = 0;
    for (0..4) |i| {
        for (0..4) |j| smag = @max(smag, @abs(sigma[i][j]));
    }
    try std.testing.expect(smag > 0);

    for (0..4) |i| {
        // (1) symmetric: σ_ij = σ_ji.
        for (0..4) |j|
            try std.testing.expect(@abs(sigma[i][j] - sigma[j][i]) <= 1e-12 * smag);

        // (2) orthogonal to the 4-velocity: σ_μν u^ν = 0 for every μ.
        var dotu: f64 = 0;
        for (0..4) |j| dotu += sigma[i][j] * u.con[j];
        try std.testing.expect(@abs(dotu) <= 1e-9 * smag);
    }
}

// The shear *algebra* was extracted from calcShearLab's grid gather into the
// pure shearFromGradients kernel (physics/radvisc.zig). It can now be checked
// with a hand-built gradient and metric — no Sim, no FD stencil — so the tensor
// construction a scientist edits is pinned by an exact, hand-verifiable case.
test "M12 radviscosity: shearFromGradients — flat-space shear flow gives σ_12 = ∂u (pure kernel)" {
    // Minkowski metric (gg = diag(−1,1,1,1) in the [4][5] store), frame at rest
    // (u^μ = (1,0,0,0), u_μ = (−1,0,0,0)), no Christoffels. A pure shear field
    // ∂_{x²}u_1 = ∂_{x¹}u_2 = s then gives σ_12 = σ_21 = s, everything else 0.
    var gg: [4][5]f64 = @splat(@splat(0));
    gg[0][0] = -1;
    gg[1][1] = 1;
    gg[2][2] = 1;
    gg[3][3] = 1;
    const kr: [64]f64 = @splat(0);
    const ucon = [4]f64{ 1, 0, 0, 0 };
    const ucov = [4]f64{ -1, 0, 0, 0 };

    const s: f64 = 0.3;
    var du: [4][4]f64 = @splat(@splat(0));
    du[1][2] = s;
    du[2][1] = s;
    const zero_grad: [4][4]f64 = @splat(@splat(0));

    const out = radvisc.shearFromGradients(&du, &zero_grad, ucon, ucov, &gg, &kr);
    try std.testing.expectEqual(@as(f64, 0.0), out.div); // no expansion
    try std.testing.expectApproxEqAbs(s, out.s[1][2], 1e-15);
    try std.testing.expectApproxEqAbs(s, out.s[2][1], 1e-15);
    for (0..4) |i| {
        for (0..4) |j| {
            if ((i == 1 and j == 2) or (i == 2 and j == 1)) continue;
            try std.testing.expectApproxEqAbs(@as(f64, 0.0), out.s[i][j], 1e-15);
        }
    }

    // the zero gradient maps to the zero tensor exactly
    const zero = radvisc.shearFromGradients(&zero_grad, &zero_grad, ucon, ucov, &gg, &kr);
    for (0..4) |i| {
        for (0..4) |j| try std.testing.expectEqual(@as(f64, 0.0), zero.s[i][j]);
    }
    try std.testing.expectEqual(@as(f64, 0.0), zero.div);
}
