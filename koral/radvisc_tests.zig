//! M12 theory gates for radiative shear viscosity that the C golden does not
//! reach: the RADVISCNUDAMP diffusion cap (calcRadVisccoeff) and the
//! RADVISCMAXVELDAMP characteristic-velocity cap (addRadViscFlux). The shear
//! tensor, the coefficient, and the assembled R^i_j are validated numerically
//! against C in golden_visc_test.zig; here we pin the two limiter properties.

const std = @import("std");
const config = @import("config.zig");
const sim_mod = @import("sim.zig");
const puffy = @import("problems/puffy.zig");
const radvisc = @import("physics/radvisc.zig");
const invert = @import("solve/invert.zig");
const invert_rad = @import("solve/invert_rad.zig");
const radforce = @import("physics/radforce.zig");

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
    const dx0 = s.grid.size(ix, 0) * @sqrt(geom.gg[1][1]);
    const dx1 = s.grid.size(iy, 1) * @sqrt(geom.gg[2][2]);
    return @min(dx0, dx1); // NZ==1
}

test "M12 radviscosity: RADVISCNUDAMP caps ν at mindx²/(4 global_dt)" {
    const a = std.testing.allocator;
    var s = try SimP.init(a, puffy.makeGrid(32, 40), viscOptions());
    defer s.deinit();
    _ = try puffy.initAll(SimP, &s);

    const cell = torusCell(&s) orelse return error.SkipZigTest;
    const ix = cell[0];
    const iy = cell[1];
    var pp: [SimP.nv]f64 = undefined;
    s.p.load(ix, iy, 0, &pp);
    const geom = s.cache.fillGeometry(ix, iy, 0);

    // tiny dt → nulimit huge → ν = ALPHARADVISC·mfp (uncapped)
    const nu_free = try radvisc.calcRadVisccoeff(SimP, &s, ix, iy, 0, &pp, &geom, 1e-30);
    // huge dt → nulimit tiny → ν clamped to nulimit
    const big_dt: f64 = 1e10;
    const nu_capped = try radvisc.calcRadVisccoeff(SimP, &s, ix, iy, 0, &pp, &geom, big_dt);

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

    const cell = torusCell(&s) orelse return error.SkipZigTest;
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
