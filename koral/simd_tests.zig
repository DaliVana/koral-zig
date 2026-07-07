//! P3a SIMD gates — the @Vector(4, f64) instantiation of the
//! comptime-T-generic residual chain (parallelization plan §2.2/§2.3 #1)
//! reproduces the scalar chain BIT-FOR-BIT, and the batched FD Jacobian
//! leaves the implicit solver bit-identical to the scalar reference path.
//!
//! Gates:
//!  * fillRadStateG/residualG with 4 distinct lane states == 4 scalar
//!    calls, every field bitwise, over MINK + KS geometries, PUFFY and
//!    grey opacity params, all 6 rung configurations
//!  * solve4dPrim with simd_jacobian on == off: ok/iters/uu/pp bitwise
//!    (converged and failed cells alike)
//!  * solveImplicitLab's full 6-rung ladder on == off over the hostile
//!    fuzz family: rung/iters/pp/uu bitwise

const std = @import("std");
const config = @import("config.zig");
const layout = @import("layout.zig");
const simd = @import("math/simd.zig");
const hydro = @import("physics/hydro.zig");
const thermo = @import("physics/thermo.zig");
const radforce = @import("physics/radforce.zig");
const invert_rad = @import("solve/invert_rad.zig");
const implicit = @import("solve/implicit.zig");
const p2u_mod = @import("p2u.zig");
const precompute = @import("metric/precompute.zig");
const metric = @import("metric/metric.zig");
const Geometry = @import("geometry.zig").Geometry;

const expect = std.testing.expect;
const geometryAt = precompute.geometryAt;

const V4 = @Vector(4, f64);
const puffy_mp = metric.MetricParams{ .a = 0.0, .mksr0 = 0.1, .mksh0 = 0.9 };
const gam: f64 = 5.0 / 3.0;

const cfg = config.Config{
    .modules = &.{ .hydro, .mhd, .radiation },
    .reconstruction = .ppm,
    .flux = .laxf,
    .timestepping = .rk2imex,
    .coords = .mink,
};
const L = layout.VarLayout(cfg);
const NV = L.count;
const ImplT = implicit.Solver(cfg);

const rad_params = invert_rad.RadParams.puffy;

/// Bitwise equality of two f64 aggregates (structs/arrays of f64),
/// walked by comptime reflection — catches sign-of-zero and NaN-payload
/// differences that `==` would hide.
fn expectAggBits(comptime S: type, want: S, got: S) !void {
    if (comptime S == f64) {
        try std.testing.expectEqual(
            @as(u64, @bitCast(want)),
            @as(u64, @bitCast(got)),
        );
        return;
    }
    switch (@typeInfo(S)) {
        .array => |info| {
            for (0..info.len) |i| try expectAggBits(info.child, want[i], got[i]);
        },
        .@"struct" => |info| {
            inline for (info.fields) |fld| {
                try expectAggBits(fld.type, @field(want, fld.name), @field(got, fld.name));
            }
        },
        else => try std.testing.expectEqual(want, got),
    }
}

fn ppFromTemps(
    c: *const thermo.Consts,
    rho: f64,
    tgas: f64,
    v: [3]f64,
    erf: f64,
    f: [3]f64,
    b: [3]f64,
) [NV]f64 {
    const uint = tgas * rho * c.kb_over_mugas_mp / (gam - 1.0);
    var pp: [NV]f64 = @splat(0);
    pp[L.index(.rho)] = rho;
    pp[L.index(.uu)] = uint;
    pp[L.index(.vx)] = v[0];
    pp[L.index(.vy)] = v[1];
    pp[L.index(.vz)] = v[2];
    pp[L.index(.entr)] = hydro.sFromU(rho, uint, gam);
    pp[L.index(.b1)] = b[0];
    pp[L.index(.b2)] = b[1];
    pp[L.index(.b3)] = b[2];
    pp[L.index(.ee)] = erf;
    pp[L.index(.fx)] = f[0];
    pp[L.index(.fy)] = f[1];
    pp[L.index(.fz)] = f[2];
    return pp;
}

/// A controlled-γ 3-velocity in the given spatial metric.
fn randVel(rng: std.Random, geo: *const Geometry, gmax: f64) [3]f64 {
    var n: [3]f64 = undefined;
    var qsq: f64 = 0;
    for (0..3) |k| n[k] = 2.0 * rng.float(f64) - 1.0;
    for (0..3) |k| {
        for (0..3) |l| qsq += n[k] * n[l] * geo.gg[k + 1][l + 1];
    }
    const gt = 1.0 + (gmax - 1.0) * rng.float(f64);
    const s = @sqrt((gt * gt - 1.0) / qsq);
    var v: [3]f64 = undefined;
    for (0..3) |k| v[k] = n[k] * s;
    return v;
}

fn randPp(rng: std.Random, c: *const thermo.Consts, geo: *const Geometry, with_b: bool) [NV]f64 {
    const rho = std.math.pow(f64, 10.0, -12.0 + 8.0 * rng.float(f64));
    const tgas = std.math.pow(f64, 10.0, 5.0 + 7.0 * rng.float(f64));
    const zeta = std.math.pow(f64, 10.0, -2.0 + 4.0 * rng.float(f64));
    const v = randVel(rng, geo, 2.5);
    const f = randVel(rng, geo, 2.5);
    var b: [3]f64 = .{ 0, 0, 0 };
    if (with_b) {
        const uint = tgas * rho * c.kb_over_mugas_mp / (gam - 1.0);
        const beta = std.math.pow(f64, 10.0, -2.0 + 4.0 * rng.float(f64));
        const bmag = @sqrt(2.0 * (gam - 1.0) * uint / beta);
        b = .{ bmag * (2.0 * rng.float(f64) - 1.0), bmag * (2.0 * rng.float(f64) - 1.0), 0 };
    }
    return ppFromTemps(c, rho, tgas, v, c.lteEfromT(zeta * tgas), f, b);
}

test "SIMD: fillRadState + residual lanes are bitwise equal to 4 scalar evaluations" {
    var prng = std.Random.DefaultPrng.init(0x53494d4431);
    const rng = prng.random();

    const params = [_]radforce.Params{
        radforce.Params.puffy(),
        radforce.Params.grey(
            @import("units.zig").Units.init(10.0),
            thermo.Composition.cdefault,
            0.34,
            0.4,
        ),
    };
    const geoms = [_]Geometry{
        geometryAt(.mink, puffy_mp, .{ 0, 0, 0, 0 }),
        geometryAt(.ks, puffy_mp, .{ 0, 7.3, 1.1, 0.4 }),
    };
    const rungs = [_]struct { p: implicit.WhichPrim, e: implicit.WhichEq, fr: implicit.WhichFrame }{
        .{ .p = .mhd, .e = .energy, .fr = .lab },
        .{ .p = .rad, .e = .energy, .fr = .lab },
        .{ .p = .mhd, .e = .energy, .fr = .ff },
        .{ .p = .rad, .e = .energy, .fr = .ff },
        .{ .p = .mhd, .e = .entropy, .fr = .ff },
        .{ .p = .rad, .e = .entropy, .fr = .ff },
    };

    outer: for (0..60) |it| {
        const p = &params[it % 2];
        const geo = &geoms[(it / 2) % 2];
        const c = &p.consts;

        // 4 distinct lane states + the shared base state
        var lane_pp: [4][NV]f64 = undefined;
        var lane_uu: [4][NV]f64 = undefined;
        for (0..4) |j| {
            lane_pp[j] = randPp(rng, c, geo, it % 3 != 0);
            lane_uu[j] = p2u_mod.p2u(cfg, lane_pp[j], geo, gam) catch continue :outer;
        }
        const pp00 = randPp(rng, c, geo, it % 3 != 0);
        const uu00 = p2u_mod.p2u(cfg, pp00, geo, gam) catch continue :outer;
        const st00 = try radforce.fillRadState(cfg, pp00, geo, gam, p);
        const dt = std.math.pow(f64, 10.0, -8.0 + 8.0 * rng.float(f64));

        // batched chain
        var ppv: [NV]V4 = undefined;
        var uuv: [NV]V4 = undefined;
        for (0..NV) |iv| {
            ppv[iv] = .{ lane_pp[0][iv], lane_pp[1][iv], lane_pp[2][iv], lane_pp[3][iv] };
            uuv[iv] = .{ lane_uu[0][iv], lane_uu[1][iv], lane_uu[2][iv], lane_uu[3][iv] };
        }
        const ggv = simd.splatBlock(V4, &geo.gg);
        const GGv = simd.splatBlock(V4, &geo.GG);
        const stv = radforce.fillRadStateG(cfg, V4, ppv, &ggv, &GGv, gam, p);

        // per-lane state fill must match the scalar fill bitwise
        for (0..4) |j| {
            const st_scalar = try radforce.fillRadState(cfg, lane_pp[j], geo, gam, p);
            const st_lane = simd.extractLane(radforce.RadState, stv, j);
            try expectAggBits(radforce.RadState, st_scalar, st_lane);
        }

        // the residual (which pulls calcGiFromStateG, the boost and the
        // error norms) must match per lane, for every rung configuration
        const rung = rungs[it % rungs.len];
        var fv: [4]V4 = undefined;
        const errv: [4]f64 = ImplT.residualG(V4, &uuv, &ppv, &stv, &uu00, &st00, dt, &ggv, &GGv, simd.splat(V4, geo.gdet), p, rung.p, rung.e, rung.fr, &fv);
        for (0..4) |j| {
            const st_scalar = try radforce.fillRadState(cfg, lane_pp[j], geo, gam, p);
            var fs: [4]f64 = undefined;
            const errs = try ImplT.residual(&lane_uu[j], &lane_pp[j], &st_scalar, &uu00, &st00, dt, geo, p, rung.p, rung.e, rung.fr, &fs);
            try expectAggBits(f64, errs, errv[j]);
            for (0..4) |k| {
                const fk: [4]f64 = fv[k];
                try expectAggBits(f64, fs[k], fk[j]);
            }
        }
    }
}

test "SIMD: solve4dPrim with the batched Jacobian is bitwise equal to the scalar path" {
    var prng = std.Random.DefaultPrng.init(0x53494d4432);
    const rng = prng.random();
    const p = radforce.Params.puffy();
    const c = &p.consts;

    var ip_on = implicit.ImplicitParams.puffy;
    ip_on.simd_jacobian = true;
    var ip_off = implicit.ImplicitParams.puffy;
    ip_off.simd_jacobian = false;

    const geoms = [_]Geometry{
        geometryAt(.mink, puffy_mp, .{ 0, 0, 0, 0 }),
        geometryAt(.ks, puffy_mp, .{ 0, 7.3, 1.1, 0.4 }),
    };

    var nok: usize = 0;
    for (0..300) |it| {
        const geo = &geoms[it % 2];
        const pp00 = randPp(rng, c, geo, it % 3 != 0);
        const uu00 = p2u_mod.p2u(cfg, pp00, geo, gam) catch continue;
        const dt = std.math.pow(f64, 10.0, -8.0 + 8.0 * rng.float(f64));
        const whichprim: implicit.WhichPrim = if (it % 4 < 2) .mhd else .rad;

        var pp_on = pp00;
        var pp_off = pp00;
        const r_on = ImplT.solve4dPrim(&uu00, &pp00, geo, dt, gam, rad_params, &p, &ip_on, whichprim, .energy, .lab, &pp_on);
        const r_off = ImplT.solve4dPrim(&uu00, &pp00, geo, dt, gam, rad_params, &p, &ip_off, whichprim, .energy, .lab, &pp_off);

        try std.testing.expectEqual(r_off.ok, r_on.ok);
        try std.testing.expectEqual(r_off.iters, r_on.iters);
        try expectAggBits([NV]f64, r_off.uu, r_on.uu);
        try expectAggBits([NV]f64, pp_off, pp_on);
        if (r_on.ok) nok += 1;
    }
    // enough convergent cells that the equality above exercised real
    // Newton paths, not just early failures
    std.debug.print("simd solve4dPrim identity: {d}/300 converged (both paths)\n", .{nok});
    try expect(nok > 80);
}

test "SIMD: the full 6-rung ladder is bitwise equal with the batched Jacobian" {
    var prng = std.Random.DefaultPrng.init(0x53494d4433);
    const rng = prng.random();
    const p = radforce.Params.puffy();
    const c = &p.consts;

    var ip_on = implicit.ImplicitParams.puffy;
    ip_on.simd_jacobian = true;
    var ip_off = implicit.ImplicitParams.puffy;
    ip_off.simd_jacobian = false;

    const geoms = [_]Geometry{
        geometryAt(.mink, puffy_mp, .{ 0, 0, 0, 0 }),
        geometryAt(.ks, puffy_mp, .{ 0, 7.3, 1.1, 0.4 }),
    };

    var nok: usize = 0;
    var nrungup: usize = 0;
    for (0..300) |it| {
        const geo = &geoms[it % 2];
        const pp00 = randPp(rng, c, geo, it % 3 != 0);
        const dt = std.math.pow(f64, 10.0, -10.0 + 10.0 * rng.float(f64));

        var pp_on = pp00;
        var pp_off = pp00;
        var uu_on: [NV]f64 = undefined;
        var uu_off: [NV]f64 = undefined;
        const r_on = ImplT.solveImplicitLab(&uu_on, &pp_on, geo, dt, gam, rad_params, &p, &ip_on);
        const r_off = ImplT.solveImplicitLab(&uu_off, &pp_off, geo, dt, gam, rad_params, &p, &ip_off);

        try std.testing.expectEqual(r_off.ok, r_on.ok);
        try std.testing.expectEqual(r_off.rung, r_on.rung);
        try std.testing.expectEqual(r_off.iters, r_on.iters);
        try expectAggBits([NV]f64, pp_off, pp_on);
        if (r_on.ok) {
            try expectAggBits([NV]f64, uu_off, uu_on);
            nok += 1;
            if (r_on.rung > 0) nrungup += 1;
        }
    }
    // the family must exercise both convergence and the deeper rungs
    // (rung > 0 means at least one failed 4dprim call ran the batched
    // Jacobian through its failure paths before a later rung succeeded)
    std.debug.print("simd ladder identity: {d}/300 ok, {d} via rung > 0\n", .{ nok, nrungup });
    try expect(nok > 80);
    try expect(nrungup > 10);
}

// ---- code review 2026-07-06, hot-path finding #4: dead-work slimming ----

test "slim RadState/Gi agree bitwise with the full path on every consumed field" {
    var prng = std.Random.DefaultPrng.init(0x53494d4434);
    const rng = prng.random();

    const params = [_]radforce.Params{
        radforce.Params.puffy(),
        radforce.Params.grey(
            @import("units.zig").Units.init(10.0),
            thermo.Composition.cdefault,
            0.34,
            0.4,
        ),
    };
    const geoms = [_]Geometry{
        geometryAt(.mink, puffy_mp, .{ 0, 0, 0, 0 }),
        geometryAt(.ks, puffy_mp, .{ 0, 7.3, 1.1, 0.4 }),
    };

    for (0..200) |it| {
        const p = &params[it % 2];
        const geo = &geoms[(it / 2) % 2];
        const c = &p.consts;
        const pp = randPp(rng, c, geo, it % 3 != 0);

        const st_full = try radforce.fillRadState(cfg, pp, geo, gam, p);
        const st_slim = try radforce.fillRadStateSlim(cfg, pp, geo, gam, p);

        // every field the residual / f1dErr consumes must be bit-identical
        inline for (.{ "rho", "uint", "tgas", "te", "ti", "ne", "bsq", "ehat", "tradbb", "trad", "kappaes", "kappa" }) |f| {
            try expectAggBits(f64, @field(st_full, f), @field(st_slim, f));
        }
        try expectAggBits([4]f64, st_full.ucon, st_slim.ucon);
        try expectAggBits([4]f64, st_full.ucov, st_slim.ucov);
        try expectAggBits([4][4]f64, st_full.rij, st_slim.rij);
        try expectAggBits(f64, st_full.opac.gas_abs, st_slim.opac.gas_abs);
        try expectAggBits(f64, st_full.opac.rad_abs, st_slim.opac.rad_abs);
        try expectAggBits(f64, st_full.opac.rad_ross, st_slim.opac.rad_ross);

        // on the default (PUFFY) opacity path the dropped work is zeroed —
        // confirms the slim path really skipped it (the grey branch fills
        // all slots unconditionally, so only check .default)
        if (it % 2 == 0) {
            try expectAggBits(f64, 0.0, st_slim.sgas);
            try expectAggBits(f64, 0.0, st_slim.opac.gas_ross);
            try expectAggBits(f64, 0.0, st_slim.opac.tot_emissivity);
        }

        // Gi from the SAME full state: the boost-skipping slim path must
        // reproduce ff[0] and all of lab bit-for-bit — the only components
        // the residual and f1dErr read.
        const vprim = [3]f64{ pp[L.index(.vx)], pp[L.index(.vy)], pp[L.index(.vz)] };
        const gi_full = try radforce.calcGiFromState(&st_full, vprim, geo, p);
        const gi_slim = try radforce.calcGiFromStateSlim(&st_full, vprim, geo, p);
        try expectAggBits(f64, gi_full.ff[0], gi_slim.ff[0]);
        try expectAggBits([4]f64, gi_full.lab, gi_slim.lab);
    }
}

test "slim_state RadState fills leave the full 6-rung ladder bitwise identical" {
    var prng = std.Random.DefaultPrng.init(0x53494d4435);
    const rng = prng.random();
    const p = radforce.Params.puffy();
    const c = &p.consts;

    const ip_on = implicit.ImplicitParams.puffy; // slim_state = true (default)
    var ip_off = implicit.ImplicitParams.puffy;
    ip_off.slim_state = false;

    const geoms = [_]Geometry{
        geometryAt(.mink, puffy_mp, .{ 0, 0, 0, 0 }),
        geometryAt(.ks, puffy_mp, .{ 0, 7.3, 1.1, 0.4 }),
    };

    var nok: usize = 0;
    var nrungup: usize = 0;
    for (0..300) |it| {
        const geo = &geoms[it % 2];
        const pp00 = randPp(rng, c, geo, it % 3 != 0);
        const dt = std.math.pow(f64, 10.0, -10.0 + 10.0 * rng.float(f64));

        var pp_on = pp00;
        var pp_off = pp00;
        var uu_on: [NV]f64 = undefined;
        var uu_off: [NV]f64 = undefined;
        const r_on = ImplT.solveImplicitLab(&uu_on, &pp_on, geo, dt, gam, rad_params, &p, &ip_on);
        const r_off = ImplT.solveImplicitLab(&uu_off, &pp_off, geo, dt, gam, rad_params, &p, &ip_off);

        try std.testing.expectEqual(r_off.ok, r_on.ok);
        try std.testing.expectEqual(r_off.rung, r_on.rung);
        try std.testing.expectEqual(r_off.iters, r_on.iters);
        try expectAggBits([NV]f64, pp_off, pp_on);
        if (r_on.ok) {
            try expectAggBits([NV]f64, uu_off, uu_on);
            nok += 1;
            if (r_on.rung > 0) nrungup += 1;
        }
    }
    std.debug.print("slim_state ladder identity: {d}/300 ok, {d} via rung > 0\n", .{ nok, nrungup });
    try expect(nok > 80);
    try expect(nrungup > 10);
}

// finding #5: the slim wavespeed/radvisc χ must equal the full C-parity
// calcChi bit-for-bit. calcChiSlim skips the radiation frame and the four
// Trad-dependent opacity channels, computing only κ = gas_abs + κ_es(Te).
test "slim calcChi agrees bitwise with the full calcChi" {
    var prng = std.Random.DefaultPrng.init(0x53494d4436);
    const rng = prng.random();

    const params = [_]radforce.Params{
        radforce.Params.puffy(),
        radforce.Params.grey(
            @import("units.zig").Units.init(10.0),
            thermo.Composition.cdefault,
            0.34,
            0.4,
        ),
    };
    const geoms = [_]Geometry{
        geometryAt(.mink, puffy_mp, .{ 0, 0, 0, 0 }),
        geometryAt(.ks, puffy_mp, .{ 0, 7.3, 1.1, 0.4 }),
    };

    for (0..200) |it| {
        const p = &params[it % 2];
        const geo = &geoms[(it / 2) % 2];
        const c = &p.consts;
        // toggle B on/off to exercise the bmagcgs > 0 select and the b = 0 path
        const pp = randPp(rng, c, geo, it % 3 != 0);

        const chi_full = try radforce.calcChi(cfg, pp, geo, gam, p);
        const chi_slim = try radforce.calcChiSlim(cfg, pp, geo, gam, p);
        try expectAggBits(f64, chi_full, chi_slim);
    }
}

// finding #10: hoisting state00 + the bisect starting guess out of the rung
// loop must leave every rung's result untouched. The wrapper solve4dPrim
// recomputes both internally (the pre-#10 per-rung path); solve4dPrimCore is
// fed the once-hoisted pair — they must agree bit-for-bit on all six rungs.
test "solve4dPrim state00/bisect hoist is bitwise faithful across all six rungs" {
    var prng = std.Random.DefaultPrng.init(0x53494d4437);
    const rng = prng.random();
    const p = radforce.Params.puffy();
    const c = &p.consts;
    const ip = implicit.ImplicitParams.puffy;

    const geoms = [_]Geometry{
        geometryAt(.mink, puffy_mp, .{ 0, 0, 0, 0 }),
        geometryAt(.ks, puffy_mp, .{ 0, 7.3, 1.1, 0.4 }),
    };
    const rungs = [_]struct { pr: implicit.WhichPrim, eq: implicit.WhichEq, fr: implicit.WhichFrame }{
        .{ .pr = .rad, .eq = .energy, .fr = .lab },
        .{ .pr = .rad, .eq = .energy, .fr = .ff },
        .{ .pr = .mhd, .eq = .energy, .fr = .lab },
        .{ .pr = .mhd, .eq = .energy, .fr = .ff },
        .{ .pr = .rad, .eq = .entropy, .fr = .ff },
        .{ .pr = .mhd, .eq = .entropy, .fr = .ff },
    };

    var nchecked: usize = 0;
    for (0..200) |it| {
        const geo = &geoms[it % 2];
        const pp00 = randPp(rng, c, geo, it % 3 != 0);
        const dt = std.math.pow(f64, 10.0, -10.0 + 10.0 * rng.float(f64));
        const uu00 = p2u_mod.p2u(cfg, pp00, geo, gam) catch continue;

        // hoist state00 + bisect once, exactly as solveImplicitLab now does
        const state00 = radforce.fillRadState(cfg, pp00, geo, gam, &p) catch continue;
        var pp0 = pp00;
        if (ip.start_with_bisect) _ = ImplT.solve1dPrim(&pp00, &state00, geo, dt, gam, &p, &pp0);

        for (rungs) |r| {
            var ppw_wrap = pp00;
            var ppw_core = pp00;
            const rw = ImplT.solve4dPrim(&uu00, &pp00, geo, dt, gam, rad_params, &p, &ip, r.pr, r.eq, r.fr, &ppw_wrap);
            const rc = ImplT.solve4dPrimCore(&uu00, &state00, &pp0, geo, dt, gam, rad_params, &p, &ip, r.pr, r.eq, r.fr, &ppw_core);
            try std.testing.expectEqual(rw.ok, rc.ok);
            try std.testing.expectEqual(rw.iters, rc.iters);
            try expectAggBits([NV]f64, rw.uu, rc.uu);
            try expectAggBits([NV]f64, ppw_wrap, ppw_core);
            nchecked += 1;
        }
    }
    std.debug.print("solve4dPrim hoist faithful: {d} (rung × case) checks\n", .{nchecked});
    try expect(nchecked > 500);
}
