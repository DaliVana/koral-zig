//! M12 keystone (mean-field dynamo): one apply_dynamo on the PUFFY t=0 state
//! with an injected toroidal field B³ = 10·B², cell-by-cell against the C
//! oracle (oracle/harness_dynamo.c → tests/golden/init/puffy_dynamo.kini.gz,
//! B¹,B²,B³ over the domain, dt=10).
//!
//! This exercises the whole dynamo pipeline end-to-end: the CALCHRONTHEGO
//! density-weighted scale height (calc_avgs_throughout), the BL field-pitch
//! angle (calc_angle_brbphibsq → trans_pmhd_coco + b^μ), the ΔA_φ prescription
//! with its ALPHAFLIPSSIGN equatorial flip, the DAMPBETA azimuthal damping,
//! the calc_BfromA curl of ΔA_φ, and the p2u/u2p round-trip.
//!
//! B¹,B² take the curl of ΔA_φ (∝ the injected B³), and every ΔA_φ factor
//! (scaleth, the field angle, Ê in β) runs through ρ/u/Ê, which carry C's
//! ~1e-3 qags kink error (the M11 keystone). So the golden matches at
//! field-scale ~1e-3 — the same C-qags-limited story, NOT a Zig error; a
//! dynamo bug would show far above this floor.

const std = @import("std");
const build_options = @import("build_options");
const config = @import("config.zig");
const sim_mod = @import("sim.zig");
const golden = @import("testing/golden.zig");
const puffy = @import("problems/puffy/puffy.zig");
const dynamo = @import("magn/dynamo.zig");
const p2u_mod = @import("p2u.zig");
const invert = @import("solve/invert.zig");
const invert_rad = @import("solve/invert_rad.zig");
const radforce = @import("physics/radforce.zig");

const SimP = sim_mod.Sim(config.puffy);
const LP = SimP.Layout;

const inject_factor: f64 = 10.0; // matches harness_dynamo.c
const dyn_dt: f64 = 10.0;

fn dynamoOptions() SimP.Options {
    return .{
        .coords = .mks2,
        .mp = puffy.mp,
        .gam = puffy.gam,
        .floors = invert.FloorParams.puffy,
        .rad = invert_rad.RadParams.puffy,
        .opac = radforce.Params.puffy(),
        .correct_polaraxis = true,
        .nccorrectpolar = 2,
        .dynamo = true,
        .bc_x = .specific,
        .bc_y = .specific,
        .specific_bc = &puffy.Bc(SimP).calc,
    };
}

const Dev = golden.FieldDev;

test "M12 dynamo: apply_dynamo B vs C (PUFFY t=0 + injected B³, 384×360)" {
    if (!build_options.slow_tests) return error.SkipZigTest;
    const a = std.testing.allocator;

    var s = try SimP.init(a, puffy.makeGrid(384, 360), dynamoOptions());
    defer s.deinit();
    _ = try puffy.initAll(SimP, &s);

    const b1 = LP.index(.b1);
    const b2 = LP.index(.b2);
    const b3 = LP.index(.b3);
    const nx = s.nxi();
    const ny = s.nyi();

    // inject B³ = 10·B² into the domain, keep u consistent (matches harness)
    {
        var iy: i64 = 0;
        while (iy < ny) : (iy += 1) {
            var ix: i64 = 0;
            while (ix < nx) : (ix += 1) {
                var pp: [SimP.nv]f64 = undefined;
                s.p.load(ix, iy, 0, &pp);
                pp[b3] = inject_factor * pp[b2];
                const geom = s.cache.fillGeometry(ix, iy, 0);
                const uu = try p2u_mod.p2u(config.puffy, pp, &geom, s.opt.gam);
                s.p.store(ix, iy, 0, &pp);
                s.u.store(ix, iy, 0, &uu);
            }
        }
    }

    try dynamo.applyDynamo(SimP, &s, 0.0, dyn_dt);

    var k = try golden.readKini(a, "init/puffy_dynamo.kini.gz");
    defer k.deinit();
    const slots = [_]usize{ b1, b2, b3 };
    var dev = [_]Dev{ .{}, .{}, .{} };
    var jy: usize = 0;
    while (jy < k.nSamplesY()) : (jy += 1) {
        var jx: usize = 0;
        while (jx < k.nSamplesX()) : (jx += 1) {
            const ix = k.cellX(jx);
            const iy = k.cellY(jy);
            const base = k.base(jx, jy, 0);
            var pp: [SimP.nv]f64 = undefined;
            s.p.load(ix, iy, 0, &pp);
            for (0..3) |c| dev[c].add(k.data[base + c], pp[slots[c]]);
        }
    }

    const names = [_][]const u8{ "B1", "B2", "B3" };
    std.debug.print("M12 dynamo field-scale deviations (dt={d}, B³=10·B²):\n", .{dyn_dt});
    for (0..3) |c| std.debug.print("  {s} {e:.3}  (|Δ|max {e:.3} / |{s}|max {e:.3})\n", .{ names[c], dev[c].dev(), dev[c].max_diff, names[c], dev[c].max_mag });

    // measured: B1 8.6e-4, B2 1.1e-3, B3 1.1e-3 — the qags kink floor
    // (scaleth, field angle, Ê in β all run through ρ/u/Ê)
    for (0..3) |c| try std.testing.expect(dev[c].dev() <= 3e-3);
}
