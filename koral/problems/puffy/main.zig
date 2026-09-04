//! PUFFY: radiative MHD limotorus around a Schwarzschild BH (a = 0)
//! (port of koral_lite/PROBLEMS/PUFFY). MASS defaults to 10 M☉ but is set
//! from the params file, so presets can retarget the scale (e.g. the
//! Sagittarius A* preset at ~4.3e6 M☉. See koral/problems/puffy/puffy3d_sgra.toml).
//!
//! The executable is `koral.driver.run` over the `App` contract below:
//! CLI, params, MPI ring, restart, the RK2IMEX loop and all output live in
//! koral/driver.zig; this file supplies only what is PUFFY's (the comptime
//! Config, setup/init, the startup notes, the seed-MRI report, and the
//! diagnostic radii for scalars.dat).
//!
//! CLI:  puffy [params.toml] [--restart <file|dir>]  (see koral/driver.zig)
//! Bridge a C serial restart (res####.head/.dat) into a KDMP with the
//! `res2kdmp` tool (tools/res2kdmp.zig).

const std = @import("std");
const koral = @import("koral");

const puffy = koral.problems.puffy;

const App = struct {
    pub const name = "puffy";
    pub const cfg = koral.config.puffy;
    pub const default_params = "koral/problems/puffy/puffy.toml";
    /// PUFFY diagnostic radii (postproc.c calc_scalars): luminosity at the
    /// outer shell and scale height at r = 15 M. Mass-flux radius is the horizon.
    pub const scalar_radii: koral.driver.ScalarRadii = .{ .lum = 5000.0, .scale = 15.0 };
    pub const Physics = puffy.Physics;
    pub const setup = puffy.setup;
    pub const initAll = puffy.initAllWith;

    pub fn reportSetupError(err: anyerror, p: *const koral.Params) void {
        if (err == error.InvalidDomain) {
            const phys = puffy.Physics.fromParams(p);
            std.debug.print(
                "puffy: invalid domain — RMAX ({d}) must exceed RMIN ({d}).\n",
                .{ phys.rmax, phys.rmin },
            );
        } else if (p.mesa_table.len > 0) {
            std.debug.print("puffy: cannot load MESA table '{s}': {s}\n", .{ p.mesa_table, @errorName(err) });
        }
    }

    pub fn banner(comptime SimT: type, phys: *const puffy.Physics, p: *const koral.Params, g: koral.Grid) void {
        _ = SimT;
        if (p.mesa_table.len > 0) {
            std.debug.print("puffy: MESA opacity table loaded from '{s}'\n", .{p.mesa_table});
        }
        const u = koral.Units.init(phys.mass);
        const r_hor = koral.metric.core.rHorizonBL(phys.mp.a);
        const cells_inside = if (phys.rmin < r_hor)
            (@log(r_hor - phys.mp.mksr0) - @log(phys.rmin - phys.mp.mksr0)) /
                ((@log(phys.rmax - phys.mp.mksr0) - @log(phys.rmin - phys.mp.mksr0)) / @as(f64, @floatFromInt(g.nx)))
        else
            0.0;
        std.debug.print(
            "puffy: M={d} M☉ (GM/c³={e:.4}s) a={d} r_h={d:.4} RMIN={d:.4} (~{d:.1} cells inside) RMAX={d:.1}\n",
            .{ phys.mass, u.gmc3(), phys.mp.a, r_hor, phys.rmin, cells_inside, phys.rmax },
        );
        if (phys.rmin >= r_hor) {
            std.debug.print(
                "puffy: *** NOTE: RMIN={d:.4} is OUTSIDE the horizon r_h={d:.4}; the plain-copy " ++
                    "inner boundary is not a clean excision. (An explicit params `rmin` override " ++
                    "defeats the auto horizon-tracking — drop it to restore a clean inner edge.) ***\n",
                .{ phys.rmin, r_hor },
            );
        }
        if (phys.mp.a != 0.0) {
            std.debug.print(
                "puffy: *** NOTE: a≠0 is UNVALIDATED — C KORAL PUFFY uses BHSPIN=0, so there is " ++
                    "no bit-for-bit oracle at this spin. Treat spinning runs as experiments. ***\n",
                .{},
            );
        }
    }

    /// Seed-MRI quality report after a fresh init (not after a restart).
    pub fn afterInit(comptime SimT: type, s: *SimT, phys: *const puffy.Physics, is_root: bool) !void {
        const sq = try puffy.seedQualityWith(SimT, s, phys);
        const m = s.globalSum(sq.mass);
        const qr = s.globalSum(sq.qr_m) / @max(m, 1e-300);
        const qth = s.globalSum(sq.qth_m) / @max(m, 1e-300);
        if (is_root and m > 0 and qth > 1e-30) {
            const target = 6.0;
            const mb_needed = phys.maxbeta * (target / qth) * (target / qth);
            std.debug.print(
                "puffy: seed MRI quality <Q_r>={d:.2} <Q_th>={d:.2} (mass-weighted; growth floor ~{d:.0}; perturb={d})\n",
                .{ qr, qth, target, phys.perturb },
            );
            if (qth < target) {
                if (mb_needed <= 0.2) {
                    std.debug.print(
                        "puffy: *** NOTE: seed <Q_th> is BELOW the growth floor — the linear MRI will be " ++
                            "grid-suppressed. maxbeta = {d:.3} would reach <Q_th> = {d:.0} on this grid.\n",
                        .{ mb_needed, target },
                    );
                } else {
                    std.debug.print(
                        "puffy: *** NOTE: seed <Q_th> is BELOW the growth floor, and no maxbeta <= 0.2 can fix " ++
                            "it on this grid (would need {d:.3}) — raise ny instead.\n",
                        .{mb_needed},
                    );
                }
            }
        }
    }
};

pub fn main(init: std.process.Init) !void {
    return koral.driver.run(App, init);
}

comptime {
    const L = koral.VarLayout(App.cfg);
    std.debug.assert(L.count == 13);
    std.debug.assert(L.index(.rho) == 0 and L.index(.entr) == 5);
    std.debug.assert(L.index(.b1) == 6 and L.index(.ee) == 9 and L.index(.fz) == 12);
}
