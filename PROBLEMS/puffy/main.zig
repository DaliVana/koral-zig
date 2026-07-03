//! PUFFY — radiative MHD limotorus around a 10 M☉ Schwarzschild BH
//! (port of koral_lite/PROBLEMS/PUFFY).
//!
//! M0 status: comptime config + runtime params + grid/field allocation and
//! an empty time loop. Physics arrives with milestones M1–M12.

const std = @import("std");
const koral = @import("koral");

/// Comptime physics/algorithm configuration (PROBLEMS/PUFFY/define.h):
/// MHD + M1 radiation, PPM, LAXF, RK2IMEX, MKS2 coordinates.
const cfg = koral.config.puffy;
const L = koral.VarLayout(cfg);

pub fn main(init: std.process.Init) !void {
    const a = init.gpa;
    const io = init.io;

    var args = std.process.Args.Iterator.init(init.minimal.args);
    defer args.deinit();
    _ = args.next(); // program name
    const params_path = args.next() orelse "PROBLEMS/puffy/puffy.toml";

    const p = koral.Params.load(a, io, params_path) catch |err| {
        std.debug.print("puffy: cannot load params from '{s}': {s}\n", .{ params_path, @errorName(err) });
        return err;
    };
    defer a.free(p.out_dir);

    // Internal (uniform) MKS2 bounds: x1 = ln(r - mksr0), x2 ∈ (0,1), x3 = φ.
    const grid = koral.Grid.init(.{
        .nx = p.nx,
        .ny = p.ny,
        .nz = p.nz,
        .ng = comptime cfg.ghostCells(),
        .minx = @log(p.rmin - p.mksr0),
        .maxx = @log(p.rmax - p.mksr0),
        .miny = p.miny,
        .maxy = p.maxy,
        .minz = p.minz,
        .maxz = p.maxz,
    });

    var prims = try koral.Field(L.count).init(a, grid);
    defer prims.deinit();
    var cons = try koral.Field(L.count).init(a, grid);
    defer cons.deinit();

    const u = koral.Units.init(p.mass);
    std.debug.print(
        "puffy: NV={d} grid {d}x{d}x{d} (+{d} ghosts) | M={d} Msun (GM/c^3 = {e:.4} s) | x1 in [{d:.4}, {d:.4}]\n",
        .{ L.count, grid.nx, grid.ny, grid.nz, grid.ng, p.mass, u.gmc3(), grid.minx, grid.maxx },
    );

    // ---- empty time loop (M0) ----
    var t: f64 = p.tstart;
    var nstep: usize = 0;
    while (t < p.tmax and nstep < p.nstep_max) {
        // M5+: dt from CFL; step(); outputs.
        nstep += 1;
        t = p.tmax; // nothing to integrate yet
    }
    std.debug.print("puffy: done (t = {d}, {d} steps)\n", .{ t, nstep });
}

comptime {
    // The C-diffability contract (mnemonics.h) — refuse to compile if broken.
    std.debug.assert(L.count == 13);
    std.debug.assert(L.index(.rho) == 0 and L.index(.entr) == 5);
    std.debug.assert(L.index(.b1) == 6 and L.index(.ee) == 9 and L.index(.fz) == 12);
}
