//! kdmp2ipole: export a PUFFY KDMP snapshot as the flat arrays ipole's KORAL
//! reader needs, for the GRRT code-to-code cross-check (docs/RENDER.md,
//! "Snapshot cross-check against ipole").
//!
//! ipole (Moscibrodzka & Gammie 2018) reads KORAL dumps natively, in KORAL's
//! own MKS2 coordinates, so the snapshot is handed over cell for cell with
//! no resampling: primitives go out in Kerr-Schild components (what KORAL
//! itself writes for ipole), together with the electron number density and
//! temperature THIS code's renderer derives from them, so both codes image
//! exactly the same plasma. tools/ipolecheck.py packs the result into the
//! HDF5 layout (HDF5 is not a dependency of this repository).
//!
//! usage: kdmp2ipole <params.toml> <file.kdmp> <out.kipl>
//!
//! Output, little-endian: "KIPL" u32 version=1, u32 n1 n2 n3, then 11 f64
//! scalars (t, startx1, startx2, dx1, dx2, mksr0, mksh0, spin, gamma,
//! GM/c² [cm], density unit [g/cm³ per code unit]), then 14 f64 arrays of
//! n1·n2·n3 values each, x1 slowest:
//!   r, theta, sqrt(-g) of the MKS2 grid,
//!   n_e [cm⁻³], u_int [code], T_e [K],
//!   U^r U^θ U^φ (KS components of the velocity relative to the normal
//!   observer, KORAL's VELR), B^r B^θ B^φ (KS, code units: b² in code units
//!   times 4π × the energy-density unit is Gauss²), b² [code], ρ [code].

const std = @import("std");
const koral = @import("koral");

const cfg = koral.config.puffy;
const puffy = koral.problems.puffy;
const render = koral.render;
const metric = koral.metric.core;
const forms = koral.metric.forms;
const Dual3 = koral.math.Dual3;
const L = koral.VarLayout(cfg);

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    var args = std.process.Args.Iterator.init(init.minimal.args);
    defer args.deinit();
    _ = args.next();
    const ppath = args.next() orelse return usage();
    const kpath = args.next() orelse return usage();
    const opath = args.next() orelse return usage();

    var p = try koral.Params.load(allocator, io, ppath);
    defer p.deinit(allocator);
    var loaded = try puffy.load(allocator, io, &p);
    defer loaded.deinit(allocator);
    const phys = &loaded.physics;

    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, kpath, allocator, .limited(1 << 33));
    var data = try render.DumpData.fromBytes(allocator, bytes);
    defer data.deinit(allocator);
    const h = data.header;
    if (h.nv != L.count) return error.DimMismatch;

    const grid = phys.makeGridNz(h.nx, h.ny, h.nz);
    var scene = render.Scene.init(grid, phys.mp, phys.consts(), phys.channels, phys.gam, &data, 1000.0, phys.rmax);
    scene.scattering = phys.scattering;
    const un = &scene.consts.units;

    const n: usize = @as(usize, h.nx) * h.ny * h.nz;
    const narr = 14;
    const arr = try allocator.alloc(f64, narr * n);
    defer allocator.free(arr);

    var te_max: f64 = 0;
    var b_max: f64 = 0;
    var ne_max: f64 = 0;
    var bad: usize = 0;
    for (0..h.nx) |ix| {
        for (0..h.ny) |iy| {
            for (0..h.nz) |iz| {
                const k = (ix * h.ny + iy) * h.nz + iz;
                const x = [4]f64{ 0, grid.xc(@intCast(ix)), grid.yc(@intCast(iy)), grid.zc(@intCast(iz)) };
                const cd = metric.compute(.mks2, phys.mp, x);
                const geom = render.geomFromCoordData(x, &cd);
                const r = @exp(x[1]) + phys.mp.mksr0;
                const th = forms.mks2Theta(Dual3.constant(x[2]), phys.mp.mksh0).v;
                const dthdx2 = forms.mks2DThetaDx2Metric(Dual3.constant(x[2]), phys.mp.mksh0).v;
                const drdx1 = r - phys.mp.mksr0;

                var pp: [L.count]f64 = undefined;
                @memcpy(&pp, data.cell(ix, iy, iz)[0..L.count]);

                var ne_cgs: f64 = 0;
                var te: f64 = 0;
                var bsq: f64 = 0;
                if (render.localState(cfg, &scene, &geom, pp)) |ls| {
                    ne_cgs = ls.ne * scene.consts.numdensgu2cgs;
                    te = ls.te;
                    bsq = ls.bsq;
                } else bad += 1;
                te_max = @max(te_max, te);
                ne_max = @max(ne_max, ne_cgs);
                b_max = @max(b_max, @sqrt(scene.consts.fourpi * un.endenGu2Cgs(bsq)));

                const vals = [narr]f64{
                    r,                              th,                              cd.gdet,
                    ne_cgs,                         pp[L.index(.uu)],                te,
                    drdx1 * pp[L.index(.vx)],       dthdx2 * pp[L.index(.vy)],       pp[L.index(.vz)],
                    drdx1 * pp[L.index(.b1)],       dthdx2 * pp[L.index(.b2)],       pp[L.index(.b3)],
                    bsq,                            pp[L.index(.rho)],
                };
                for (vals, 0..) |v, a| arr[a * n + k] = v;
            }
        }
    }

    var aw: std.Io.Writer.Allocating = try .initCapacity(allocator, 64 + 8 * 11 + 8 * arr.len);
    defer aw.deinit();
    try aw.writer.writeAll("KIPL");
    try aw.writer.writeInt(u32, 1, .little);
    try aw.writer.writeInt(u32, h.nx, .little);
    try aw.writer.writeInt(u32, h.ny, .little);
    try aw.writer.writeInt(u32, h.nz, .little);
    const scalars = [11]f64{
        h.t,            grid.minx,      grid.miny,   grid.dx, grid.dy,
        phys.mp.mksr0,  phys.mp.mksh0,  phys.mp.a,   phys.gam,
        un.masscm,      un.rhoGu2Cgs(1.0),
    };
    for (scalars) |v| try aw.writer.writeInt(u64, @bitCast(v), .little);
    try aw.writer.writeAll(std.mem.sliceAsBytes(arr));
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = opath, .data = aw.written() });

    std.debug.print(
        "kdmp2ipole: {s} -> {s}: {d}x{d}x{d} cells, t = {d:.3}; max T_e {e:.3} K (theta_e {d:.3}), max n_e {e:.3} cm^-3, max B {e:.3} G; {d} invalid cells\n",
        .{ kpath, opath, h.nx, h.ny, h.nz, h.t, te_max, te_max * 1.3806488e-16 / (9.1094e-28 * 2.9979246e10 * 2.9979246e10), ne_max, b_max, bad },
    );
}

fn usage() !void {
    std.debug.print("usage: kdmp2ipole <params.toml> <file.kdmp> <out.kipl>\n", .{});
    return error.BadArgs;
}
