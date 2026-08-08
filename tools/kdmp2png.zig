//! kdmp2png — GRRT-render a PUFFY KDMP snapshot into a PNG image: null
//! geodesics through the run's own Kerr/MKS2 metric + frequency-integrated
//! radiative transfer using the run's own opacities and M1 radiation field
//! (see koral/render/render.zig for the method).
//!
//! The sim configuration is rebuilt from the SAME params file and the SAME
//! code path as the puffy driver and kdmp2silo, so the grid, metric, thermo
//! constants and opacity channels match what produced the checkpoint.
//!
//! usage: kdmp2png <params.toml> <file.kdmp> [out.png] [options]
//!   --size N       image width and height in pixels   (default 512)
//!   --fov M        field of view at the hole, in GM/c² (default 100)
//!   --incl DEG     inclination from the spin axis      (default 60)
//!   --phi DEG      camera azimuth                      (default 0)
//!   --rcam M       camera radius in GM/c²              (default 1000)
//!   --nu GHZ       observed frequency: monochromatic synthetic image at
//!                  ν_obs (thermal synchrotron + free-free + scattering,
//!                  emission.zig) with I_ν reported in CGS and brightness
//!                  temperature; 0 = gray bolometric      (default 0)
//!   --dist KPC     source distance — with --nu, also print the integrated
//!                  flux density in Jy (0 = skip)         (default 0)
//!   --screen       VALIDATION mode: ignore the fluid, render a bright
//!                  celestial sphere (captured rays dark) and overlay the
//!                  analytic Bardeen shadow curve in green — the dark
//!                  boundary must hug the curve (render/shadow.zig)
//!   --ss N         antialiasing, N×N rays per pixel    (default 2; 1 = off)
//!   --sigma-cut S  zero emission where b²/ρ > S        (default 1; 0 = off)
//!   --floor-cut F  zero emission where ρ < F × the atmosphere floor
//!                  profile RHOATMMIN·(r/2)^-1.5        (default 1e3; 0 = off)
//!   --gamma G      display stretch exponent            (default 0.65)
//!   --wp PCT       white-point percentile              (default 99.8)
//!   --blur PX      Gaussian beam blur sigma, pixels    (default 0 = off)
//!   --eps E        geodesic step, cells                (default 0.5)
//!   --tau T        optical-depth cutoff                (default 30)
//!   --threads N    render threads                      (default: CPU count)
//!
//! The σ- and floor-cuts are the standard GRMHD-imaging hygiene: funnel and
//! atmosphere cells sit at density floors, so their (huge, fake) temperatures
//! must not glow. In the t≈151 sgra_spin dump the ρ/floor histogram is
//! cleanly bimodal (atmosphere ≲ 10×, torus ≳ 10⁷×), so any F in 10³..10⁶
//! selects the same physical material.
//!
//! e.g.  zig build kdmp2png -Doptimize=ReleaseFast
//!       ./zig-out/bin/kdmp2png koral/problems/puffy/puffy3d_sgra_spin.toml \
//!           dumps/puffy3d_sgra_spin/prims00200.kdmp sgra.png --fov 80 --blur 6

const std = @import("std");
const koral = @import("koral");

const cfg = koral.config.puffy;
const puffy = koral.problems.puffy;
const render = koral.render;
const image = koral.render.image;
const metric = koral.metric.core;

fn parseF(s: []const u8, what: []const u8) !f64 {
    return std.fmt.parseFloat(f64, s) catch {
        std.debug.print("kdmp2png: bad value '{s}' for {s}\n", .{ s, what });
        return error.BadArgs;
    };
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    var args = std.process.Args.Iterator.init(init.minimal.args);
    defer args.deinit();
    _ = args.next(); // program name

    var params_path: ?[]const u8 = null;
    var kdmp_path: ?[]const u8 = null;
    var out_path: ?[]const u8 = null;
    var size: usize = 512;
    var fov: f64 = 100.0;
    var incl: f64 = 60.0;
    var cam_phi: f64 = 0.0;
    var rcam: f64 = 1000.0;
    var ss: usize = 2;
    var sigma_cut: f64 = 1.0;
    var floor_cut: f64 = 1.0e3;
    var nu_ghz: f64 = 0.0;
    var dist_kpc: f64 = 0.0;
    var screen = false;
    var gamma: f64 = 0.65;
    var wp_pct: f64 = 99.8;
    var blur: f64 = 0.0;
    var eps: f64 = 0.5;
    var tau_max: f64 = 30.0;
    var nthreads: usize = std.Thread.getCpuCount() catch 8;

    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--size")) {
            const v = args.next() orelse return usageErr();
            size = std.fmt.parseInt(usize, v, 10) catch return usageErr();
        } else if (std.mem.eql(u8, arg, "--threads")) {
            const v = args.next() orelse return usageErr();
            nthreads = std.fmt.parseInt(usize, v, 10) catch return usageErr();
        } else if (std.mem.eql(u8, arg, "--fov")) {
            fov = try parseF(args.next() orelse return usageErr(), "--fov");
        } else if (std.mem.eql(u8, arg, "--incl")) {
            incl = try parseF(args.next() orelse return usageErr(), "--incl");
        } else if (std.mem.eql(u8, arg, "--phi")) {
            cam_phi = try parseF(args.next() orelse return usageErr(), "--phi");
        } else if (std.mem.eql(u8, arg, "--rcam")) {
            rcam = try parseF(args.next() orelse return usageErr(), "--rcam");
        } else if (std.mem.eql(u8, arg, "--ss")) {
            const v = args.next() orelse return usageErr();
            ss = std.fmt.parseInt(usize, v, 10) catch return usageErr();
        } else if (std.mem.eql(u8, arg, "--sigma-cut")) {
            sigma_cut = try parseF(args.next() orelse return usageErr(), "--sigma-cut");
        } else if (std.mem.eql(u8, arg, "--floor-cut")) {
            floor_cut = try parseF(args.next() orelse return usageErr(), "--floor-cut");
        } else if (std.mem.eql(u8, arg, "--nu")) {
            nu_ghz = try parseF(args.next() orelse return usageErr(), "--nu");
        } else if (std.mem.eql(u8, arg, "--dist")) {
            dist_kpc = try parseF(args.next() orelse return usageErr(), "--dist");
        } else if (std.mem.eql(u8, arg, "--screen")) {
            screen = true;
        } else if (std.mem.eql(u8, arg, "--gamma")) {
            gamma = try parseF(args.next() orelse return usageErr(), "--gamma");
        } else if (std.mem.eql(u8, arg, "--wp")) {
            wp_pct = try parseF(args.next() orelse return usageErr(), "--wp");
        } else if (std.mem.eql(u8, arg, "--blur")) {
            blur = try parseF(args.next() orelse return usageErr(), "--blur");
        } else if (std.mem.eql(u8, arg, "--eps")) {
            eps = try parseF(args.next() orelse return usageErr(), "--eps");
        } else if (std.mem.eql(u8, arg, "--tau")) {
            tau_max = try parseF(args.next() orelse return usageErr(), "--tau");
        } else if (std.mem.startsWith(u8, arg, "--")) {
            std.debug.print("kdmp2png: unknown option '{s}'\n", .{arg});
            return usageErr();
        } else if (params_path == null) {
            params_path = arg;
        } else if (kdmp_path == null) {
            kdmp_path = arg;
        } else if (out_path == null) {
            out_path = arg;
        } else {
            return usageErr();
        }
    }
    const ppath = params_path orelse return usageErr();
    const kpath = kdmp_path orelse return usageErr();

    // default output name: <dump-stem>.png next to the dump
    var opath_buf: [1024]u8 = undefined;
    const opath = out_path orelse blk: {
        const stem = if (std.mem.lastIndexOfScalar(u8, kpath, '.')) |dot| kpath[0..dot] else kpath;
        break :blk try std.fmt.bufPrint(&opath_buf, "{s}.png", .{stem});
    };

    var p = koral.Params.load(allocator, io, ppath) catch |err| {
        std.debug.print("kdmp2png: cannot load params from '{s}': {s}\n", .{ ppath, @errorName(err) });
        return err;
    };
    defer p.deinit(allocator);

    // The driver's exact setup order (main.zig / kdmp2silo.zig): scales
    // first, then physics overrides (they can move mksr0, which rmin and
    // the grid read), then the domain and opacity table.
    puffy.mass = p.mass;
    puffy.mp.a = p.bhspin;
    puffy.applyPhysicsOverrides(&p);
    puffy.rmin = if (p.rmin > 0.0) p.rmin else puffy.rminForSpin(p.bhspin);
    if (p.rmax > 0.0) puffy.rmax = p.rmax;

    var mtab: ?koral.physics.mesa.MesaTable = null;
    defer if (mtab) |*t| t.deinit();
    if (p.mesa_table.len > 0) {
        mtab = koral.physics.mesa.MesaTable.load(allocator, io, p.mesa_table) catch |err| {
            std.debug.print("kdmp2png: cannot load MESA table '{s}': {s}\n", .{ p.mesa_table, @errorName(err) });
            return err;
        };
        puffy.channels.mesa = &mtab.?;
    }

    const bytes = std.Io.Dir.cwd().readFileAlloc(io, kpath, allocator, .limited(1 << 32)) catch |err| {
        std.debug.print("kdmp2png: cannot read '{s}': {s}\n", .{ kpath, @errorName(err) });
        return err;
    };
    defer allocator.free(bytes);

    var data = render.DumpData.fromBytes(allocator, bytes) catch |err| {
        std.debug.print("kdmp2png: cannot parse '{s}': {s}\n", .{ kpath, @errorName(err) });
        return err;
    };
    defer data.deinit(allocator);
    const h = data.header;
    if (h.nv != koral.VarLayout(cfg).count) {
        std.debug.print("kdmp2png: dump has nv={d}, puffy config expects {d}\n", .{ h.nv, koral.VarLayout(cfg).count });
        return error.DimMismatch;
    }

    const grid = puffy.makeGridNz(h.nx, h.ny, h.nz);
    var scene = render.Scene.init(
        grid,
        puffy.mp,
        puffy.consts(),
        puffy.channels,
        puffy.gam,
        &data,
        rcam,
        puffy.rmax,
    );
    scene.sigma_cut = sigma_cut;
    if (floor_cut > 0) {
        // PUFFY's floor atmosphere (setHdAtmosphere): ρ = RHOATMMIN·(r/2)^-1.5
        scene.floor = .{ .rho0 = puffy.rhoatmmin, .r0 = 2.0, .power = -1.5, .factor = floor_cut };
    }

    var cam = render.Camera{
        .r = rcam,
        .incl_deg = incl,
        .phi = cam_phi * std.math.pi / 180.0,
        .fov = fov,
        .width = size,
        .height = size,
        .ss = @max(ss, 1),
    };
    cam.setup(scene.mp);

    std.debug.print(
        "kdmp2png: {s} ({d}x{d}x{d}, t={e:.4}) | a={d:.4} M={e:.3} Msun | cam r={d} incl={d} fov={d}M | {d}px ss{d} sigma-cut={d} floor-cut={e:.0} {d} threads\n",
        .{ kpath, h.nx, h.ny, h.nz, h.t, puffy.mp.a, puffy.mass, rcam, incl, fov, size, ss, sigma_cut, floor_cut, nthreads },
    );

    const img = try allocator.alloc(f64, size * size);
    defer allocator.free(img);

    const nu_obs = nu_ghz * 1.0e9;
    const t0 = std.Io.Timestamp.now(io, .awake);
    render.renderImage(cfg, &scene, &cam, img, .{ .eps = eps, .tau_max = tau_max, .nu_obs = nu_obs, .screen = screen }, nthreads);
    const dur = t0.durationTo(std.Io.Timestamp.now(io, .awake));
    const secs = @as(f64, @floatFromInt(@as(i64, @intCast(dur.nanoseconds)))) / 1e9;

    var max_i: f64 = 0;
    var sum_i: f64 = 0;
    var lit: usize = 0;
    for (img) |v| {
        if (v > 0) lit += 1;
        if (v > max_i) max_i = v;
        sum_i += v;
    }

    // With --nu the traced values are I_ν at the camera in CGS — report the
    // peak brightness temperature (RJ: T_b = I_ν c²/2kν²) and, given a
    // distance, the integrated flux density.
    if (nu_obs > 0) {
        const em = koral.render.emission;
        const tb_max = max_i * em.c_cgs * em.c_cgs / (2.0 * em.k_cgs * nu_obs * nu_obs);
        std.debug.print("kdmp2png: nu={d} GHz  I_max={e:.3} erg/s/cm2/sr/Hz  T_b,max={e:.3} K\n", .{ nu_ghz, max_i, tb_max });
        if (dist_kpc > 0) {
            const d_cm = dist_kpc * 3.086e21;
            const pix_rad = fov * scene.consts.units.masscm / d_cm / @as(f64, @floatFromInt(size));
            const s_jy = sum_i * pix_rad * pix_rad * 1.0e23;
            std.debug.print("kdmp2png: S_nu({d} GHz) = {e:.3} Jy at {d} kpc\n", .{ nu_ghz, s_jy, dist_kpc });
        }
    }

    try image.gaussianBlur(allocator, img, size, size, blur);
    const wp = try image.whitePoint(allocator, img, wp_pct);
    if (wp <= 0) {
        std.debug.print("kdmp2png: image is entirely dark — check --fov/--incl (or the dump)\n", .{});
    }
    image.stretch(img, wp, gamma);
    const rgb = try image.colorize(allocator, img, null);
    defer allocator.free(rgb);

    // --screen: overlay the analytic Bardeen critical curve in green. The
    // image-plane map is pixel = (0.5 + α/fov)·W (our small-angle camera:
    // a pixel's asymptotic impact parameter is angle·r_cam, and the O(M/r)
    // finite-distance error is far sub-pixel for rcam ≳ 1000).
    if (screen) {
        const sh = koral.render.shadow;
        const a = puffy.mp.a;
        const incl_rad = std.math.clamp(incl * std.math.pi / 180.0, 0.05, std.math.pi - 0.05);
        const mark = struct {
            fn dot(rgbv: []u8, n: usize, alpha_m: f64, beta_m: f64, fov_m: f64) void {
                const nf: f64 = @floatFromInt(n);
                const fx = (0.5 + alpha_m / fov_m) * nf;
                const fy = (0.5 - beta_m / fov_m) * nf;
                if (fx < 0 or fy < 0 or fx >= nf or fy >= nf) return;
                const idx = 3 * (@as(usize, @intFromFloat(fy)) * n + @as(usize, @intFromFloat(fx)));
                rgbv[idx] = 30;
                rgbv[idx + 1] = 255;
                rgbv[idx + 2] = 80;
            }
        };
        if (@abs(a) > 1e-6) {
            const shell = sh.photonShell(a);
            var i: usize = 0;
            const n_samp = 20000;
            while (i < n_samp) : (i += 1) {
                const f = (@as(f64, @floatFromInt(i)) + 0.5) / n_samp;
                const rp = shell.r_pro + f * (shell.r_retro - shell.r_pro);
                const cp = sh.curvePoint(rp, a, incl_rad) orelse continue;
                mark.dot(rgb, size, cp.alpha, cp.beta, fov);
                mark.dot(rgb, size, cp.alpha, -cp.beta, fov);
            }
            std.debug.print("kdmp2png: screen mode — Bardeen curve overlaid (a={d}, r_ph {d:.4}..{d:.4})\n", .{ a, shell.r_pro, shell.r_retro });
        } else {
            // a = 0: the shadow is the b = √27 circle at any inclination
            var i: usize = 0;
            const n_samp = 20000;
            while (i < n_samp) : (i += 1) {
                const t = 2.0 * std.math.pi * @as(f64, @floatFromInt(i)) / n_samp;
                const b = @sqrt(27.0);
                mark.dot(rgb, size, b * @cos(t), b * @sin(t), fov);
            }
            std.debug.print("kdmp2png: screen mode — Schwarzschild b=sqrt(27) circle overlaid\n", .{});
        }
    }
    const png = try image.encodePng(allocator, @intCast(size), @intCast(size), rgb);
    defer allocator.free(png);

    std.Io.Dir.cwd().writeFile(io, .{ .sub_path = opath, .data = png }) catch |err| {
        std.debug.print("kdmp2png: cannot write '{s}': {s}\n", .{ opath, @errorName(err) });
        return err;
    };
    std.debug.print(
        "kdmp2png: {s} written ({d:.1}s render, {d:.1}% lit, I_max={e:.3}, wp={e:.3})\n",
        .{ opath, secs, 100.0 * @as(f64, @floatFromInt(lit)) / @as(f64, @floatFromInt(img.len)), max_i, wp },
    );
}

fn usageErr() error{BadArgs} {
    std.debug.print(
        "usage: kdmp2png <params.toml> <file.kdmp> [out.png]\n" ++
            "       [--size N] [--fov M] [--incl DEG] [--phi DEG] [--rcam M]\n" ++
            "       [--ss N] [--sigma-cut S] [--floor-cut F] [--nu GHZ] [--dist KPC] [--screen]\n" ++
            "       [--gamma G] [--wp PCT] [--blur PX] [--eps E] [--tau T] [--threads N]\n",
        .{},
    );
    return error.BadArgs;
}
