//! kdmp2png: GRRT-render a PUFFY KDMP snapshot into a PNG image: null
//! geodesics through the run's own Kerr/MKS2 metric + frequency-integrated
//! radiative transfer using the run's own opacities and M1 radiation field
//! (see koral/render/render.zig for the method).
//!
//! The sim configuration is rebuilt from the SAME params file and the SAME
//! code path as the puffy driver and kdmp2silo, so the grid, metric, thermo
//! constants and opacity channels match what produced the checkpoint.
//!
//! usage: kdmp2png <params.toml> <file.kdmp> [out.png] [options]
//!   slow light: kdmp2png <params.toml> --slow DIR [ref.kdmp] [out.png] [options]
//!
//!   --slow DIR     SLOW-LIGHT mode: sample the KDMP series in DIR at each
//!                  photon's own retarded coordinate time x⁰ (the geodesic
//!                  integrator carries it exactly; gravitational time
//!                  delay, Shapiro delay and photon-ring lags included),
//!                  interpolating linearly between bracketing frames. The
//!                  sweep streams the series once, newest→oldest, keeping
//!                  ≤ 2 frames + the reference resident (render/sweep.zig).
//!                  An optional ref.kdmp positional overrides the reference
//!                  frame used for the static zone r > rslow (default: the
//!                  series frame nearest --tobs).
//!   --tobs T       sim time imaged at the hole [M]: photons arrive at the
//!                  camera at t = T + rcam (default: t_last − rslow, the
//!                  latest T whose near side needs no extrapolation)
//!   --stride N     use every Nth series frame, counted from the newest
//!                  (default 1; the sgra_spin cadence of 0.3-0.8 M is much
//!                  finer than the flow evolves; 4-8 is plenty)
//!   --rslow R      treat the flow as static outside this radius [M]
//!                  (default 40; the sampled time range inside spans about
//!                  [T − R − windings, T + R])
//!   --adapt D      adaptive photon-ring refinement: a vacuum pre-pass
//!                  (geodesics only, no transfer) finds pixels straddling
//!                  the critical curve; capture flips or corner flight
//!                  times differing by > --adapt-dt, and subdivides them
//!                  as a quadtree to depth D (default 0 = off; 6-10 gives
//!                  sub-pixel photon-ring sampling; works in both modes)
//!   --adapt-dt M   corner flight-time disagreement that triggers
//!                  refinement (default 5 M; a photon half-orbit is ~16 M)
//!
//!   --fits PATH    also write the UNPROCESSED image (no blur/stretch) as
//!                  a FITS file in Jy/pixel with ehtim-compatible headers
//!                  (render/fits.zig); the entry point to the EHT
//!                  synthetic-observation pipeline (eht-imaging, SYMBA).
//!                  Requires --nu and --dist (physical units).
//!   --ra DEG       source RA for the FITS header  (default Sgr A*)
//!   --dec DEG      source Dec for the FITS header (default Sgr A*)
//!   --mjd D        observation MJD for the FITS header (default 57850,
//!                  2017 Apr 7; the EHT Sgr A* campaign)
//!
//!   --size N       image width and height in pixels   (default 512)
//!   --fov M        field of view at the hole, in GM/c² (default 100)
//!   --incl DEG     inclination from the spin axis      (default 60)
//!   --phi DEG      camera azimuth                      (default 0)
//!   --rcam M       camera radius in GM/c²              (default 1000)
//!   --nu GHZ       observed frequency: monochromatic synthetic image at
//!                  ν_obs (thermal synchrotron + free-free + scattering,
//!                  emission.zig) with I_ν reported in CGS and brightness
//!                  temperature: 0 = gray bolometric      (default 0)
//!   --dist KPC     source distance; with --nu, also print the integrated
//!                  flux density in Jy (0 = skip)         (default 0)
//!   --screen       VALIDATION mode: ignore the fluid, render a bright
//!                  celestial sphere (captured rays dark) and overlay the
//!                  analytic Bardeen shadow curve in green; the dark
//!                  boundary must hug the curve (render/shadow.zig)
//!   --ss N         antialiasing, N×N rays per pixel    (default 2; 1 = off)
//!   --sigma-cut S  zero emission where b²/ρ > S        (default 1; 0 = off)
//!   --floor-cut F  zero emission where ρ < F × the atmosphere floor
//!                  profile RHOATMMIN·(r/2)^-1.5        (default 1e3; 0 = off)
//!   --gamma G      display stretch exponent            (default 0.65)
//!   --wp PCT       white-point percentile              (default 99.8)
//!   --blur PX      Gaussian beam blur sigma, pixels    (default 0 = off)
//!   --eps E        geodesic step, cells                (default 0.5)
//!   --max-steps N  per-ray step budget (default 200000). Screen-mode
//!                  deep-adapt renders want ~15000: with no absorber, rays
//!                  pinned to the photon shell otherwise orbit out the
//!                  whole budget (matter stops them via tau instead)
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
    var max_steps: usize = 200_000;
    var nthreads: usize = std.Thread.getCpuCount() catch 8;
    var slow_dir: ?[]const u8 = null;
    var tobs: f64 = -1.0; // < 0 = auto (t_last − rslow)
    var stride: usize = 1;
    var rslow: f64 = 40.0;
    var adapt: usize = 0;
    var adapt_dt: f64 = 5.0;
    var fits_path: ?[]const u8 = null;
    var ra_deg: f64 = 266.4168371; // Sgr A* J2000
    var dec_deg: f64 = -29.0078106;
    var mjd: f64 = 57850.0;

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
        } else if (std.mem.eql(u8, arg, "--max-steps")) {
            const v = args.next() orelse return usageErr();
            max_steps = std.fmt.parseInt(usize, v, 10) catch return usageErr();
        } else if (std.mem.eql(u8, arg, "--slow")) {
            slow_dir = args.next() orelse return usageErr();
        } else if (std.mem.eql(u8, arg, "--tobs")) {
            tobs = try parseF(args.next() orelse return usageErr(), "--tobs");
        } else if (std.mem.eql(u8, arg, "--stride")) {
            const v = args.next() orelse return usageErr();
            stride = std.fmt.parseInt(usize, v, 10) catch return usageErr();
        } else if (std.mem.eql(u8, arg, "--rslow")) {
            rslow = try parseF(args.next() orelse return usageErr(), "--rslow");
        } else if (std.mem.eql(u8, arg, "--adapt")) {
            const v = args.next() orelse return usageErr();
            adapt = std.fmt.parseInt(usize, v, 10) catch return usageErr();
        } else if (std.mem.eql(u8, arg, "--adapt-dt")) {
            adapt_dt = try parseF(args.next() orelse return usageErr(), "--adapt-dt");
        } else if (std.mem.eql(u8, arg, "--fits")) {
            fits_path = args.next() orelse return usageErr();
        } else if (std.mem.eql(u8, arg, "--ra")) {
            ra_deg = try parseF(args.next() orelse return usageErr(), "--ra");
        } else if (std.mem.eql(u8, arg, "--dec")) {
            dec_deg = try parseF(args.next() orelse return usageErr(), "--dec");
        } else if (std.mem.eql(u8, arg, "--mjd")) {
            mjd = try parseF(args.next() orelse return usageErr(), "--mjd");
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

    // Slow mode makes the kdmp positional optional (it becomes a reference-
    // frame OVERRIDE): a lone non-.kdmp second positional is the out path.
    if (slow_dir != null) {
        if (kdmp_path) |k| {
            if (!std.mem.endsWith(u8, k, ".kdmp")) {
                if (out_path != null) return usageErr();
                out_path = k;
                kdmp_path = null;
            }
        }
    }
    const kpath: ?[]const u8 = kdmp_path;
    if (slow_dir == null and kpath == null) return usageErr();

    if (fits_path != null and (nu_ghz <= 0 or dist_kpc <= 0 or screen)) {
        std.debug.print("kdmp2png: --fits needs physical units: give --nu and --dist (and not --screen)\n", .{});
        return error.BadArgs;
    }

    // default output name: <dump-stem>.png next to the dump; the slow-mode
    // default needs the resolved t_obs, so it is derived in the branch below
    var opath_buf: [1024]u8 = undefined;
    var opath: []const u8 = out_path orelse "";
    if (opath.len == 0 and slow_dir == null) {
        const kp = kpath.?;
        const stem = if (std.mem.lastIndexOfScalar(u8, kp, '.')) |dot| kp[0..dot] else kp;
        opath = try std.fmt.bufPrint(&opath_buf, "{s}.png", .{stem});
    }

    var p = koral.Params.load(allocator, io, ppath) catch |err| {
        std.debug.print("kdmp2png: cannot load params from '{s}': {s}\n", .{ ppath, @errorName(err) });
        return err;
    };
    defer p.deinit(allocator);

    var loaded = puffy.load(allocator, io, &p) catch |err| {
        std.debug.print("kdmp2png: cannot load physics from '{s}': {s}\n", .{ ppath, @errorName(err) });
        return err;
    };
    defer loaded.deinit(allocator);
    const phys = &loaded.physics;

    const Load = struct {
        fn kdmp(alloc: std.mem.Allocator, io_: std.Io, path: []const u8) !render.DumpData {
            const bytes = std.Io.Dir.cwd().readFileAlloc(io_, path, alloc, .limited(1 << 33)) catch |err| {
                std.debug.print("kdmp2png: cannot read '{s}': {s}\n", .{ path, @errorName(err) });
                return err;
            };
            defer alloc.free(bytes);
            return render.DumpData.fromBytes(alloc, bytes) catch |err| {
                std.debug.print("kdmp2png: cannot parse '{s}': {s}\n", .{ path, @errorName(err) });
                return err;
            };
        }
    };

    var cam = render.Camera{
        .r = rcam,
        .incl_deg = incl,
        .phi = cam_phi * std.math.pi / 180.0,
        .fov = fov,
        .width = size,
        .height = size,
        .ss = @max(ss, 1),
    };
    cam.setup(phys.mp);

    const img = try allocator.alloc(f64, size * size);
    defer allocator.free(img);
    const nu_obs = nu_ghz * 1.0e9;
    const topts = render.TraceOpts{ .eps = eps, .tau_max = tau_max, .max_steps = max_steps, .nu_obs = nu_obs, .screen = screen };
    const nvar = koral.VarLayout(cfg).count;
    var secs: f64 = 0;

    if (slow_dir) |sdir| {
        // ---- slow light: retarded-time sampling of the whole series ----
        var ser = render.series.Series.scan(allocator, io, sdir, stride) catch |err| {
            std.debug.print("kdmp2png: cannot scan series '{s}': {s}\n", .{ sdir, @errorName(err) });
            return err;
        };
        defer ser.deinit(allocator);
        if (ser.shape.nv != nvar) {
            std.debug.print("kdmp2png: series has nv={d}, puffy config expects {d}\n", .{ ser.shape.nv, nvar });
            return error.DimMismatch;
        }
        const t_first = ser.ts[0];
        const t_last = ser.ts[ser.ts.len - 1];
        if (tobs < 0) tobs = @max(t_first, t_last - rslow);
        if (opath.len == 0) {
            opath = try std.fmt.bufPrint(&opath_buf, "{s}/slow_t{d:.0}.png", .{ sdir, tobs });
        }

        const grid = phys.makeGridNz(ser.shape.nx, ser.shape.ny, ser.shape.nz);
        var src = try render.series.FileSource.init(allocator, io, &ser);
        defer src.deinit();

        // reference frame for the static zone: explicit override, or the
        // series frame nearest t_obs
        var ref_owned: ?render.DumpData = null;
        defer if (ref_owned) |*d| d.deinit(allocator);
        var ref_idx: usize = 0;
        var ref_acquired = false;
        const ref: *const render.DumpData = if (kpath) |kp| blk: {
            ref_owned = try Load.kdmp(allocator, io, kp);
            const rh = ref_owned.?.header;
            if (rh.nx != ser.shape.nx or rh.ny != ser.shape.ny or rh.nz != ser.shape.nz or rh.nv != ser.shape.nv) {
                std.debug.print("kdmp2png: reference dump '{s}' shape differs from the series\n", .{kp});
                return error.DimMismatch;
            }
            break :blk &ref_owned.?;
        } else blk: {
            ref_idx = ser.nearest(tobs);
            ref_acquired = true;
            break :blk try src.acquire(ref_idx);
        };
        defer if (ref_acquired) src.release(ref_idx);

        var scene = render.Scene.init(grid, phys.mp, phys.consts(), phys.channels, phys.gam, ref, rcam, phys.rmax);
        scene.scattering = phys.scattering;
        scene.sigma_cut = sigma_cut;
        if (floor_cut > 0) {
            scene.floor = .{ .rho0 = phys.rhoatmmin, .r0 = 2.0, .power = -1.5, .factor = floor_cut };
        }

        std.debug.print(
            "kdmp2png: SLOW LIGHT {s}: {d} frames ({d}x{d}x{d}), t=[{d:.2}, {d:.2}] M, stride {d} | t_obs={d:.2} (arrival t={d:.2}) r_slow={d} | a={d:.4} M={e:.3} Msun | cam r={d} incl={d} fov={d}M {d}px ss{d} {d} threads\n",
            .{ sdir, ser.ts.len, ser.shape.nx, ser.shape.ny, ser.shape.nz, t_first, t_last, stride, tobs, tobs + rcam, rslow, phys.mp.a, phys.mass, rcam, incl, fov, size, ss, nthreads },
        );

        var plan_specs: []render.sweep.RaySpec = &.{};
        var plan: ?render.adaptive.Plan = null;
        defer if (plan) |*pl| pl.deinit(allocator) else allocator.free(plan_specs);
        if (adapt > 0) {
            plan = try render.adaptive.planRays(cfg, allocator, &scene, &cam, topts, .{ .depth = adapt, .dt_thresh = adapt_dt }, nthreads);
            plan_specs = plan.?.specs;
            var nmark: usize = 0;
            for (plan.?.marked) |m| nmark += @intFromBool(m);
            std.debug.print("kdmp2png: adaptive plan: {d} rays ({d} px refined to depth {d}, {d} vacuum probes)\n", .{ plan_specs.len, nmark, adapt, plan.?.vacuum_rays });
        } else {
            plan_specs = try render.sweep.uniformPlan(allocator, &cam);
        }

        const t0 = std.Io.Timestamp.now(io, .awake);
        const stats = try render.sweep.renderSlow(cfg, allocator, &scene, &cam, &src, plan_specs, img, topts, .{ .t_cam = tobs + rcam, .r_slow = rslow }, nthreads, true);
        const dur = t0.durationTo(std.Io.Timestamp.now(io, .awake));
        secs = @as(f64, @floatFromInt(@as(i64, @intCast(dur.nanoseconds)))) / 1e9;
        std.debug.print(
            "kdmp2png: sweep: {d} rounds, {d} frame loads | rays: {d} entered slow zone, {d} tails, {d} captured, {d} thick | series-edge holds: {d} samples late, {d} early, {d} rays past start\n",
            .{ stats.rounds, src.loads, stats.entered, stats.tails, stats.captured, stats.thick, stats.clamped_above, stats.clamped_below, stats.past_start },
        );
    } else {
        // ---- fast light: one snapshot everywhere (the original path) ----
        var data = try Load.kdmp(allocator, io, kpath.?);
        defer data.deinit(allocator);
        const h = data.header;
        if (h.nv != nvar) {
            std.debug.print("kdmp2png: dump has nv={d}, puffy config expects {d}\n", .{ h.nv, nvar });
            return error.DimMismatch;
        }

        const grid = phys.makeGridNz(h.nx, h.ny, h.nz);
        var scene = render.Scene.init(
            grid,
            phys.mp,
            phys.consts(),
            phys.channels,
            phys.gam,
            &data,
            rcam,
            phys.rmax,
        );
        scene.scattering = phys.scattering;
        scene.sigma_cut = sigma_cut;
        if (floor_cut > 0) {
            // PUFFY's floor atmosphere (setHdAtmosphere): ρ = RHOATMMIN·(r/2)^-1.5
            scene.floor = .{ .rho0 = phys.rhoatmmin, .r0 = 2.0, .power = -1.5, .factor = floor_cut };
        }

        std.debug.print(
            "kdmp2png: {s} ({d}x{d}x{d}, t={e:.4}) | a={d:.4} M={e:.3} Msun | cam r={d} incl={d} fov={d}M | {d}px ss{d} sigma-cut={d} floor-cut={e:.0} {d} threads\n",
            .{ kpath.?, h.nx, h.ny, h.nz, h.t, phys.mp.a, phys.mass, rcam, incl, fov, size, ss, sigma_cut, floor_cut, nthreads },
        );

        const t0 = std.Io.Timestamp.now(io, .awake);
        if (adapt > 0) {
            var plan = try render.adaptive.planRays(cfg, allocator, &scene, &cam, topts, .{ .depth = adapt, .dt_thresh = adapt_dt }, nthreads);
            defer plan.deinit(allocator);
            var nmark: usize = 0;
            for (plan.marked) |m| nmark += @intFromBool(m);
            std.debug.print("kdmp2png: adaptive plan: {d} rays ({d} px refined to depth {d}, {d} vacuum probes)\n", .{ plan.specs.len, nmark, adapt, plan.vacuum_rays });
            try render.adaptive.renderPlan(cfg, allocator, &scene, &cam, plan.specs, img, topts, nthreads);
        } else {
            render.renderImage(cfg, &scene, &cam, img, topts, nthreads);
        }
        const dur = t0.durationTo(std.Io.Timestamp.now(io, .awake));
        secs = @as(f64, @floatFromInt(@as(i64, @intCast(dur.nanoseconds)))) / 1e9;
    }

    // ---- FITS export: raw I_nu -> Jy/pixel, before any display processing ----
    if (fits_path) |fpath| {
        const masscm = phys.consts().units.masscm;
        const d_cm = dist_kpc * 3.086e21;
        const pix_rad = fov * masscm / d_cm / @as(f64, @floatFromInt(size));
        const scale = pix_rad * pix_rad * 1.0e23; // I_nu [cgs] -> Jy/pixel
        const jy = try allocator.alloc(f64, img.len);
        defer allocator.free(jy);
        var s_total: f64 = 0;
        for (jy, img) |*o, v| {
            o.* = v * scale;
            s_total += o.*;
        }
        const fbytes = try render.fits.encode(allocator, jy, size, size, .{
            .cdelt_deg = pix_rad * 180.0 / std.math.pi,
            .ra_deg = ra_deg,
            .dec_deg = dec_deg,
            .freq_hz = nu_obs,
            .mjd = mjd,
        });
        defer allocator.free(fbytes);
        std.Io.Dir.cwd().writeFile(io, .{ .sub_path = fpath, .data = fbytes }) catch |err| {
            std.debug.print("kdmp2png: cannot write '{s}': {s}\n", .{ fpath, @errorName(err) });
            return err;
        };
        std.debug.print("kdmp2png: {s} written ({d}x{d}, {d:.3} uas/px, S_nu = {e:.3} Jy)\n", .{ fpath, size, size, pix_rad * 180.0 / std.math.pi * 3.6e9, s_total });
    }

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
            const pix_rad = fov * phys.consts().units.masscm / d_cm / @as(f64, @floatFromInt(size));
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
        const a = phys.mp.a;
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
            "       kdmp2png <params.toml> --slow DIR [ref.kdmp] [out.png]   (slow light)\n" ++
            "       [--size N] [--fov M] [--incl DEG] [--phi DEG] [--rcam M]\n" ++
            "       [--ss N] [--sigma-cut S] [--floor-cut F] [--nu GHZ] [--dist KPC] [--screen]\n" ++
            "       [--gamma G] [--wp PCT] [--blur PX] [--eps E] [--tau T] [--max-steps N] [--threads N]\n" ++
            "       [--slow DIR] [--tobs T] [--stride N] [--rslow R] [--adapt D] [--adapt-dt M]\n" ++
            "       [--fits PATH] [--ra DEG] [--dec DEG] [--mjd D]\n",
        .{},
    );
    return error.BadArgs;
}
