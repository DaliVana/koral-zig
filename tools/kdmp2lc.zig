//! kdmp2lc — slow-light 230 GHz (or any-ν) LIGHT CURVE of a KDMP series:
//! the EHT variability observable (Sgr A* Paper V's modulation-index
//! criterion — the constraint most GRMHD models fail).
//!
//! All observation epochs are batched into ONE streaming sweep
//! (sweep.SlowOpts.t_cam_of): every epoch's rays ride the same sliding
//! two-frame window, so the whole light curve costs a single pass over the
//! series — N_epochs × the per-frame ray count, but each KDMP file is read
//! from disk exactly once.
//!
//! Output: one text row per epoch —
//!     idx   t_obs[M]   t_obs[s]   S_nu[Jy]   I_max[cgs]
//! plus mean flux, σ, and the modulation index M = σ/mean at the end.
//! t_obs[s] uses GM/c³ of the params mass, so the curve is directly
//! comparable to observed light-curve segments (for Sgr A*: 1 M = 21.2 s,
//! a 3-hour EHT window is ~510 M — mind the series span).
//!
//! usage: kdmp2lc <params.toml> --slow DIR [out.txt]
//!   --nu GHZ       observing frequency          (default 230)
//!   --dist KPC     source distance              (default 8.277, Sgr A*)
//!   --t0 M         first epoch t_obs            (default: t_first + rslow + 60,
//!                  past which the sweep needs no pre-series extrapolation
//!                  except deep photon-ring tails)
//!   --t1 M         last epoch t_obs             (default: t_last − rslow)
//!   --nt N         number of epochs             (default 32)
//!   --frames DIR   also write one PNG per epoch (movie frames, shared
//!                  white point across epochs so brightness is comparable)
//!   --size N --fov M --incl DEG --phi DEG --rcam M --ss N  (camera;
//!                  defaults 96 / 60 / 60 / 0 / 1000 / 1)
//!   --rslow R --stride N                        (slow-light; 40 / 1)
//!   --sigma-cut S --floor-cut F                 (imaging hygiene; 1 / 1e3)
//!   --eps E --tau T --max-steps N --threads N
//!   --gamma G --wp PCT                          (frames display; 0.65 / 99.8)
//!
//! CHOOSING --rslow: it must ENCLOSE the time-variable emission region —
//! everything outside samples one frozen reference frame, so variability
//! from out there is erased. For compact synchrotron sources (EHT-type,
//! emission within ~20 M) the default 40 is right; verified on the
//! sgra_spin dumps, whose 230 GHz flux is bulk free-free: rslow 40 gave
//! M ~ 1e-6 (inner region only) while rslow 200 recovered the true
//! ~3e-3/50 M secular rate — at the cost of a lookback that outruns the
//! series (watch the "rays past start" stat).
//!
//! CAVEAT the tool prints: a φ-wedge domain is tiled m-fold to 2π, so
//! azimuthal structure — and therefore variability — is artificially
//! m-periodic; treat wedge-run light-curve statistics as machinery
//! validation, not physics, until full-2π campaign data exists.

const std = @import("std");
const koral = @import("koral");

const cfg = koral.config.puffy;
const puffy = koral.problems.puffy;
const render = koral.render;
const image = koral.render.image;

fn parseF(s: []const u8, what: []const u8) !f64 {
    return std.fmt.parseFloat(f64, s) catch {
        std.debug.print("kdmp2lc: bad value '{s}' for {s}\n", .{ s, what });
        return error.BadArgs;
    };
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    var args = std.process.Args.Iterator.init(init.minimal.args);
    defer args.deinit();
    _ = args.next();

    var params_path: ?[]const u8 = null;
    var out_path: ?[]const u8 = null;
    var slow_dir: ?[]const u8 = null;
    var frames_dir: ?[]const u8 = null;
    var nu_ghz: f64 = 230.0;
    var dist_kpc: f64 = 8.277;
    var t0: f64 = -1.0;
    var t1: f64 = -1.0;
    var nt: usize = 32;
    var size: usize = 96;
    var fov: f64 = 60.0;
    var incl: f64 = 60.0;
    var cam_phi: f64 = 0.0;
    var rcam: f64 = 1000.0;
    var ss: usize = 1;
    var rslow: f64 = 40.0;
    var stride: usize = 1;
    var sigma_cut: f64 = 1.0;
    var floor_cut: f64 = 1.0e3;
    var eps: f64 = 0.5;
    var tau_max: f64 = 30.0;
    var max_steps: usize = 200_000;
    var gamma: f64 = 0.65;
    var wp_pct: f64 = 99.8;
    var nthreads: usize = std.Thread.getCpuCount() catch 8;

    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--slow")) {
            slow_dir = args.next() orelse return usageErr();
        } else if (std.mem.eql(u8, arg, "--frames")) {
            frames_dir = args.next() orelse return usageErr();
        } else if (std.mem.eql(u8, arg, "--nu")) {
            nu_ghz = try parseF(args.next() orelse return usageErr(), "--nu");
        } else if (std.mem.eql(u8, arg, "--dist")) {
            dist_kpc = try parseF(args.next() orelse return usageErr(), "--dist");
        } else if (std.mem.eql(u8, arg, "--t0")) {
            t0 = try parseF(args.next() orelse return usageErr(), "--t0");
        } else if (std.mem.eql(u8, arg, "--t1")) {
            t1 = try parseF(args.next() orelse return usageErr(), "--t1");
        } else if (std.mem.eql(u8, arg, "--nt")) {
            const v = args.next() orelse return usageErr();
            nt = std.fmt.parseInt(usize, v, 10) catch return usageErr();
        } else if (std.mem.eql(u8, arg, "--size")) {
            const v = args.next() orelse return usageErr();
            size = std.fmt.parseInt(usize, v, 10) catch return usageErr();
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
        } else if (std.mem.eql(u8, arg, "--rslow")) {
            rslow = try parseF(args.next() orelse return usageErr(), "--rslow");
        } else if (std.mem.eql(u8, arg, "--stride")) {
            const v = args.next() orelse return usageErr();
            stride = std.fmt.parseInt(usize, v, 10) catch return usageErr();
        } else if (std.mem.eql(u8, arg, "--sigma-cut")) {
            sigma_cut = try parseF(args.next() orelse return usageErr(), "--sigma-cut");
        } else if (std.mem.eql(u8, arg, "--floor-cut")) {
            floor_cut = try parseF(args.next() orelse return usageErr(), "--floor-cut");
        } else if (std.mem.eql(u8, arg, "--eps")) {
            eps = try parseF(args.next() orelse return usageErr(), "--eps");
        } else if (std.mem.eql(u8, arg, "--tau")) {
            tau_max = try parseF(args.next() orelse return usageErr(), "--tau");
        } else if (std.mem.eql(u8, arg, "--max-steps")) {
            const v = args.next() orelse return usageErr();
            max_steps = std.fmt.parseInt(usize, v, 10) catch return usageErr();
        } else if (std.mem.eql(u8, arg, "--gamma")) {
            gamma = try parseF(args.next() orelse return usageErr(), "--gamma");
        } else if (std.mem.eql(u8, arg, "--wp")) {
            wp_pct = try parseF(args.next() orelse return usageErr(), "--wp");
        } else if (std.mem.eql(u8, arg, "--threads")) {
            const v = args.next() orelse return usageErr();
            nthreads = std.fmt.parseInt(usize, v, 10) catch return usageErr();
        } else if (std.mem.startsWith(u8, arg, "--")) {
            std.debug.print("kdmp2lc: unknown option '{s}'\n", .{arg});
            return usageErr();
        } else if (params_path == null) {
            params_path = arg;
        } else if (out_path == null) {
            out_path = arg;
        } else {
            return usageErr();
        }
    }
    const ppath = params_path orelse return usageErr();
    const sdir = slow_dir orelse return usageErr();
    if (nt < 2) return usageErr();
    if (nu_ghz <= 0 or dist_kpc <= 0) {
        std.debug.print("kdmp2lc: needs physical units — positive --nu and --dist\n", .{});
        return error.BadArgs;
    }

    var p = koral.Params.load(allocator, io, ppath) catch |err| {
        std.debug.print("kdmp2lc: cannot load params from '{s}': {s}\n", .{ ppath, @errorName(err) });
        return err;
    };
    defer p.deinit(allocator);

    // the driver's exact setup order (see kdmp2png)
    puffy.mass = p.mass;
    puffy.mp.a = p.bhspin;
    puffy.applyPhysicsOverrides(&p);
    puffy.rmin = if (p.rmin > 0.0) p.rmin else puffy.rminForSpin(p.bhspin);
    if (p.rmax > 0.0) puffy.rmax = p.rmax;

    var mtab: ?koral.physics.mesa.MesaTable = null;
    defer if (mtab) |*t| t.deinit();
    if (p.mesa_table.len > 0) {
        mtab = koral.physics.mesa.MesaTable.load(allocator, io, p.mesa_table) catch |err| {
            std.debug.print("kdmp2lc: cannot load MESA table '{s}': {s}\n", .{ p.mesa_table, @errorName(err) });
            return err;
        };
        puffy.channels.mesa = &mtab.?;
    }

    var ser = render.series.Series.scan(allocator, io, sdir, stride) catch |err| {
        std.debug.print("kdmp2lc: cannot scan series '{s}': {s}\n", .{ sdir, @errorName(err) });
        return err;
    };
    defer ser.deinit(allocator);
    if (ser.shape.nv != koral.VarLayout(cfg).count) {
        std.debug.print("kdmp2lc: series has nv={d}, puffy config expects {d}\n", .{ ser.shape.nv, koral.VarLayout(cfg).count });
        return error.DimMismatch;
    }
    const t_first = ser.ts[0];
    const t_last = ser.ts[ser.ts.len - 1];
    if (t1 < 0) t1 = t_last - rslow;
    if (t0 < 0) t0 = @min(t_first + rslow + 60.0, t1);
    if (!(t1 > t0)) {
        std.debug.print("kdmp2lc: empty epoch range [{d:.2}, {d:.2}] — series span [{d:.2}, {d:.2}] too short for rslow={d}\n", .{ t0, t1, t_first, t_last, rslow });
        return error.BadArgs;
    }

    const grid = puffy.makeGridNz(ser.shape.nx, ser.shape.ny, ser.shape.nz);
    var src = try render.series.FileSource.init(allocator, io, &ser);
    defer src.deinit();

    // reference frame for the static zone: nearest the mid epoch
    const ref_idx = ser.nearest(0.5 * (t0 + t1));
    const ref = try src.acquire(ref_idx);
    defer src.release(ref_idx);

    var scene = render.Scene.init(grid, puffy.mp, puffy.consts(), puffy.channels, puffy.gam, ref, rcam, puffy.rmax);
    scene.scattering = puffy.scattering;
    scene.sigma_cut = sigma_cut;
    if (floor_cut > 0) {
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
    cam.setup(puffy.mp);

    // one spec set per epoch, each epoch aimed at its own pixel block
    const base = try render.sweep.uniformPlan(allocator, &cam);
    defer allocator.free(base);
    const npix = size * size;
    const specs = try allocator.alloc(render.sweep.RaySpec, base.len * nt);
    defer allocator.free(specs);
    const t_of = try allocator.alloc(f64, base.len * nt);
    defer allocator.free(t_of);
    const dt = (t1 - t0) / @as(f64, @floatFromInt(nt - 1));
    for (0..nt) |e| {
        const tc = t0 + dt * @as(f64, @floatFromInt(e)) + rcam;
        for (base, 0..) |sp, j| {
            var s_ = sp;
            s_.pix += @intCast(e * npix);
            specs[e * base.len + j] = s_;
            t_of[e * base.len + j] = tc;
        }
    }

    const un = puffy.consts().units;
    const gmc3 = un.gmc3();
    const wedge = 2.0 * std.math.pi / (grid.maxz - grid.minz);
    std.debug.print(
        "kdmp2lc: {s}: {d} frames, t=[{d:.2}, {d:.2}] M, stride {d} | {d} epochs t_obs=[{d:.2}, {d:.2}] M (dt={d:.2} M = {d:.1} s) | nu={d} GHz dist={d} kpc | {d}px fov={d}M incl={d} ss{d} rslow={d} | {d} rays, {d} threads\n",
        .{ sdir, ser.ts.len, t_first, t_last, stride, nt, t0, t1, dt, dt * gmc3, nu_ghz, dist_kpc, size, fov, incl, ss, rslow, specs.len, nthreads },
    );
    if (wedge > 1.5) {
        std.debug.print("kdmp2lc: NOTE phi-wedge domain tiled x{d:.0} — variability is artificially m={d:.0}-periodic; statistics are machinery-grade, not physics-grade\n", .{ wedge, wedge });
    }

    const out = try allocator.alloc(f64, npix * nt);
    defer allocator.free(out);
    const topts = render.TraceOpts{ .eps = eps, .tau_max = tau_max, .max_steps = max_steps, .nu_obs = nu_ghz * 1.0e9 };
    const tstart = std.Io.Timestamp.now(io, .awake);
    const stats = try render.sweep.renderSlow(cfg, allocator, &scene, &cam, &src, specs, out, topts, .{ .t_cam = 0, .r_slow = rslow, .t_cam_of = t_of }, nthreads, false);
    const dur = tstart.durationTo(std.Io.Timestamp.now(io, .awake));
    std.debug.print(
        "kdmp2lc: sweep done in {d:.1}s: {d} rounds, {d} frame loads | {d} entered, {d} tails, {d} captured | edge holds: {d} late, {d} early, {d} rays past start\n",
        .{ @as(f64, @floatFromInt(@as(i64, @intCast(dur.nanoseconds)))) / 1e9, stats.rounds, src.loads, stats.entered, stats.tails, stats.captured, stats.clamped_above, stats.clamped_below, stats.past_start },
    );

    // ---- per-epoch flux + the text table ----
    const masscm = un.masscm;
    const d_cm = dist_kpc * 3.086e21;
    const pix_rad = fov * masscm / d_cm / @as(f64, @floatFromInt(size));
    const jy_scale = pix_rad * pix_rad * 1.0e23;

    var text: std.ArrayList(u8) = .empty;
    defer text.deinit(allocator);
    var lbuf: [256]u8 = undefined;
    try text.appendSlice(allocator, try std.fmt.bufPrint(&lbuf, "# kdmp2lc {s} nu={d}GHz dist={d}kpc fov={d}M incl={d} size={d} rslow={d} stride={d}\n", .{ sdir, nu_ghz, dist_kpc, fov, incl, size, rslow, stride }));
    try text.appendSlice(allocator, "# idx  t_obs[M]  t_obs[s]  S_nu[Jy]  I_max[cgs]\n");

    const flux = try allocator.alloc(f64, nt);
    defer allocator.free(flux);
    var mean: f64 = 0;
    for (0..nt) |e| {
        const im = out[e * npix ..][0..npix];
        var s_: f64 = 0;
        var mx: f64 = 0;
        for (im) |v| {
            s_ += v;
            mx = @max(mx, v);
        }
        flux[e] = s_ * jy_scale;
        mean += flux[e];
        const te = t0 + dt * @as(f64, @floatFromInt(e));
        try text.appendSlice(allocator, try std.fmt.bufPrint(&lbuf, "{d}  {d:.4}  {d:.2}  {e:.6}  {e:.4}\n", .{ e, te, te * gmc3, flux[e], mx }));
    }
    mean /= @as(f64, @floatFromInt(nt));
    var varsum: f64 = 0;
    for (flux) |f| varsum += (f - mean) * (f - mean);
    const sigma = @sqrt(varsum / @as(f64, @floatFromInt(nt - 1)));
    const mi = if (mean > 0) sigma / mean else 0.0;
    try text.appendSlice(allocator, try std.fmt.bufPrint(&lbuf, "# mean = {e:.6} Jy  sigma = {e:.6} Jy  modulation index M = {d:.4} over {d:.1} M = {d:.1} min\n", .{ mean, sigma, mi, t1 - t0, (t1 - t0) * gmc3 / 60.0 }));

    if (out_path) |op| {
        std.Io.Dir.cwd().writeFile(io, .{ .sub_path = op, .data = text.items }) catch |err| {
            std.debug.print("kdmp2lc: cannot write '{s}': {s}\n", .{ op, @errorName(err) });
            return err;
        };
        std.debug.print("kdmp2lc: {s} written | mean = {e:.3} Jy, M = sigma/mean = {d:.4}\n", .{ op, mean, mi });
    } else {
        std.debug.print("{s}", .{text.items});
    }

    // ---- optional movie frames: shared white point across ALL epochs ----
    if (frames_dir) |fdir| {
        std.Io.Dir.cwd().createDirPath(io, fdir) catch {};
        const wp = try image.whitePoint(allocator, out, wp_pct);
        for (0..nt) |e| {
            const frame = try allocator.dupe(f64, out[e * npix ..][0..npix]);
            defer allocator.free(frame);
            image.stretch(frame, wp, gamma);
            const rgb = try image.colorize(allocator, frame, null);
            defer allocator.free(rgb);
            const png = try image.encodePng(allocator, @intCast(size), @intCast(size), rgb);
            defer allocator.free(png);
            var buf: [1024]u8 = undefined;
            const path = try std.fmt.bufPrint(&buf, "{s}/lc{d:0>4}.png", .{ fdir, e });
            std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = png }) catch |err| {
                std.debug.print("kdmp2lc: cannot write '{s}': {s}\n", .{ path, @errorName(err) });
                return err;
            };
        }
        std.debug.print("kdmp2lc: {d} frames written to {s}/ (shared white point {e:.3})\n", .{ nt, fdir, wp });
    }
}

fn usageErr() error{BadArgs} {
    std.debug.print(
        "usage: kdmp2lc <params.toml> --slow DIR [out.txt]\n" ++
            "       [--nu GHZ] [--dist KPC] [--t0 M] [--t1 M] [--nt N] [--frames DIR]\n" ++
            "       [--size N] [--fov M] [--incl DEG] [--phi DEG] [--rcam M] [--ss N]\n" ++
            "       [--rslow R] [--stride N] [--sigma-cut S] [--floor-cut F]\n" ++
            "       [--eps E] [--tau T] [--max-steps N] [--gamma G] [--wp PCT] [--threads N]\n",
        .{},
    );
    return error.BadArgs;
}
