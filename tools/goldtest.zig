//! goldtest: run the Gold et al. 2020 (ApJ 897, 148) EHT GRRT
//! code-comparison tests (render/verify.zig) and report total fluxes
//! against the paper's Table 2: the EXACT (arbitrary-precision) solution
//! and the seven participating codes (BHOSS, GRTRANS, IPOLE, ODYSSEY,
//! RAIKOU, RAPTOR, VRT2). Also writes one PNG per test for visual
//! comparison with their Figure 2; tools/goldfig.py turns `--fits` dumps
//! into a full Figure 2/3 reproduction (verified: our FITS orientation
//! matches their West-right panels, Doppler crescent on the east side).
//!
//! usage: goldtest [outdir]
//!   --size N     image resolution        (default 128; the paper's)
//!   --ss N       rays per pixel axis     (default 2)
//!   --eps E      geodesic step, cells    (default 0.25)
//!   --tests STR  subset, e.g. "145"      (default "12345")
//!   --fits       also dump gold_testN.fits (Jy/pixel; for the Figure 2/3
//!                reproduction script, tools/goldfig.py)
//!   --tag STR    suffix for output names (e.g. "ref" -> gold_testN_ref.fits)
//!   --threads N                          (default: CPU count)

const std = @import("std");
const koral = @import("koral");
const verify = koral.render.verify;
const image = koral.render.image;

/// Table 2 of the paper: per-code total fluxes [Jy] for tests 1..5.
const codes = [_]struct { name: []const u8, s: [5]f64 }{
    .{ .name = "BHOSS", .s = .{ 1.6466, 1.4361, 0.4419, 0.2711, 0.0256 } },
    .{ .name = "GRTRANS", .s = .{ 1.6606, 1.4498, 0.4457, 0.2727, 0.0257 } },
    .{ .name = "IPOLE", .s = .{ 1.6604, 1.4486, 0.4454, 0.2729, 0.0258 } },
    .{ .name = "ODYSSEY", .s = .{ 1.6466, 1.4361, 0.4419, 0.2709, 0.0254 } },
    .{ .name = "RAIKOU", .s = .{ 1.6617, 1.4710, 0.4508, 0.2763, 0.0260 } },
    .{ .name = "RAPTOR", .s = .{ 1.6609, 1.4486, 0.4456, 0.2729, 0.0258 } },
    .{ .name = "VRT2", .s = .{ 1.6694, 1.4568, 0.4480, 0.2749, 0.0259 } },
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    var args = std.process.Args.Iterator.init(init.minimal.args);
    defer args.deinit();
    _ = args.next();

    var outdir: []const u8 = ".";
    var size: usize = 128;
    var ss: usize = 2;
    var eps: f64 = 0.25;
    var which: []const u8 = "12345";
    var want_fits = false;
    var tag: []const u8 = "";
    var nthreads: usize = std.Thread.getCpuCount() catch 8;

    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--size")) {
            size = std.fmt.parseInt(usize, args.next() orelse return usageErr(), 10) catch return usageErr();
        } else if (std.mem.eql(u8, arg, "--ss")) {
            ss = std.fmt.parseInt(usize, args.next() orelse return usageErr(), 10) catch return usageErr();
        } else if (std.mem.eql(u8, arg, "--eps")) {
            eps = std.fmt.parseFloat(f64, args.next() orelse return usageErr()) catch return usageErr();
        } else if (std.mem.eql(u8, arg, "--tests")) {
            which = args.next() orelse return usageErr();
        } else if (std.mem.eql(u8, arg, "--fits")) {
            want_fits = true;
        } else if (std.mem.eql(u8, arg, "--tag")) {
            tag = args.next() orelse return usageErr();
        } else if (std.mem.eql(u8, arg, "--threads")) {
            nthreads = std.fmt.parseInt(usize, args.next() orelse return usageErr(), 10) catch return usageErr();
        } else if (std.mem.startsWith(u8, arg, "--")) {
            return usageErr();
        } else {
            outdir = arg;
        }
    }

    std.debug.print(
        "goldtest: Gold et al. 2020 (ApJ 897,148) EHT GRRT verification | {d}px ss{d} eps={d} | camera r=1000M i=60deg fov=30M | {d} threads\n" ++
            "  test |  koral-zig |   EXACT  |  dev     | 7-code range      | verdict\n",
        .{ size, ss, eps, nthreads },
    );

    const img = try allocator.alloc(f64, size * size);
    defer allocator.free(img);

    var worst: f64 = 0;
    for (verify.tests, 0..) |td, i| {
        const digit: u8 = @intCast('1' + i);
        if (std.mem.indexOfScalar(u8, which, digit) == null) continue;

        const t0 = std.Io.Timestamp.now(io, .awake);
        verify.renderGold(&td, size, ss, .{ .eps = eps }, nthreads, img);
        const dur = t0.durationTo(std.Io.Timestamp.now(io, .awake));
        const secs = @as(f64, @floatFromInt(@as(i64, @intCast(dur.nanoseconds)))) / 1e9;

        const s = verify.totalFluxJy(img, size);
        const dev = (s - td.s_exact) / td.s_exact;
        worst = @max(worst, @abs(dev));
        var lo: f64 = 1e30;
        var hi: f64 = -1e30;
        for (codes) |c| {
            lo = @min(lo, c.s[i]);
            hi = @max(hi, c.s[i]);
        }
        const inside = s >= lo and s <= hi;
        std.debug.print("    {d}  |  {d:.4} Jy | {d:.4} Jy | {d:>6.2}%  | [{d:.4}, {d:.4}]  | {s} ({d:.1}s)\n", .{
            i + 1, s, td.s_exact, 100.0 * dev, lo, hi, if (inside) "in 7-code range" else if (@abs(dev) < 0.01) "within 1% of EXACT" else "OUTSIDE", secs,
        });

        // FITS dump (Jy/pixel) for the Figure 2/3 reproduction script
        if (want_fits) {
            const pix_rad = verify.fov_m * verify.m_cm / verify.d_cm / @as(f64, @floatFromInt(size));
            const jy = try allocator.dupe(f64, img);
            defer allocator.free(jy);
            for (jy) |*v| v.* *= pix_rad * pix_rad * 1.0e23;
            var fb: [1024]u8 = undefined;
            const sep: []const u8 = if (tag.len > 0) "_" else "";
            const fpath = try std.fmt.bufPrint(&fb, "{s}/gold_test{d}{s}{s}.fits", .{ outdir, i + 1, sep, tag });
            const fbytes = try koral.render.fits.encode(allocator, jy, size, size, .{
                .cdelt_deg = pix_rad * 180.0 / std.math.pi,
                .ra_deg = 0,
                .dec_deg = 0,
                .freq_hz = 230.0e9,
                .object = "GOLD2020",
            });
            defer allocator.free(fbytes);
            std.Io.Dir.cwd().writeFile(io, .{ .sub_path = fpath, .data = fbytes }) catch |err| {
                std.debug.print("goldtest: cannot write '{s}': {s}\n", .{ fpath, @errorName(err) });
                return err;
            };
        }

        // PNG for visual comparison with their Figure 2
        var buf: [1024]u8 = undefined;
        const path = try std.fmt.bufPrint(&buf, "{s}/gold_test{d}.png", .{ outdir, i + 1 });
        const copy = try allocator.dupe(f64, img);
        defer allocator.free(copy);
        const wp = try image.whitePoint(allocator, copy, 99.9);
        image.stretch(copy, wp, 0.65);
        const rgb = try image.colorize(allocator, copy, null);
        defer allocator.free(rgb);
        const png = try image.encodePng(allocator, @intCast(size), @intCast(size), rgb);
        defer allocator.free(png);
        std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = png }) catch |err| {
            std.debug.print("goldtest: cannot write '{s}': {s}\n", .{ path, @errorName(err) });
            return err;
        };
    }
    std.debug.print("goldtest: worst deviation from EXACT: {d:.2}%\n", .{100.0 * worst});
}

fn usageErr() error{BadArgs} {
    std.debug.print("usage: goldtest [outdir] [--size N] [--ss N] [--eps E] [--tests 12345] [--fits] [--tag STR] [--threads N]\n", .{});
    return error.BadArgs;
}
