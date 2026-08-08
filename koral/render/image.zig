//! Image backend for the GRRT renderer: intensity-map post-processing
//! (normalization, gamma stretch, Gaussian beam blur), the `afmhot`
//! black–red–orange–yellow–white colormap (the EHT palette), and a
//! dependency-free PNG encoder.
//!
//! The PNG writer emits *stored* (uncompressed) deflate blocks inside a
//! valid zlib stream — no compressor needed, byte-exact on every Zig
//! version, and a 512×512 RGB frame is still under 1 MB. Serialization is
//! pure (returns a caller-owned byte buffer); the executable does the IO,
//! matching io/dump.zig's convention.

const std = @import("std");

// ---- colormap --------------------------------------------------------------

/// matplotlib's `afmhot` at t ∈ [0,1]: piecewise-linear
/// R = 2t, G = 2t−½, B = 2t−1, each clamped to [0,1].
pub fn afmhot(t: f64) [3]u8 {
    const tt = std.math.clamp(t, 0.0, 1.0);
    return .{
        quant(2.0 * tt),
        quant(2.0 * tt - 0.5),
        quant(2.0 * tt - 1.0),
    };
}

fn quant(v: f64) u8 {
    const c = std.math.clamp(v, 0.0, 1.0);
    return @intFromFloat(@round(255.0 * c));
}

// ---- intensity post-processing ---------------------------------------------

/// White point for normalization: the `pct` percentile (e.g. 99.8) of the
/// *positive* intensities — a lone hot pixel must not dim the whole disk.
/// Returns 0 when the image is entirely dark.
pub fn whitePoint(allocator: std.mem.Allocator, img: []const f64, pct: f64) !f64 {
    var pos: std.ArrayList(f64) = .empty;
    defer pos.deinit(allocator);
    for (img) |v| {
        if (v > 0 and std.math.isFinite(v)) try pos.append(allocator, v);
    }
    if (pos.items.len == 0) return 0;
    std.mem.sort(f64, pos.items, {}, std.sort.asc(f64));
    const p = std.math.clamp(pct, 0.0, 100.0) / 100.0;
    const idx_f = p * @as(f64, @floatFromInt(pos.items.len - 1));
    const idx: usize = @intFromFloat(@round(idx_f));
    return pos.items[@min(idx, pos.items.len - 1)];
}

/// In-place normalize to [0,1] by `wp` and apply the gamma stretch
/// v ← (v/wp)^gamma. A wp of 0 blacks the image out (guarded by caller).
pub fn stretch(img: []f64, wp: f64, gamma: f64) void {
    if (wp <= 0) {
        for (img) |*v| v.* = 0;
        return;
    }
    for (img) |*v| {
        const n = std.math.clamp(v.* / wp, 0.0, 1.0);
        v.* = std.math.pow(f64, n, gamma);
    }
}

/// Separable Gaussian blur of the intensity map (σ in pixels, radius 3σ),
/// clamped at the edges. Models the instrument beam — the EHT's ~20 μas
/// resolution is what gives the published images their soft look.
/// No-op for sigma <= 0.
pub fn gaussianBlur(allocator: std.mem.Allocator, img: []f64, w: usize, h: usize, sigma: f64) !void {
    if (sigma <= 0) return;
    std.debug.assert(img.len == w * h);
    const radius: usize = @intFromFloat(@ceil(3.0 * sigma));
    if (radius == 0) return;

    const kernel = try allocator.alloc(f64, 2 * radius + 1);
    defer allocator.free(kernel);
    var ksum: f64 = 0;
    for (kernel, 0..) |*kv, i| {
        const x = @as(f64, @floatFromInt(i)) - @as(f64, @floatFromInt(radius));
        kv.* = @exp(-0.5 * x * x / (sigma * sigma));
        ksum += kv.*;
    }
    for (kernel) |*kv| kv.* /= ksum;

    const tmp = try allocator.alloc(f64, img.len);
    defer allocator.free(tmp);

    // horizontal pass img -> tmp
    for (0..h) |y| {
        for (0..w) |x| {
            var acc: f64 = 0;
            for (kernel, 0..) |kv, i| {
                const off = @as(i64, @intCast(x)) + @as(i64, @intCast(i)) - @as(i64, @intCast(radius));
                const xi: usize = @intCast(std.math.clamp(off, 0, @as(i64, @intCast(w - 1))));
                acc += kv * img[y * w + xi];
            }
            tmp[y * w + x] = acc;
        }
    }
    // vertical pass tmp -> img
    for (0..h) |y| {
        for (0..w) |x| {
            var acc: f64 = 0;
            for (kernel, 0..) |kv, i| {
                const off = @as(i64, @intCast(y)) + @as(i64, @intCast(i)) - @as(i64, @intCast(radius));
                const yi: usize = @intCast(std.math.clamp(off, 0, @as(i64, @intCast(h - 1))));
                acc += kv * tmp[yi * w + x];
            }
            img[y * w + x] = acc;
        }
    }
}

/// Stretched [0,1] intensities → RGB bytes through `afmhot`.
pub fn colorize(allocator: std.mem.Allocator, img: []const f64, rgb_out: ?[]u8) ![]u8 {
    const rgb = rgb_out orelse try allocator.alloc(u8, img.len * 3);
    std.debug.assert(rgb.len == img.len * 3);
    for (img, 0..) |v, i| {
        const c = afmhot(v);
        rgb[3 * i + 0] = c[0];
        rgb[3 * i + 1] = c[1];
        rgb[3 * i + 2] = c[2];
    }
    return rgb;
}

// ---- PNG encoder -----------------------------------------------------------

pub const png_signature = [8]u8{ 0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A };

/// Encode an 8-bit RGB image as a PNG file image (caller owns the bytes).
/// IDAT is a zlib stream of stored deflate blocks: header 0x78 0x01, blocks
/// of ≤ 65535 raw bytes (BFINAL/BTYPE=00 + LEN + ~LEN), Adler-32 trailer.
pub fn encodePng(allocator: std.mem.Allocator, w: u32, h: u32, rgb: []const u8) ![]u8 {
    std.debug.assert(rgb.len == @as(usize, w) * h * 3);

    // raw scanlines, each prefixed with filter type 0 (None)
    const stride = 1 + 3 * @as(usize, w);
    const raw = try allocator.alloc(u8, @as(usize, h) * stride);
    defer allocator.free(raw);
    for (0..h) |y| {
        raw[y * stride] = 0;
        @memcpy(raw[y * stride + 1 ..][0 .. 3 * @as(usize, w)], rgb[y * 3 * @as(usize, w) ..][0 .. 3 * @as(usize, w)]);
    }

    // zlib stream of stored blocks
    var z: std.ArrayList(u8) = .empty;
    defer z.deinit(allocator);
    try z.appendSlice(allocator, &.{ 0x78, 0x01 });
    var off: usize = 0;
    while (off < raw.len) {
        const n: usize = @min(65535, raw.len - off);
        const final: u8 = if (off + n == raw.len) 1 else 0;
        try z.append(allocator, final);
        const len16: u16 = @intCast(n);
        try z.append(allocator, @truncate(len16));
        try z.append(allocator, @truncate(len16 >> 8));
        try z.append(allocator, @truncate(~len16));
        try z.append(allocator, @truncate(~len16 >> 8));
        try z.appendSlice(allocator, raw[off..][0..n]);
        off += n;
    }
    var adler: [4]u8 = undefined;
    std.mem.writeInt(u32, &adler, std.hash.Adler32.hash(raw), .big);
    try z.appendSlice(allocator, &adler);

    // chunks
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, &png_signature);

    var ihdr: [13]u8 = undefined;
    std.mem.writeInt(u32, ihdr[0..4], w, .big);
    std.mem.writeInt(u32, ihdr[4..8], h, .big);
    ihdr[8] = 8; // bit depth
    ihdr[9] = 2; // color type: truecolor RGB
    ihdr[10] = 0; // compression
    ihdr[11] = 0; // filter
    ihdr[12] = 0; // interlace
    try writeChunk(allocator, &out, "IHDR", &ihdr);
    try writeChunk(allocator, &out, "IDAT", z.items);
    try writeChunk(allocator, &out, "IEND", "");

    return out.toOwnedSlice(allocator);
}

fn writeChunk(allocator: std.mem.Allocator, out: *std.ArrayList(u8), typ: *const [4]u8, data: []const u8) !void {
    var len_be: [4]u8 = undefined;
    std.mem.writeInt(u32, &len_be, @intCast(data.len), .big);
    try out.appendSlice(allocator, &len_be);
    try out.appendSlice(allocator, typ);
    try out.appendSlice(allocator, data);
    var crc = std.hash.Crc32.init();
    crc.update(typ);
    crc.update(data);
    var crc_be: [4]u8 = undefined;
    std.mem.writeInt(u32, &crc_be, crc.final(), .big);
    try out.appendSlice(allocator, &crc_be);
}
