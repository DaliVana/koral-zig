//! Minimal FITS image writer for the EHT-comparison pipeline.
//!
//! One primary HDU: BITPIX = -64 (big-endian f64), NAXIS = 2, with the
//! AIPS-style radio keywords the eht-imaging loader reads; CDELT1/2 in
//! degrees (CDELT1 negative: RA grows east = LEFT), OBSRA/OBSDEC + CRVAL1/2
//! (both spellings, different loaders read different ones), FREQ [Hz],
//! MJD, OBJECT, BUNIT = 'JY/PIXEL'. Pixel values are flux density per
//! pixel [Jy]; I_ν · Ω_pix · 10²³; the unit ehtim images carry, so
//! `eh.image.load_fits(...)` → `im.observe(...)` works directly.
//!
//! The renderer's image buffer is row-major with row 0 at the TOP and +x
//! toward +α (the camera's azimuthal impact direction); FITS stores row 0
//! at the BOTTOM with RA increasing left. The writer flips rows and keeps
//! +x → −RA, so the projected spin axis points to +DEC (up) and +α maps to
//! east. The absolute sky position angle of a simulation is arbitrary;
//! rotate in post (ehtim `im.rotate`) when comparing to a source with a
//! known jet/spin PA.
//!
//! Serialization is pure (allocator-owned buffer, no filesystem) in the
//! io/dump.zig style, so the format is testable byte-by-byte.

const std = @import("std");

pub const block = 2880;
pub const card = 80;

pub const Opts = struct {
    /// pixel scale [deg], positive (the writer negates the RA axis)
    cdelt_deg: f64,
    /// source J2000 coordinates [deg]
    ra_deg: f64,
    dec_deg: f64,
    /// observing frequency [Hz]
    freq_hz: f64,
    /// modified Julian date of the (synthetic) observation
    mjd: f64 = 57850.0, // 2017 Apr 7, the EHT Sgr A* campaign
    object: []const u8 = "SGRA",
};

/// Serialize `img` (width×height, row-major, row 0 = top, values in
/// Jy/pixel) into a complete FITS file. Caller owns the returned buffer.
pub fn encode(allocator: std.mem.Allocator, img: []const f64, width: usize, height: usize, o: Opts) ![]u8 {
    std.debug.assert(img.len == width * height);
    const data_bytes = img.len * 8;
    const total = block + ((data_bytes + block - 1) / block) * block;
    const out = try allocator.alloc(u8, total);
    errdefer allocator.free(out);
    @memset(out, 0); // data padding; header blanks are set below

    // ---- header (one 2880-byte block, 36 cards) ----
    var h = Header{ .buf = out[0..block] };
    h.logical("SIMPLE", true, "koral-zig GRRT image");
    h.int("BITPIX", -64, "IEEE double");
    h.int("NAXIS", 2, "");
    h.int("NAXIS1", @intCast(width), "");
    h.int("NAXIS2", @intCast(height), "");
    h.str("OBJECT", o.object, "");
    h.str("TELESCOP", "VLBI", "");
    h.str("BUNIT", "JY/PIXEL", "flux density per pixel");
    h.str("CTYPE1", "RA---SIN", "");
    h.num("CRVAL1", o.ra_deg, "deg");
    h.num("CDELT1", -o.cdelt_deg, "deg (RA grows east=left)");
    h.num("CRPIX1", 0.5 * @as(f64, @floatFromInt(width + 1)), "image center");
    h.str("CUNIT1", "deg", "");
    h.str("CTYPE2", "DEC--SIN", "");
    h.num("CRVAL2", o.dec_deg, "deg");
    h.num("CDELT2", o.cdelt_deg, "deg");
    h.num("CRPIX2", 0.5 * @as(f64, @floatFromInt(height + 1)), "image center");
    h.str("CUNIT2", "deg", "");
    h.num("OBSRA", o.ra_deg, "deg (ehtim reads this spelling)");
    h.num("OBSDEC", o.dec_deg, "deg");
    h.num("FREQ", o.freq_hz, "Hz");
    h.num("MJD", o.mjd, "");
    h.raw("HISTORY koral-zig kdmp2png GRRT image (I_nu * Omega_pix * 1e23)");
    h.raw("END");
    h.blankFill();

    // ---- data: big-endian f64, rows flipped to FITS bottom-up ----
    var w: usize = block;
    var py = height;
    while (py > 0) {
        py -= 1;
        for (img[py * width ..][0..width]) |v| {
            std.mem.writeInt(u64, out[w..][0..8], @bitCast(v), .big);
            w += 8;
        }
    }
    return out;
}

const Header = struct {
    buf: []u8,
    n: usize = 0,

    fn next(h: *Header) []u8 {
        std.debug.assert(h.n < block / card); // one block is plenty here
        const c = h.buf[h.n * card ..][0..card];
        h.n += 1;
        @memset(c, ' ');
        return c;
    }

    fn key(c: []u8, name: []const u8) void {
        @memcpy(c[0..name.len], name);
        c[8] = '=';
        // c[9] stays ' '
    }

    fn logical(h: *Header, name: []const u8, v: bool, comment: []const u8) void {
        const c = h.next();
        key(c, name);
        c[29] = if (v) 'T' else 'F';
        putComment(c, comment);
    }

    fn int(h: *Header, name: []const u8, v: i64, comment: []const u8) void {
        const c = h.next();
        key(c, name);
        var tmp: [20]u8 = undefined;
        const s = std.fmt.bufPrint(&tmp, "{d}", .{v}) catch unreachable;
        @memcpy(c[30 - s.len .. 30], s);
        putComment(c, comment);
    }

    /// Fixed-format float card: value right-justified to column 30,
    /// uppercase exponent (FITS standard; astropy's strict parser wants it).
    fn num(h: *Header, name: []const u8, v: f64, comment: []const u8) void {
        const c = h.next();
        key(c, name);
        var tmp: [26]u8 = undefined;
        const s = std.fmt.bufPrint(&tmp, "{e:.12}", .{v}) catch unreachable;
        for (s) |*ch| {
            if (ch.* == 'e') ch.* = 'E';
        }
        if (s.len <= 20) {
            @memcpy(c[30 - s.len .. 30], s);
        } else {
            @memcpy(c[10..][0..s.len], s); // free format past col 30
        }
        putComment(c, comment);
    }

    fn str(h: *Header, name: []const u8, v: []const u8, comment: []const u8) void {
        const c = h.next();
        key(c, name);
        // 'value' single-quoted from col 11, padded to the 8-char minimum
        c[10] = '\'';
        const n = @min(v.len, 60);
        @memcpy(c[11..][0..n], v[0..n]);
        const close = 11 + @max(n, 8);
        c[close] = '\'';
        putComment(c, comment);
    }

    fn raw(h: *Header, text: []const u8) void {
        const c = h.next();
        const n = @min(text.len, card);
        @memcpy(c[0..n], text[0..n]);
    }

    fn putComment(c: []u8, comment: []const u8) void {
        if (comment.len == 0) return;
        c[31] = '/';
        const n = @min(comment.len, card - 33);
        @memcpy(c[33..][0..n], comment[0..n]);
    }

    fn blankFill(h: *Header) void {
        while (h.n < block / card) {
            const c = h.buf[h.n * card ..][0..card];
            h.n += 1;
            @memset(c, ' ');
        }
    }
};
