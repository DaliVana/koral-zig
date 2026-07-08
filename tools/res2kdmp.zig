//! res2kdmp — convert a C KORAL *serial binary* restart dump into the Zig
//! KDMP checkpoint format, so a run initialized/advanced by the C code can be
//! continued by the Zig `puffy` executable (`puffy <toml> --restart out.kdmp`).
//!
//! The C serial restart is a pair (fileop.c fprint_restartfile_bin):
//!   * res####.head — one ASCII line: `## nfout1 nfout2 t PROBLEM TNX TNY TNZ`
//!   * res####.dat  — binary: TNX·TNY·TNZ index triples (int gix,giy,giz), then
//!                    the same number of primitive blocks (NV doubles each), in
//!                    the C write order (ix outer, iy, iz inner).
//! NV is not in the header, so it is recovered from the .dat size. The two
//! codes agree that `ldouble == double` and both store host-endian, so on a
//! little-endian host (every KORAL target) the primitive bytes copy across
//! verbatim; only the per-cell ordering is remapped to KDMP's iz-slow/ix-fast
//! layout using each record's stored global index.
//!
//! Usage:  zig build res2kdmp -- <path/res####.head> [out.kdmp]
//! (default output: the same stem with a .kdmp extension.)

const std = @import("std");
const builtin = @import("builtin");
const koral = @import("koral");
const dump = koral.io.dump;

pub fn main(init: std.process.Init) !void {
    // C's fwrite emits host-endian; KDMP is defined little-endian. The raw
    // primitive-byte copy below is only valid where they coincide.
    comptime std.debug.assert(builtin.cpu.arch.endian() == .little);

    const allocator = init.gpa;
    const io = init.io;

    var args = std.process.Args.Iterator.init(init.minimal.args);
    defer args.deinit();
    _ = args.next(); // program name
    const head_path = args.next() orelse {
        std.debug.print(
            "usage: res2kdmp <path/res####.head> [out.kdmp]\n" ++
                "  Convert a C KORAL serial restart (res####.head + res####.dat)\n" ++
                "  into a single self-describing KDMP checkpoint for `puffy --restart`.\n" ++
                "  Assumes a single (serial) tile and a little-endian host.\n",
            .{},
        );
        return error.MissingArg;
    };
    if (!std.mem.endsWith(u8, head_path, ".head")) {
        std.debug.print("res2kdmp: expected a .head file, got '{s}'\n", .{head_path});
        return error.BadArg;
    }
    const stem = head_path[0 .. head_path.len - ".head".len];
    const dat_path = try std.fmt.allocPrint(allocator, "{s}.dat", .{stem});
    defer allocator.free(dat_path);
    const out_path = if (args.next()) |a|
        try allocator.dupe(u8, a)
    else
        try std.fmt.allocPrint(allocator, "{s}.kdmp", .{stem});
    defer allocator.free(out_path);

    // ---- header: `## nfout1 nfout2 t PROBLEM TNX TNY TNZ` -------------------
    const head = try std.Io.Dir.cwd().readFileAlloc(io, head_path, allocator, .limited(1 << 16));
    defer allocator.free(head);
    var tok = std.mem.tokenizeAny(u8, head, " \t\r\n");
    if (!std.mem.eql(u8, tok.next() orelse "", "##")) {
        std.debug.print("res2kdmp: malformed head (no leading `##`): '{s}'\n", .{head_path});
        return error.BadHead;
    }
    const nfout1 = try parseU32(&tok);
    _ = try parseU32(&tok); // nfout2 (avg-file counter — not needed to continue)
    const t = try std.fmt.parseFloat(f64, tok.next() orelse return error.BadHead);
    _ = try parseU32(&tok); // PROBLEM id
    const nx: usize = try parseU32(&tok);
    const ny: usize = try parseU32(&tok);
    const nz: usize = try parseU32(&tok);

    // ---- body: index block then primitive block ----------------------------
    const dat = try std.Io.Dir.cwd().readFileAlloc(io, dat_path, allocator, .limited(1 << 31));
    defer allocator.free(dat);
    const ncell = nx * ny * nz;
    const idx_bytes = ncell * 3 * 4; // 3 × int32 per cell
    if (dat.len < idx_bytes) return error.TruncatedDat;
    const prim_bytes = dat.len - idx_bytes;
    if (ncell == 0 or prim_bytes % (ncell * 8) != 0) {
        std.debug.print(
            "res2kdmp: .dat size {d} inconsistent with {d} cells (prim region {d} not a multiple of {d})\n",
            .{ dat.len, ncell, prim_bytes, ncell * 8 },
        );
        return error.BadDat;
    }
    const nv = prim_bytes / (ncell * 8);

    // ---- assemble KDMP -----------------------------------------------------
    const out = try allocator.alloc(u8, dump.header_size + ncell * nv * 8);
    defer allocator.free(out);
    _ = dump.writeDumpHeader(out, .{
        .nx = @intCast(nx),
        .ny = @intCast(ny),
        .nz = @intCast(nz),
        .nv = @intCast(nv),
        .t = t,
        .nstep = 0, // C does not preserve the step count across restarts
        .out_idx = nfout1,
    });

    // Remap each C record into KDMP's (iz-outer, iy, ix-inner) cell order using
    // its stored global index; the NV doubles copy across as raw bytes.
    const cell_bytes = nv * 8;
    var ic: usize = 0;
    while (ic < ncell) : (ic += 1) {
        const gix = std.mem.readInt(i32, dat[ic * 12 ..][0..4], .little);
        const giy = std.mem.readInt(i32, dat[ic * 12 + 4 ..][0..4], .little);
        const giz = std.mem.readInt(i32, dat[ic * 12 + 8 ..][0..4], .little);
        if (gix < 0 or giy < 0 or giz < 0 or
            @as(usize, @intCast(gix)) >= nx or @as(usize, @intCast(giy)) >= ny or @as(usize, @intCast(giz)) >= nz)
        {
            std.debug.print("res2kdmp: record {d} has out-of-range global index ({d},{d},{d})\n", .{ ic, gix, giy, giz });
            return error.BadIndex;
        }
        const dst_cell = (@as(usize, @intCast(giz)) * ny + @as(usize, @intCast(giy))) * nx + @as(usize, @intCast(gix));
        const src_off = idx_bytes + ic * cell_bytes;
        const dst_off = dump.header_size + dst_cell * cell_bytes;
        @memcpy(out[dst_off..][0..cell_bytes], dat[src_off..][0..cell_bytes]);
    }

    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = out_path, .data = out });
    std.debug.print(
        "res2kdmp: {s} + {s} -> {s}  (grid {d}×{d}×{d}, nv={d}, t={e:.6}, frame #{d})\n",
        .{ head_path, dat_path, out_path, nx, ny, nz, nv, t, nfout1 },
    );
}

fn parseU32(tok: *std.mem.TokenIterator(u8, .any)) !u32 {
    return std.fmt.parseInt(u32, tok.next() orelse return error.BadHead, 10);
}
