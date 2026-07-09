//! MESA Rosseland opacity tables (C: MESA_KAPPA — opacities.c read_MESA_table
//! :978, return_MESA_kappa :1026, mybilinear macro ko.h:659).
//!
//! A `a09_z<Z>_x<X>.data` table gives log₁₀(κ_Rosseland [cm²/g]) on a
//! (logT, logR) grid, where logR = logρ − 3·logT + 18. The logT axis is
//! non-uniform (binary search); the logR axis is uniform (direct index). The
//! lookup is plain bilinear in log-log space, returning 10^(interpolated
//! log₁₀κ) in cm²/g — transcribed literally from C so the AGN run matches
//! koral_lite_puffy.
//!
//! In C the file is picked automatically from (Z = MFRAC, X = HFRAC) by an
//! exact-match table (get_MESA_opacity_filename, opacities.c:1070). koral-zig
//! takes the path explicitly (params `mesa_table`) — the caller is responsible
//! for choosing the file that matches the run's composition. MESA enters ONLY
//! the Rosseland opacity channels (see physics/opacities.zig); the free-free
//! Planck absorption is unchanged.

const std = @import("std");
const simd = @import("../math/simd.zig");

pub const MesaTable = struct {
    /// number of logT rows / logR columns (C: NLOGT / NLOGRHO — the latter is
    /// misnamed in C; it is the logR axis).
    nlogt: usize,
    nlogr: usize,
    /// logT grid (non-uniform), length nlogt.
    logt: []f64,
    /// logR grid (uniform, step logr[1]−logr[0]), length nlogr.
    logr: []f64,
    /// log₁₀(κ [cm²/g]) row-major, table[i*nlogr + j] = value at (logT_i, logR_j).
    table: []f64,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *MesaTable) void {
        self.allocator.free(self.logt);
        self.allocator.free(self.logr);
        self.allocator.free(self.table);
        self.* = undefined;
    }

    /// Load a MESA `.data` table (C: read_MESA_table, opacities.c:978). The C
    /// loader skips two header lines, reads ten dimension numbers (form,
    /// version, X, Z, NlogR, logR_min, logR_max, NlogT, logT_min, logT_max),
    /// skips the rest of that line plus a blank and the "logT … logR =" label
    /// line, then reads NlogR column headers (the logR grid) and NlogT rows of
    /// (logT, NlogR κ values). fscanf skips all whitespace, so blank separator
    /// lines are harmless — we mirror that by tokenising every float after the
    /// fixed five-line header.
    pub fn load(allocator: std.mem.Allocator, io: std.Io, path: []const u8) !MesaTable {
        const text = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(4 << 20));
        defer allocator.free(text);
        return parse(allocator, text);
    }

    pub fn parse(allocator: std.mem.Allocator, text: []const u8) !MesaTable {
        var lines = std.mem.splitScalar(u8, text, '\n');
        // line 0: title; line 1: column labels
        _ = lines.next() orelse return error.MesaTruncated;
        _ = lines.next() orelse return error.MesaTruncated;
        // line 2: the ten dimension numbers
        const dim_line = lines.next() orelse return error.MesaTruncated;
        var dims: [10]f64 = undefined;
        {
            var it = std.mem.tokenizeAny(u8, dim_line, " \t\r");
            var i: usize = 0;
            while (i < 10) : (i += 1) {
                const tok = it.next() orelse return error.MesaBadHeader;
                dims[i] = std.fmt.parseFloat(f64, tok) catch return error.MesaBadHeader;
            }
        }
        const nlogr: usize = @intFromFloat(dims[4]);
        const nlogt: usize = @intFromFloat(dims[7]);
        if (nlogr == 0 or nlogt == 0) return error.MesaBadHeader;
        // line 3: blank; line 4: "logT … logR = logRho - 3*logT + 18"
        _ = lines.next() orelse return error.MesaTruncated;
        _ = lines.next() orelse return error.MesaTruncated;

        // Everything from here on is whitespace-separated floats: nlogr column
        // headers (the logR grid) followed by nlogt rows of (logT + nlogr κ's).
        const want = nlogr + nlogt * (1 + nlogr);
        var vals = try allocator.alloc(f64, want);
        errdefer allocator.free(vals);
        var nread: usize = 0;
        while (lines.next()) |raw| {
            var it = std.mem.tokenizeAny(u8, raw, " \t\r");
            while (it.next()) |tok| {
                if (nread >= want) break;
                vals[nread] = std.fmt.parseFloat(f64, tok) catch return error.MesaBadData;
                nread += 1;
            }
            if (nread >= want) break;
        }
        if (nread < want) return error.MesaTruncated;

        var logt = try allocator.alloc(f64, nlogt);
        errdefer allocator.free(logt);
        var logr = try allocator.alloc(f64, nlogr);
        errdefer allocator.free(logr);
        var table = try allocator.alloc(f64, nlogt * nlogr);
        errdefer allocator.free(table);

        for (0..nlogr) |j| logr[j] = vals[j];
        var pos: usize = nlogr;
        for (0..nlogt) |i| {
            logt[i] = vals[pos];
            pos += 1;
            for (0..nlogr) |j| {
                table[i * nlogr + j] = vals[pos];
                pos += 1;
            }
        }
        allocator.free(vals);

        return .{
            .nlogt = nlogt,
            .nlogr = nlogr,
            .logt = logt,
            .logr = logr,
            .table = table,
            .allocator = allocator,
        };
    }

    /// C: return_MESA_kappa (opacities.c:1026) — Rosseland κ in cm²/g for
    /// log₁₀T and log₁₀ρ (cgs). logR = logρ − 3·logT + 18; clamp both to the
    /// grid; bracket logT by binary search (non-uniform) and logR by division
    /// (uniform); bilinear in log-log (mybilinear); return 10^result.
    pub fn lookup(self: *const MesaTable, logt_in: f64, logrho: f64) f64 {
        var lt = logt_in;
        var lr = logrho - 3.0 * logt_in + 18.0;

        const lt_lo = self.logt[0];
        const lt_hi = self.logt[self.nlogt - 1];
        const lr_lo = self.logr[0];
        const lr_hi = self.logr[self.nlogr - 1];
        if (lt < lt_lo) lt = lt_lo;
        if (lt > lt_hi) lt = lt_hi;
        if (lr < lr_lo) lr = lr_lo;
        if (lr > lr_hi) lr = lr_hi;

        // binary search for the logT bracket (grid non-uniform)
        var it_lo: usize = 0;
        var it_hi: usize = self.nlogt - 2;
        while (it_hi - it_lo > 1) {
            const mid = (it_lo + it_hi) / 2;
            if (lt < self.logt[mid + 1]) it_hi = mid else it_lo = mid;
        }

        // direct index for logR (grid uniform)
        const dlogr = self.logr[1] - self.logr[0];
        var ir_lo: usize = 0;
        {
            const raw = (lr - lr_lo) / dlogr;
            if (raw > 0.0) {
                const ri: usize = @intFromFloat(raw);
                ir_lo = if (ri > self.nlogr - 2) self.nlogr - 2 else ri;
            }
        }

        const x0 = self.logt[it_lo];
        const x1 = self.logt[it_lo + 1];
        const y0 = self.logr[ir_lo];
        const y1 = self.logr[ir_lo + 1];
        const z00 = self.table[it_lo * self.nlogr + ir_lo];
        const z10 = self.table[(it_lo + 1) * self.nlogr + ir_lo];
        const z01 = self.table[it_lo * self.nlogr + ir_lo + 1];
        const z11 = self.table[(it_lo + 1) * self.nlogr + ir_lo + 1];

        const logkappa = (z00 * (x1 - lt) * (y1 - lr) +
            z10 * (lt - x0) * (y1 - lr) +
            z01 * (x1 - lt) * (lr - y0) +
            z11 * (lt - x0) * (lr - y0)) /
            ((x1 - x0) * (y1 - y0));

        return std.math.pow(f64, 10.0, logkappa);
    }

    /// Rosseland κ (cm²/g) for a *linear* temperature [K] and mass density
    /// [g/cm³], taking log₁₀ internally (C: log10(Te)/log10(Trad) and
    /// log10(rhocgs), opacities.c:299-315). Mirrors C's positivity guard
    /// (rhocgs≤0 or T≤0 → clamp to keep the logs finite).
    pub fn lookupLinear(self: *const MesaTable, temp_k: f64, rhocgs: f64) f64 {
        const t = if (temp_k > 0.0) temp_k else 1.0;
        const r = if (rhocgs > 0.0) rhocgs else 1.0e-40;
        return self.lookup(std.math.log10(t), std.math.log10(r));
    }

    /// Per-lane MESA lookup over lane type T (f64 or @Vector(n, f64)), linear
    /// inputs. The table bracket/interpolation is inherently scalar, so vector
    /// lanes are resolved one at a time and reassembled — the opacity chain is
    /// comptime-T-generic (implicit-solver Jacobian batch), and every lane
    /// matches the scalar path.
    pub fn lookupG(self: *const MesaTable, comptime T: type, temp_k: T, rhocgs: T) T {
        if (T == f64) return self.lookupLinear(temp_k, rhocgs);
        const n = simd.lanes(T);
        var out: T = undefined;
        inline for (0..n) |k| out[k] = self.lookupLinear(temp_k[k], rhocgs[k]);
        return out;
    }
};

test "mesa: parse + lookup a synthetic 2x2 table" {
    // NlogR=2 (uniform), NlogT=2. logR grid {0, 1}; logT grid {3, 4}.
    // table log10(kappa): row logT=3 -> {0, 1}; row logT=4 -> {2, 3}.
    const text =
        \\title line
        \\form version X Z logRs logRmin logRmax logTs logTmin logTmax
        \\1 1 0.7 0.02 2 0.0 1.0 2 3.0 4.0
        \\
        \\   logT     logR = logRho - 3*logT + 18
        \\ 0.0 1.0
        \\ 3.0  0.0 1.0
        \\ 4.0  2.0 3.0
    ;
    var t = try MesaTable.parse(std.testing.allocator, text);
    defer t.deinit();
    try std.testing.expectEqual(@as(usize, 2), t.nlogt);
    try std.testing.expectEqual(@as(usize, 2), t.nlogr);
    // pick logT=3, logrho chosen so logR = logrho - 9 + 18 = 0 -> logrho = -9.
    // corner (logT=3, logR=0) has log10(kappa)=0 -> kappa=1.
    try std.testing.expectApproxEqRel(@as(f64, 1.0), t.lookup(3.0, -9.0), 1e-12);
    // (logT=4, logR=1): logrho = 1 + 12 - 18 = -5; corner value 3 -> kappa=1000.
    try std.testing.expectApproxEqRel(@as(f64, 1000.0), t.lookup(4.0, -5.0), 1e-12);
    // midpoint logT=3.5, logR=0.5 -> bilinear of {0,1,2,3} = 1.5 -> 10^1.5.
    // logrho for logR=0.5 at logT=3.5: 0.5 = logrho - 10.5 + 18 -> logrho = -7.
    try std.testing.expectApproxEqRel(std.math.pow(f64, 10.0, 1.5), t.lookup(3.5, -7.0), 1e-12);
}

test "mesa: out-of-range inputs clamp to the grid edge (no extrapolation)" {
    // Same 2x2 table as above: logT grid {3,4}, logR grid {0,1}, corners
    // log10(kappa) = {{0,1},{2,3}}.
    const text =
        \\title line
        \\form version X Z logRs logRmin logRmax logTs logTmin logTmax
        \\1 1 0.7 0.02 2 0.0 1.0 2 3.0 4.0
        \\
        \\   logT     logR = logRho - 3*logT + 18
        \\ 0.0 1.0
        \\ 3.0  0.0 1.0
        \\ 4.0  2.0 3.0
    ;
    var t = try MesaTable.parse(std.testing.allocator, text);
    defer t.deinit();

    // Far below both axes clamps to the (logT=3, logR=0) corner. lookup takes
    // logrho, and logR = logrho − 3·logT + 18, so pick logrho to drive logR low
    // too: logt_in=-2 → lt→3; logR=-29+6+18=-5 → lr→0. Corner value 0 → κ=1.
    // Identical to the in-range corner lookup(3,-9), i.e. the edge is held flat.
    try std.testing.expectApproxEqRel(t.lookup(3.0, -9.0), t.lookup(-2.0, -29.0), 1e-12);
    try std.testing.expectApproxEqRel(@as(f64, 1.0), t.lookup(-2.0, -29.0), 1e-12);

    // Far above both axes clamps to the (logT=4, logR=1) corner: value 3 → κ=1000
    // (not an extrapolated 10^huge).
    try std.testing.expectApproxEqRel(@as(f64, 1000.0), t.lookup(100.0, 1000.0), 1e-12);

    // A finite value never leaves the table's κ range [10^0, 10^3] whatever the
    // inputs — the defining safety property of the clamp.
    const k = t.lookup(-50.0, 12.0);
    try std.testing.expect(k >= 1.0 and k <= 1000.0);
}
