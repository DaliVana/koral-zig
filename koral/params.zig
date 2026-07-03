//! Runtime parameters — the numbers a physicist tweaks between runs
//! (C: the numeric `#define`s of PROBLEMS/*/define.h that do not change
//! generated code: MASS, BHSPIN, GAMMA, resolution, floors, cadences).
//!
//! Loaded from a flat TOML-subset file: `key = value` lines, `#` comments,
//! `[section]` headers ignored (sections are cosmetic grouping). Keys match
//! field names. Unknown keys are an error — catches typos, the config file
//! is the run record.

const std = @import("std");

pub const Params = struct {
    // physical
    /// BH mass in solar masses (C: MASS).
    mass: f64 = 1.0 / 147700.0,
    /// dimensionless spin (C: BHSPIN).
    bhspin: f64 = 0.0,
    /// gas adiabatic index (C: GAMMA).
    gam: f64 = 5.0 / 3.0,

    // grid (active cells; serial ⇒ == total)
    nx: usize = 1,
    ny: usize = 1,
    nz: usize = 1,

    // radial/angular domain in *output* coordinates; problem code derives
    // internal bounds (e.g. MINX = ln(rmin − mksr0) for MKS2).
    rmin: f64 = 0.0,
    rmax: f64 = 0.0,
    mksr0: f64 = 0.0,
    mksh0: f64 = 1.0,
    miny: f64 = 0.0,
    maxy: f64 = 1.0,
    minz: f64 = 0.0,
    maxz: f64 = 1.0,

    // run control
    tstart: f64 = 0.0,
    tmax: f64 = 0.0,
    nstep_max: usize = std.math.maxInt(usize),
    /// CFL factor (C: TSTEPLIM).
    tsteplim: f64 = 0.5,
    dtout1: f64 = 0.0,
    dtout2: f64 = 0.0,
    out_dir: []const u8 = "dumps",

    // floors & ceilings (C: choices.h / define.h)
    rhofloor: f64 = 1.0e-30,
    uurhoratiomin: f64 = 1.0e-8,
    uurhoratiomax: f64 = 1.0e2,
    eerhoratiomin: f64 = 1.0e-20,
    eerhoratiomax: f64 = 1.0e20,
    b2rhoratiomax: f64 = 100.0,
    b2uuratiomax: f64 = 100.0,
    gammamaxhd: f64 = 100.0,
    gammamaxrad: f64 = 100.0,

    // execution
    deterministic: bool = false,
    nthreads: usize = 1,

    pub fn load(a: std.mem.Allocator, io: std.Io, path: []const u8) !Params {
        const text = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 20));
        defer a.free(text);
        return parse(a, text);
    }

    /// Parse from a buffer. String values are duplicated with `a`.
    pub fn parse(a: std.mem.Allocator, text: []const u8) !Params {
        var p = Params{};
        var lines = std.mem.splitScalar(u8, text, '\n');
        var lineno: usize = 0;
        while (lines.next()) |raw| {
            lineno += 1;
            const line = stripComment(raw);
            const trimmed = std.mem.trim(u8, line, " \t\r");
            if (trimmed.len == 0) continue;
            if (trimmed[0] == '[') continue; // section header — cosmetic
            const eq = std.mem.indexOfScalar(u8, trimmed, '=') orelse {
                std.log.warn("params: line {d}: expected `key = value`: '{s}'", .{ lineno, trimmed });
                return error.BadParamsSyntax;
            };
            const key = std.mem.trim(u8, trimmed[0..eq], " \t");
            const val = std.mem.trim(u8, trimmed[eq + 1 ..], " \t");
            try setField(a, &p, key, val, lineno);
        }
        return p;
    }

    fn stripComment(line: []const u8) []const u8 {
        // '#' starts a comment unless inside a quoted string.
        var in_str = false;
        for (line, 0..) |c, i| {
            if (c == '"') in_str = !in_str;
            if (c == '#' and !in_str) return line[0..i];
        }
        return line;
    }

    fn setField(a: std.mem.Allocator, p: *Params, key: []const u8, val: []const u8, lineno: usize) !void {
        inline for (std.meta.fields(Params)) |f| {
            if (std.mem.eql(u8, key, f.name)) {
                @field(p, f.name) = parseValue(f.type, a, val) catch {
                    std.log.warn("params: line {d}: bad value for {s}: '{s}'", .{ lineno, key, val });
                    return error.BadParamsValue;
                };
                return;
            }
        }
        std.log.warn("params: line {d}: unknown key '{s}'", .{ lineno, key });
        return error.UnknownParamsKey;
    }

    fn parseValue(comptime T: type, a: std.mem.Allocator, val: []const u8) !T {
        return switch (T) {
            f64 => std.fmt.parseFloat(f64, val),
            usize => std.fmt.parseInt(usize, val, 10),
            bool => if (std.mem.eql(u8, val, "true"))
                true
            else if (std.mem.eql(u8, val, "false"))
                false
            else
                error.BadBool,
            []const u8 => blk: {
                if (val.len >= 2 and val[0] == '"' and val[val.len - 1] == '"')
                    break :blk try a.dupe(u8, val[1 .. val.len - 1]);
                break :blk try a.dupe(u8, val);
            },
            else => @compileError("params: unsupported field type " ++ @typeName(T)),
        };
    }
};

//
// ---- M0 gate tests -------------------------------------------------------
//

test "params: parse TOML subset with sections, comments, all value kinds" {
    const text =
        \\# PUFFY-like run configuration
        \\[physical]
        \\mass = 10.0
        \\gam = 1.6666666666666667
        \\
        \\[grid]
        \\nx = 384
        \\ny = 360   # theta cells
        \\rmin = 1.85
        \\rmax = 500.0
        \\mksr0 = 0.1
        \\
        \\[run]
        \\deterministic = true
        \\out_dir = "out/puffy"
    ;
    const p = try Params.parse(std.testing.allocator, text);
    defer std.testing.allocator.free(p.out_dir);
    try std.testing.expectEqual(@as(f64, 10.0), p.mass);
    try std.testing.expectEqual(@as(usize, 384), p.nx);
    try std.testing.expectEqual(@as(usize, 360), p.ny);
    try std.testing.expectEqual(@as(usize, 1), p.nz); // default
    try std.testing.expectApproxEqRel(@as(f64, 1.85), p.rmin, 1e-15);
    try std.testing.expectEqual(true, p.deterministic);
    try std.testing.expectEqualStrings("out/puffy", p.out_dir);
    // untouched defaults survive
    try std.testing.expectEqual(@as(f64, 1.0e-30), p.rhofloor);
}

test "params: unknown key is an error (typo protection)" {
    try std.testing.expectError(
        error.UnknownParamsKey,
        Params.parse(std.testing.allocator, "masss = 10.0\n"),
    );
}

test "params: bad value is an error" {
    try std.testing.expectError(
        error.BadParamsValue,
        Params.parse(std.testing.allocator, "mass = ten\n"),
    );
}
