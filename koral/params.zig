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
    // MKS2 coordinate shape (C: MKSR0/MKSH0). Optional so a preset can retarget
    // them (MKSR0 may be negative) while a run that omits them keeps the
    // problem's built-in constants. null = not overridden.
    mksr0: ?f64 = null,
    mksh0: ?f64 = null,
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
    /// Output every N steps regardless of code time (0 = disabled → the
    /// time-based DTOUT1/DTOUT2 cadence only). A convenience the C code lacks:
    /// for interactive VisIt watching a fixed step interval gives a
    /// predictable file count no matter how the CFL dt evolves (DTOUT1 is in
    /// GM/c³, so a slow run can reach nstep_max long before the first frame).
    nout_step: usize = 0,
    out_dir: []const u8 = "dumps",

    // Floors & ceilings and other per-run physics knobs (C: choices.h /
    // define.h). All OPTIONAL overrides: `null` (a key the file omits) keeps
    // the problem's built-in value; a set value overrides it. This is how the
    // `puffy_agn.toml` preset retargets koral-zig to the koral_lite_puffy AGN
    // configuration without disturbing the validated `.puffy` constants the
    // goldens pin against. Consumed by PROBLEMS/puffy/main.zig; see
    // docs/PUFFY_AGN_DIVERGENCES.md.
    rhofloor: ?f64 = null,
    uurhoratiomin: ?f64 = null,
    uurhoratiomax: ?f64 = null,
    eerhoratiomin: ?f64 = null,
    eerhoratiomax: ?f64 = null,
    eeuuratiomin: ?f64 = null,
    eeuuratiomax: ?f64 = null,
    b2rhoratiomax: ?f64 = null,
    b2uuratiomax: ?f64 = null,
    gammamaxhd: ?f64 = null,
    gammamaxrad: ?f64 = null,

    // implicit rad–gas solver (C: RADIMP*)
    radimpeps: ?f64 = null,
    radimpmaxiter: ?usize = null,
    /// C: OPDAMPINIMPLICIT / OPDAMPMAXLEVELS — opacity-damping ladder. If a whole
    /// implicit solve (all rungs) fails, retry with the radiation four-force
    /// scaled by opdamp_factor⁻ˡᵉᵛᵉˡ (level = 1..opdamp_maxlevels). null/0 = off
    /// (single pass, the validated behavior). koral_lite_puffy: 3.
    opdamp_maxlevels: ?usize = null,
    /// C: OPDAMPFACTOR — per-level opacity/four-force damping divisor.
    /// koral_lite_puffy: 10. Only used when opdamp_maxlevels > 0.
    opdamp_factor: ?f64 = null,
    /// C: DORADIMPFIXUPS — neighbour-average failed-implicit cells (both MHD and
    /// radiation prims, never ρ/B). The pass exists; this toggles it on.
    /// null = keep the built-in (off). koral_lite_puffy: on.
    doradimpfixups: ?bool = null,

    // reconstruction / wavespeed near-boundary tweaks (C: define.h)
    /// C: REDUCEORDERATBH — drop the reconstruction order by one inside the BH
    /// horizon (PPM→linear there). null = keep the built-in (off).
    reduceorderatbh: ?bool = null,
    /// C: DAMPRADWAVESPEEDNEARAXIS + …NCELLS — within this many cells of each
    /// pole, force the radiative wavespeed back to the undamped value 1/3
    /// (raises numerical diffusion near the axis for stability). null/0 = off.
    /// koral_lite_puffy: 2.
    dampradwavespeednearaxis: ?usize = null,

    // opacity channels (C: BREMSSTRAHLUNG / KLEINNISHINA on/off)
    bremsstrahlung: ?bool = null,
    kleinnishina: ?bool = null,
    /// C: USE_SYNCHROTRON_BRIDGE_FUNCTIONS — replace the Ramesh Terelfactor
    /// non-relativistic suppression with the Ramesh NR bridge (a Trad clamp, an
    /// NR component added to the synchrotron absorption opacity, and a crossover
    /// factor on the number opacity). null = keep the built-in (off).
    synchrotron_bridge: ?bool = null,
    /// Electron scattering opacity on/off. koral_lite_puffy (PROBLEM 147) leaves
    /// PR_KAPPAES undefined, so `calc_kappaes` returns 0 — scattering AND the
    /// Comptonization four-force term (∝ κ_es) both vanish. Set `false` to match
    /// that (kappaes → .none). null = keep the built-in PUFFY Klein–Nishina hook.
    scattering: ?bool = null,
    /// Path to a MESA Rosseland opacity table (`a09_z<Z>_x<X>.data`). Non-empty
    /// enables C's MESA_KAPPA: the free-free *Rosseland* opacity is replaced by
    /// the table lookup (minus electron scattering), while the free-free *Planck*
    /// absorption stays the bremsstrahlung formula (computed regardless of the
    /// `bremsstrahlung` toggle, exactly as C's MESA branch does). "" = off.
    /// The file must match the gas composition (hfrac→X, mfrac→Z).
    mesa_table: []const u8 = "",

    // magnetic floor frame (C: B2RHOFLOORFRAME). null/false = DRIFTFRAME
    // (Ressler+2017, the validated build); true = ZAMOFRAME (koral_lite_puffy):
    // the b²/ρ floor injects new mass in the ZAMO (normal-observer) frame,
    // isentropically (ISENTROPIC_B2RHOFLOORS), with a fluid-frame backup.
    zamo_floor_frame: ?bool = null,

    // gas composition (C: HFRAC/HEFRAC; MFRAC is derived = 1−hfrac−hefrac).
    // When hfrac is set the μ's come from the composition formula (no MU_*
    // override); when omitted the problem's built-in composition is kept.
    hfrac: ?f64 = null,
    hefrac: ?f64 = null,

    // torus / atmosphere / β-normalization (C: LT_KAPPA, MAXBETA, RHOATMMIN,
    // TGASATMMIN, ATMTRADINIT, and the ERADATMMIN factor form).
    lt_kappa: ?f64 = null,
    maxbeta: ?f64 = null,
    rhoatmmin: ?f64 = null,
    atm_tgas: ?f64 = null,
    atm_trad_init: ?f64 = null,
    atm_erad_factor: ?f64 = null,

    // execution
    deterministic: bool = false,
    nthreads: usize = 1,
    /// MPI φ-ring size (MPI plan §5: 1 rank per node, φ-only decomposition;
    /// nx/ny/nz above are GLOBAL dims). 0 = auto (take the launched world
    /// size); a nonzero value must match it — the double entry makes the
    /// params file a complete run record. Serial builds require ntz ∈ {0, 1}.
    /// 3D only: decompose() rejects ntz > 1 when nz == 1.
    ntz: usize = 0,

    pub fn load(allocator: std.mem.Allocator, io: std.Io, path: []const u8) !Params {
        const text = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(1 << 20));
        defer allocator.free(text);
        return parse(allocator, text);
    }

    /// Free the allocator-owned string fields. Every []const u8 field is
    /// heap-owned (parse dupes the defaults too), so this frees them all
    /// unconditionally — no literal-vs-heap guessing at the call site.
    pub fn deinit(self: *Params, allocator: std.mem.Allocator) void {
        inline for (std.meta.fields(Params)) |f| {
            if (f.type == []const u8) allocator.free(@field(self, f.name));
        }
        self.* = undefined;
    }

    /// Parse from a buffer. Every []const u8 field is allocator-owned on
    /// success (including untouched defaults); release with `deinit`.
    pub fn parse(allocator: std.mem.Allocator, text: []const u8) !Params {
        var p = Params{};
        // Uniform string ownership: dupe every string default up front so all
        // strings are heap-owned and deinit frees unconditionally. Neutralize
        // to "" first so p is deinit-safe at every step — free("") is a no-op,
        // so an OOM mid-loop never leaves a field pointing at a literal for the
        // errdefer to invalid-free.
        errdefer p.deinit(allocator);
        inline for (std.meta.fields(Params)) |f| {
            if (f.type == []const u8) @field(p, f.name) = "";
        }
        inline for (std.meta.fields(Params)) |f| {
            if (f.type == []const u8) {
                @field(p, f.name) = try allocator.dupe(u8, comptime f.defaultValue().?);
            }
        }

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
            try setField(allocator, &p, key, val, lineno);
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

    fn setField(allocator: std.mem.Allocator, p: *Params, key: []const u8, val: []const u8, lineno: usize) !void {
        inline for (std.meta.fields(Params)) |f| {
            if (std.mem.eql(u8, key, f.name)) {
                const parsed = parseValue(f.type, allocator, val) catch {
                    std.log.warn("params: line {d}: bad value for {s}: '{s}'", .{ lineno, key, val });
                    return error.BadParamsValue;
                };
                // Repeated key: release the previous (always-owned) dupe before
                // overwriting so the earlier value doesn't leak.
                if (f.type == []const u8) allocator.free(@field(p, f.name));
                @field(p, f.name) = parsed;
                return;
            }
        }
        std.log.warn("params: line {d}: unknown key '{s}'", .{ lineno, key });
        return error.UnknownParamsKey;
    }

    fn parseValue(comptime T: type, allocator: std.mem.Allocator, val: []const u8) !T {
        // Optional field: parse the inner type; a present key always means a
        // value (there is no syntax for writing `null`), which coerces to the
        // optional. An absent key keeps the null default and never gets here.
        switch (@typeInfo(T)) {
            .optional => |opt| return try parseValue(opt.child, allocator, val),
            else => {},
        }
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
                    break :blk try allocator.dupe(u8, val[1 .. val.len - 1]);
                break :blk try allocator.dupe(u8, val);
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
    var p = try Params.parse(std.testing.allocator, text);
    defer p.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(f64, 10.0), p.mass);
    try std.testing.expectEqual(@as(usize, 384), p.nx);
    try std.testing.expectEqual(@as(usize, 360), p.ny);
    try std.testing.expectEqual(@as(usize, 1), p.nz); // default
    try std.testing.expectApproxEqRel(@as(f64, 1.85), p.rmin, 1e-15);
    try std.testing.expectEqual(@as(?f64, 0.1), p.mksr0); // optional, provided
    try std.testing.expectEqual(true, p.deterministic);
    try std.testing.expectEqualStrings("out/puffy", p.out_dir);
    // untouched optional override defaults stay null (→ keep the built-in)
    try std.testing.expectEqual(@as(?f64, null), p.rhofloor);
    try std.testing.expectEqual(@as(?f64, null), p.mksh0);
}

test "params: optional physics overrides parse and default to null" {
    var p = try Params.parse(
        std.testing.allocator,
        "hfrac = 0.70\nkleinnishina = false\nradimpmaxiter = 50\nlt_kappa = 8.e-2\nmksr0 = -1.5\n",
    );
    defer p.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(?f64, 0.70), p.hfrac);
    try std.testing.expectEqual(@as(?bool, false), p.kleinnishina);
    try std.testing.expectEqual(@as(?usize, 50), p.radimpmaxiter);
    try std.testing.expectEqual(@as(?f64, 8.0e-2), p.lt_kappa);
    try std.testing.expectEqual(@as(?f64, -1.5), p.mksr0);
    // an omitted override stays null → main.zig keeps the built-in value
    try std.testing.expectEqual(@as(?bool, null), p.bremsstrahlung);
    try std.testing.expectEqual(@as(?f64, null), p.maxbeta);
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

test "params: default string field is heap-owned, deinit is safe" {
    // No out_dir line: the default "dumps" must be duped, not left a literal,
    // so deinit frees a real allocation rather than invalid-freeing a literal.
    // std.testing.allocator asserts both no-leak and no-invalid-free.
    var p = try Params.parse(std.testing.allocator, "mass = 10.0\n");
    defer p.deinit(std.testing.allocator);
    try std.testing.expectEqualStrings("dumps", p.out_dir);
}

test "params: repeated string key frees the earlier dupe" {
    // The first out_dir dupe must be freed when the key is set again; otherwise
    // std.testing.allocator reports the leak.
    var p = try Params.parse(std.testing.allocator, "out_dir = \"a\"\nout_dir = \"b\"\n");
    defer p.deinit(std.testing.allocator);
    try std.testing.expectEqualStrings("b", p.out_dir);
}
