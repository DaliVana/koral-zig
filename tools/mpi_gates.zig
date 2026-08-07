//! MPI validation-ladder harness (MPI plan §10, gates 2–5). Gate 1 (the
//! decompose()/Grid arithmetic) lives in the unit test suite; this binary
//! covers everything that needs live ranks. Each rank recomputes the serial
//! reference IN-PROCESS (serial Zig is the oracle — there are no C-MPI
//! goldens and there will be none) and compares its own slab, so the gates
//! involve no files and no rank-0 gathering.
//!
//!   zig build mpi-gates -Dmpi -Doptimize=ReleaseSafe
//!   mpiexec -n 1 zig-out/bin/mpi-gates            # gate 2 (1-rank ≡ serial, bitwise)
//!   mpiexec -n 2 zig-out/bin/mpi-gates            # gates 3–5 at 1×1×2
//!   mpiexec -n 3 ... ; mpiexec -n 4 ...           # 1×1×3, 1×1×4
//!
//! Exit 0 = all gates green on every rank (any failure exits nonzero, so a
//! shell loop over -n 1..4 is the whole ladder).
//!
//! What is (and isn't) bitwise, by design (plan §1.1-8): interior z-face
//! ghost planes are exchanged bit-identically, so cells away from the x/y
//! boundaries stay BITWISE equal to serial across φ seams for the first
//! steps; x/y-ghost × z-ghost corner regions are constructed differently
//! (serial corner machinery vs exchange + C's middle-corner BC pass), so
//! cells near the x/y edges are compared at field-scale tolerance only.
//! The PUFFY gate is tolerance-only everywhere: the dynamo's ring-reduced
//! scale height regroups a global sum, which shifts every cell by ulps.

const std = @import("std");
const koral = @import("koral");

const Comm = koral.comm.Comm;
const Grid = koral.Grid;

// The 3D Minkowski radiation-MHD box (threading_tests' config + a z extent).
const mink_cfg = koral.Config{
    .modules = &.{ .hydro, .mhd, .radiation },
    .reconstruction = .ppm,
    .flux = .laxf,
    .timestepping = .rk2imex,
    .coords = .mink,
};
const SimM = koral.Sim(mink_cfg);
const LM = SimM.Layout;
const SimP = koral.Sim(koral.config.puffy);

const puffy = koral.problems.puffy;
const invert = koral.solve.invert;
const invert_rad = koral.solve.invert_rad;
const implicit = koral.solve.implicit;
const radforce = koral.physics.radforce;
const hydro = koral.physics.hydro;

const mink_nx = 24;
const mink_ny = 20;
const mink_nz = 12;

fn minkGrid() Grid {
    return Grid.init(.{
        .nx = mink_nx,
        .ny = mink_ny,
        .nz = mink_nz,
        .ng = 3,
        .minx = 0,
        .maxx = 1,
        .miny = 0,
        .maxy = 1,
        .minz = 0,
        .maxz = 1,
    });
}

fn minkOpts(comm: ?*Comm, decomp: ?koral.comm.Decomp) SimM.Options {
    return .{
        .coords = .mink,
        .gam = 5.0 / 3.0,
        .floors = invert.FloorParams.puffy,
        .rad = invert_rad.RadParams.puffy,
        .opac = radforce.Params.puffy(),
        .implicit = implicit.ImplicitParams.puffy,
        .bc_x = .copy,
        .bc_y = .copy,
        .bc_z = .periodic,
        .comm = comm,
        .decomp = decomp,
    };
}

/// Fill the domain from GLOBAL coordinates (zc is izoff-aware), so every
/// decomposition initializes the same physical state bit-for-bit.
fn initMinkBox(s: *SimM) !void {
    const tau = 6.28;
    var iz: i64 = 0;
    while (iz < s.nzi()) : (iz += 1) {
        var iy: i64 = 0;
        while (iy < s.nyi()) : (iy += 1) {
            var ix: i64 = 0;
            while (ix < s.nxi()) : (ix += 1) {
                const fx = s.grid.xc(ix);
                const fy = s.grid.yc(iy);
                const fz = s.grid.zc(iz);
                const bump = 1.0 + 0.3 * @sin(tau * fx) * @cos(tau * fy) * @cos(tau * fz);
                var pp: [SimM.nv]f64 = @splat(0);
                pp[LM.index(.rho)] = 1.0 * bump;
                pp[LM.index(.uu)] = 0.1 * bump;
                pp[LM.index(.vx)] = 0.05 * @sin(tau * fy);
                pp[LM.index(.vy)] = 0.02 * @cos(tau * fx);
                pp[LM.index(.vz)] = 0.03 * @sin(tau * fz);
                pp[LM.index(.entr)] = hydro.sFromU(pp[LM.index(.rho)], pp[LM.index(.uu)], 5.0 / 3.0);
                pp[LM.index(.b1)] = 0.01;
                pp[LM.index(.b2)] = 0.02 * bump;
                pp[LM.index(.b3)] = 0.015 * @cos(tau * fz);
                pp[LM.index(.ee)] = 0.2 * bump;
                try s.initCell(ix, iy, iz, pp);
            }
        }
    }
    s.exchangeHalos(); // no-op serially; fills z-ghost p before setBc's p2u path
    try s.setBc(0.0, true);
    s.initTimestepGuess();
}

const Gate = struct {
    rank: usize,
    name: []const u8,
    nfail: usize = 0,

    fn expect(g: *Gate, ok: bool, comptime fmt: []const u8, args: anytype) void {
        if (ok) return;
        g.nfail += 1;
        if (g.nfail <= 10) {
            std.debug.print("FAIL[{s} r{d}]: ", .{ g.name, g.rank });
            std.debug.print(fmt ++ "\n", args);
        }
    }

    fn finish(g: *Gate) bool {
        if (g.nfail == 0) {
            std.debug.print("PASS[{s} r{d}]\n", .{ g.name, g.rank });
            return true;
        }
        std.debug.print("FAIL[{s} r{d}]: {d} mismatches\n", .{ g.name, g.rank, g.nfail });
        return false;
    }
};

// ---- gate 2: 1-rank MPI ≡ serial, bitwise ---------------------------------

fn gate2(allocator: std.mem.Allocator, comm: *Comm) !bool {
    var g = Gate{ .rank = comm.rank(), .name = "gate2 1-rank≡serial" };
    const grid = minkGrid();

    var ref = try SimM.init(allocator, grid, minkOpts(null, null));
    defer ref.deinit();
    const dc = try koral.comm.decompose(grid, 1, 0);
    var s = try SimM.init(allocator, dc.local, minkOpts(comm, dc));
    defer s.deinit();

    try initMinkBox(&ref);
    try initMinkBox(&s);
    for (0..3) |_| {
        try ref.step(null); // own CFL dt on both — the dt path is part of the gate
        try s.step(null);
    }
    g.expect(ref.t == s.t, "t {e} vs {e}", .{ ref.t, s.t });
    g.expect(ref.dt == s.dt, "dt {e} vs {e}", .{ ref.dt, s.dt });
    g.expect(ref.tstepdenmax == s.tstepdenmax, "tstepdenmax", .{});
    g.expect(ref.tstepdenmin == s.tstepdenmin, "tstepdenmin", .{});
    g.expect(ref.n_radimp_failures == s.n_radimp_failures, "radimp failures", .{});
    for (ref.p.data, s.p.data, 0..) |a, b, i| {
        if (!(a == b or (std.math.isNan(a) and std.math.isNan(b))))
            g.expect(false, "p[{d}] {e} vs {e}", .{ i, a, b });
    }
    for (ref.u.data, s.u.data, 0..) |a, b, i| {
        if (!(a == b or (std.math.isNan(a) and std.math.isNan(b))))
            g.expect(false, "u[{d}] {e} vs {e}", .{ i, a, b });
    }
    for (ref.flags, s.flags, 0..) |a, b, i| {
        if (a != b) g.expect(false, "flag[{d}] {d} vs {d}", .{ i, a, b });
    }
    return g.finish();
}

// ---- gate 3a: zero-copy channel round-trip (property test) ----------------

fn gate3Roundtrip(allocator: std.mem.Allocator, comm: *Comm, dc: koral.comm.Decomp) !bool {
    var g = Gate{ .rank = comm.rank(), .name = "gate3a roundtrip" };
    const lg = dc.local;
    const NV = 13;
    var f = try koral.Field(NV).init(allocator, lg);
    defer f.deinit();

    // Stamp every storage slot with an exact-integer encoding of its GLOBAL
    // z-plane, storage row/col, and variable. Ghost-plane stamps use the
    // raw (unwrapped) global index, so a successful exchange must replace
    // them with the neighbor's wrapped-domain stamps.
    const tnz: i64 = @intCast(dc.global.nz);
    const sx: i64 = @intCast(lg.sx());
    const sy: i64 = @intCast(lg.sy());
    const sz: i64 = @intCast(lg.sz());
    const stamp = struct {
        fn at(gz: i64, jy: i64, jx: i64, iv: usize) f64 {
            return @floatFromInt(gz * 1_000_000 + jy * 10_000 + jx * 100 + @as(i64, @intCast(iv)));
        }
    }.at;
    {
        var k: usize = 0;
        var jz: i64 = 0;
        while (jz < sz) : (jz += 1) {
            const gz = lg.izoff + jz - @as(i64, @intCast(lg.ngz));
            var jy: i64 = 0;
            while (jy < sy) : (jy += 1) {
                var jx: i64 = 0;
                while (jx < sx) : (jx += 1) {
                    for (0..NV) |iv| {
                        f.data[k] = stamp(gz, jy, jx, iv);
                        k += 1;
                    }
                }
            }
        }
    }

    try comm.bindExchange(f.data, lg, NV);
    comm.exchange();
    comm.unbindExchange();

    var k: usize = 0;
    var jz: i64 = 0;
    while (jz < sz) : (jz += 1) {
        const ngz: i64 = @intCast(lg.ngz);
        const gz_raw = lg.izoff + jz - ngz;
        const is_ghost = jz < ngz or jz >= sz - ngz;
        const gz = if (is_ghost) @mod(gz_raw, tnz) else gz_raw;
        var jy: i64 = 0;
        while (jy < sy) : (jy += 1) {
            var jx: i64 = 0;
            while (jx < sx) : (jx += 1) {
                for (0..NV) |iv| {
                    const want = stamp(gz, jy, jx, iv);
                    if (f.data[k] != want)
                        g.expect(false, "slot jz={d} jy={d} jx={d} iv={d}: got {e} want {e}", .{ jz, jy, jx, iv, f.data[k], want });
                    k += 1;
                }
            }
        }
    }
    return g.finish();
}

// ---- gate 3b/5 shared comparison helpers ----------------------------------

/// Bitwise compare of the central x/y band ([margin, n-margin)) over ALL
/// local z, mapping local iz to the reference's global index.
fn compareCentralBitwise(
    comptime SimT: type,
    g: *Gate,
    ref: *const SimT,
    s: *const SimT,
    tok: i64,
    margin: i64,
) void {
    var iz: i64 = 0;
    while (iz < s.nzi()) : (iz += 1) {
        const gz = tok + iz;
        var iy: i64 = margin;
        while (iy < s.nyi() - margin) : (iy += 1) {
            var ix: i64 = margin;
            while (ix < s.nxi() - margin) : (ix += 1) {
                for (0..SimT.nv) |iv| {
                    const a = ref.p.get(iv, ix, iy, gz);
                    const b = s.p.get(iv, ix, iy, iz);
                    if (a != b)
                        g.expect(false, "central p iv={d} ({d},{d},{d}|gz{d}): {e} vs {e}", .{ iv, ix, iy, iz, gz, a, b });
                }
            }
        }
    }
}

/// Two-zone field-scale tolerance compare over the whole local domain: per
/// variable, |Δ| ≤ rtol · max|ref|, with a tighter rtol in the central x/y
/// band ([margin, n-margin)) than near the x/y edges. The edge band is
/// where the LEGITIMATE serial-vs-MPI divergence lives (§1.1-8): serial's
/// corner machinery and the exchange + C middle-BC pass construct the
/// x/y-ghost × z-ghost corner regions differently, and that difference
/// propagates a few cells inward per step. Central cells only see
/// bit-identical exchanged planes (plus, for the dynamo, the ring-reduced
/// scale height's ulp regrouping).
fn compareDomainTolerance(
    comptime SimT: type,
    g: *Gate,
    ref: *const SimT,
    s: *const SimT,
    tok: i64,
    margin: i64,
    rtol_central: f64,
    rtol_edge: f64,
    max_edge_frac: f64,
    /// Variables at iv ≥ loose_iv_start (the radiation flux tail in the
    /// 13-var layout) get only the plain `rtol_loose` bound instead of the
    /// fraction gate: F is a stiff, nearly force-balanced SMALL field whose
    /// seam response legitimately reaches ~0.1 of its own (already tiny)
    /// field scale — a genuine-decomposition effect, not a bug signal. Pass
    /// SimT.nv to disable.
    loose_iv_start: usize,
    rtol_loose: f64,
) void {
    var scale: [SimT.nv]f64 = @splat(1e-300);
    var gz: i64 = 0;
    while (gz < ref.nzi()) : (gz += 1) {
        var iy: i64 = 0;
        while (iy < ref.nyi()) : (iy += 1) {
            var ix: i64 = 0;
            while (ix < ref.nxi()) : (ix += 1) {
                for (0..SimT.nv) |iv| {
                    const v = @abs(ref.p.get(iv, ix, iy, gz));
                    if (v > scale[iv]) scale[iv] = v;
                }
            }
        }
    }
    var edge_total: usize = 0;
    var edge_over: usize = 0;
    var over_iv: [SimT.nv]usize = @splat(0);
    var maxdev_iv: [SimT.nv]f64 = @splat(0);
    var iz: i64 = 0;
    while (iz < s.nzi()) : (iz += 1) {
        var iy: i64 = 0;
        while (iy < s.nyi()) : (iy += 1) {
            var ix: i64 = 0;
            while (ix < s.nxi()) : (ix += 1) {
                const central = ix >= margin and ix < s.nxi() - margin and
                    iy >= margin and iy < s.nyi() - margin;
                for (0..SimT.nv) |iv| {
                    const a = ref.p.get(iv, ix, iy, tok + iz);
                    const b = s.p.get(iv, ix, iy, iz);
                    // NaN where the reference is finite is always a failure.
                    if (std.math.isFinite(a) and !std.math.isFinite(b)) {
                        g.expect(false, "non-finite p iv={d} ({d},{d},{d}): {e} vs {e}", .{ iv, ix, iy, iz, a, b });
                        continue;
                    }
                    const dev = @abs(a - b) / scale[iv];
                    if (dev > maxdev_iv[iv]) maxdev_iv[iv] = dev;
                    if (iv >= loose_iv_start) {
                        if (dev > rtol_loose)
                            g.expect(false, "loose p iv={d} ({d},{d},{d}): {e} vs {e} (dev/scale {e})", .{ iv, ix, iy, iz, a, b, dev });
                        continue;
                    }
                    const ok = dev <= (if (central) rtol_central else rtol_edge);
                    if (central) {
                        if (!ok)
                            g.expect(false, "tol p iv={d} ({d},{d},{d}) central: {e} vs {e} (scale {e})", .{ iv, ix, iy, iz, a, b, scale[iv] });
                    } else {
                        // Edge zone: gate the FRACTION of violations, not each
                        // one — isolated u2p/fixup/implicit branch flips seeded
                        // by the corner-class differences produce arbitrary-
                        // magnitude outliers at a handful of cells, which is
                        // C-faithful behavior; a broken exchange makes the
                        // whole band (and the central zone) wrong.
                        edge_total += 1;
                        if (!ok) {
                            edge_over += 1;
                            over_iv[iv] += 1;
                        }
                    }
                }
            }
        }
    }
    if (edge_total > 0) {
        const frac = @as(f64, @floatFromInt(edge_over)) / @as(f64, @floatFromInt(edge_total));
        if (frac > max_edge_frac) {
            g.expect(false, "edge-band violations {d}/{d} ({d:.4}) exceed allowed fraction {d:.4}", .{ edge_over, edge_total, frac, max_edge_frac });
            for (0..SimT.nv) |iv| {
                if (over_iv[iv] > 0)
                    std.debug.print("  [r{d}] iv={d}: {d} over, max dev/scale {e:.2}\n", .{ g.rank, iv, over_iv[iv], maxdev_iv[iv] });
            }
        }
    }
}

// ---- reduction gates: the collectives themselves --------------------------
//
// The field comparisons above are blind to the collectives: they force dt
// from the in-process serial reference and compare only `p`, so deleting
// the end-of-step Allreduce (or half the scale-height ring reduce) leaves
// every one of them green. Two independent checks close that, and BOTH are
// needed:
//
//  * ranks-agree — folds MAX and MIN of the same buffer and requires both
//    to equal the local value. Catches a MISSING fold (ranks keep their own
//    slab-local values) but is vacuous if the fold itself is a no-op, and
//    blind to any error that is identical on every rank.
//  * matches-serial — compares against the decomposition-independent
//    ground truth. Catches a no-op/identity fold and the φ-uniform case
//    where every rank is wrong in the same way (e.g. a dropped Allreduce
//    operand leaving scaleth short by a factor of √ntz on all ranks).

/// Every rank must hold bit-identical values in `values`.
fn expectRanksAgree(
    g: *Gate,
    comm: *Comm,
    allocator: std.mem.Allocator,
    what: []const u8,
    values: []const f64,
) !void {
    const hi = try allocator.dupe(f64, values);
    defer allocator.free(hi);
    const lo = try allocator.dupe(f64, values);
    defer allocator.free(lo);
    comm.allreduceMax(hi);
    comm.allreduceMin(lo);
    for (values, hi, lo, 0..) |v, h, l, i| {
        if (!(v == h and v == l))
            g.expect(false, "{s}[{d}] disagrees across ranks: local {e}, ring max {e}, ring min {e}", .{ what, i, v, h, l });
    }
}

/// The reduced values must match the serial reference within `rtol`
/// (relative to the reference's own magnitude).
fn expectMatchesSerial(
    g: *Gate,
    what: []const u8,
    ref_vals: []const f64,
    got: []const f64,
    rtol: f64,
) void {
    for (ref_vals, got, 0..) |a, b, i| {
        const denom = @max(@abs(a), 1e-300);
        if (!(@abs(a - b) / denom <= rtol))
            g.expect(false, "{s}[{d}] != serial: {e} vs {e} (rel {e}, bound {e})", .{ what, i, a, b, @abs(a - b) / denom, rtol });
    }
}

// ---- gate 3b: MINK slab vs serial — central bitwise, seams at tolerance ---

fn gate3Mink(allocator: std.mem.Allocator, comm: *Comm, dc: koral.comm.Decomp) !bool {
    var g = Gate{ .rank = comm.rank(), .name = "gate3b mink-vs-serial" };
    const grid = minkGrid();

    var ref = try SimM.init(allocator, grid, minkOpts(null, null));
    defer ref.deinit();
    try initMinkBox(&ref);
    var s = try SimM.init(allocator, dc.local, minkOpts(comm, dc));
    defer s.deinit();
    try initMinkBox(&s);

    // The init dt guess agrees to ulps, not bitwise: C's min-size scan is
    // first-seen-quirky (sim.zig init, bug-compatible with finite.c:1946),
    // so a slab's "first dz" is a different global cell whose spacing can
    // differ by an ulp on a non-dyadic grid. All RANKS agree exactly (the
    // MIN fold); only serial-vs-slab may differ in the last bits.
    const tsd_rel = @abs(ref.tstepdenmax - s.tstepdenmax) / ref.tstepdenmax;
    g.expect(tsd_rel <= 1e-12, "init tstepdenmax {e} vs {e}", .{ ref.tstepdenmax, s.tstepdenmax });

    // Two forced-dt steps (serial's own choices, replayed on the slab).
    for (0..2) |_| {
        const dt = ref.cflDt();
        try ref.step(dt);
        try s.step(dt);
    }
    // The load-bearing claim: central cells BITWISE across the φ seams
    // (their whole dependency cone saw only bit-identical exchanged
    // planes); the x/y-edge band only at a coarse field-scale bound. The
    // x/y-ghost × z-ghost corners are built from DIFFERENT sources by
    // design (§1.1-8): serial's x-z tube copies the x-ghost value at the
    // z-domain edge, C's MPI middle pass (mirrored here) applies the x-BC
    // at the exchanged z-ghost plane — for this box's strong cos(2πz)
    // variation the two differ at O(field), so corner-adjacent cells drift
    // to ~5e-3 of field scale within two steps. 5e-2 still catches a
    // broken exchange (wrong plane ⇒ O(1)).
    compareCentralBitwise(SimM, &g, &ref, &s, dc.tok, 8);
    compareDomainTolerance(SimM, &g, &ref, &s, dc.tok, 8, 1e-12, 5e-2, 0.02, SimM.nv, 0);

    // The end-of-step folded Allreduce (sim.zig reduceStepGlobals). This box
    // varies strongly in φ (cos 2πz), so a rank's slab-local wavespeed
    // extremum is genuinely different from the global one — which makes
    // both a missing fold and an identity fold detectable here.
    const cfl = [2]f64{ s.tstepdenmax, s.tstepdenmin };
    try expectRanksAgree(&g, comm, allocator, "tstepden", cfl[0..]);
    expectMatchesSerial(&g, "tstepden", &.{ ref.tstepdenmax, ref.tstepdenmin }, cfl[0..], 1e-6);
    // The init MIN fold (sim.zig min_dx/min_dy/min_dz -> initTimestepGuess).
    const mins = [3]f64{ s.min_dx, s.min_dy, s.min_dz };
    try expectRanksAgree(&g, comm, allocator, "min_d", mins[0..]);
    return g.finish();
}

// ---- gate 4: run-to-run determinism at fixed decomposition ----------------

fn gate4(allocator: std.mem.Allocator, comm: *Comm, dc: koral.comm.Decomp) !bool {
    var g = Gate{ .rank = comm.rank(), .name = "gate4 determinism" };

    var a = try SimM.init(allocator, dc.local, minkOpts(comm, dc));
    try initMinkBox(&a);
    for (0..4) |_| try a.step(null);
    // snapshot, then rerun from scratch
    const pa = try allocator.dupe(f64, a.p.data);
    defer allocator.free(pa);
    const ua = try allocator.dupe(f64, a.u.data);
    defer allocator.free(ua);
    const ta = a.t;
    a.deinit(); // frees the exchange binding so run 2 can bind

    var b = try SimM.init(allocator, dc.local, minkOpts(comm, dc));
    defer b.deinit();
    try initMinkBox(&b);
    for (0..4) |_| try b.step(null);

    g.expect(ta == b.t, "t {e} vs {e}", .{ ta, b.t });
    for (pa, b.p.data, 0..) |x, y, i| {
        if (!(x == y or (std.math.isNan(x) and std.math.isNan(y))))
            g.expect(false, "p[{d}] {e} vs {e}", .{ i, x, y });
    }
    for (ua, b.u.data, 0..) |x, y, i| {
        if (!(x == y or (std.math.isNan(x) and std.math.isNan(y))))
            g.expect(false, "u[{d}] {e} vs {e}", .{ i, x, y });
    }
    return g.finish();
}

// ---- gate 5: PUFFY wedge (full production physics) vs serial --------------

fn puffyOpts(comm: ?*Comm, decomp: ?koral.comm.Decomp) SimP.Options {
    return .{
        .coords = .mks2,
        .mp = puffy.mp,
        .gam = puffy.gam,
        .floors = invert.FloorParams.puffy,
        .rad = invert_rad.RadParams.puffy,
        .opac = radforce.Params.puffy(),
        .implicit = implicit.ImplicitParams.puffy,
        .correct_polaraxis = true,
        .nccorrectpolar = 2,
        .radviscosity = true,
        .dynamo = true,
        .bc_x = .specific,
        .bc_y = .specific,
        .bc_z = .periodic,
        .specific_bc = &puffy.Bc(SimP).calc,
        .comm = comm,
        .decomp = decomp,
    };
}

fn gate5(allocator: std.mem.Allocator, comm: *Comm) !bool {
    var g = Gate{ .rank = comm.rank(), .name = "gate5 puffy-wedge" };
    const grid = puffy.makeGridNz(48, 44, 12);
    const dc = try koral.comm.decompose(grid, comm.size(), comm.rank());

    var ref = try SimP.init(allocator, grid, puffyOpts(null, null));
    defer ref.deinit();
    const fac_ref = try puffy.initAll(SimP, &ref);
    ref.initTimestepGuess();

    var s = try SimP.init(allocator, dc.local, puffyOpts(comm, dc));
    defer s.deinit();
    const fac_s = try puffy.initAll(SimP, &s);
    s.initTimestepGuess();

    // Init state: torus + calc_BfromA + β-normalization. The global β-max
    // sits at an x-edge cell touched by the corner-class differences, so
    // the BETANORM fac itself shifts by ~2e-7 relative — scaling EVERY B
    // (and its conserveds) uniformly, at full field scale in the torus
    // bulk (C-MPI vs C-serial has the identical property). 1e-6 rides just
    // above it; non-B variables are untouched by fac and would surface any
    // real init bug far above this bound.
    compareDomainTolerance(SimP, &g, &ref, &s, dc.tok, 8, 1e-6, 1e-6, 0.01, SimP.nv, 0);

    // The production sequence at test scale (threading_tests P1 gate ×12φ):
    // the tiny guess-dt startup step, then a CFL-sized step, both forced
    // from the serial reference's choices.
    for (0..2) |_| {
        const dt = 1.0 / ref.tstepdenmax;
        try ref.step(dt);
        try s.step(dt);
    }
    // Post-step there is NO seam-immune zone in PUFFY: the scale height is
    // a GLOBAL per-radius sum (feeding the dynamo's ΔA_φ at every θ) and
    // radviscosity widens the stencil, so the x/y-edge corner-class
    // differences reach seam-adjacent planes at all x/y within two steps
    // (the F-fields even pick up the slab-periodic seam imprint — exactly
    // what C-MPI does vs C-serial). The gate is therefore physics
    // equivalence, fraction-gated: ≥99% of (cell,var) slots within 1e-3 of
    // field scale, no NaNs. Gross breakage (a wrong exchange plane) is O(1)
    // over the whole domain and fails this instantly; the bitwise-level
    // machinery checks live in gates 2/3 where no global coupling smears
    // them.
    compareDomainTolerance(SimP, &g, &ref, &s, dc.tok, std.math.maxInt(i32), 0, 1e-3, 0.01, comptime SimP.Layout.index(.fx), 2e-1);

    // ---- the PUFFY-only reductions -------------------------------------
    // scaleth is the one MPI-specific *physics* reduction in the change
    // (dynamo.zig calcScaleHeightRing: two pre-√ Allreduce(SUM)s of the
    // per-radius partial sums). Nothing else in the ladder can see it —
    // and because this torus is φ-uniform, every rank computes the SAME
    // wrong answer if an operand is dropped, so ranks-agree is blind and
    // the serial comparison is what actually catches it.
    try expectRanksAgree(&g, comm, allocator, "scaleth", s.scaleth);
    expectMatchesSerial(&g, "scaleth", ref.scaleth, s.scaleth, 1e-6);
    // BETANORM's global max (puffy.zig postinit): a missing fold makes each
    // rank normalize B by its own slab maximum.
    try expectRanksAgree(&g, comm, allocator, "betanorm_fac", &.{fac_s});
    expectMatchesSerial(&g, "betanorm_fac", &.{fac_ref}, &.{fac_s}, 1e-6);
    // And the per-step fold again, here under the full production physics.
    try expectRanksAgree(&g, comm, allocator, "tstepden", &.{ s.tstepdenmax, s.tstepdenmin });
    return g.finish();
}

// ---- driver ----------------------------------------------------------------

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;

    try Comm.initWorld();
    defer Comm.finalizeWorld();
    const wsize = Comm.worldSize();
    var comm = try Comm.init(wsize);
    defer comm.deinit();

    if (!koral.comm.enabled and comm.rank() == 0)
        std.debug.print("mpi-gates: serial build (-Dmpi absent) — only the degenerate 1-rank gates run\n", .{});

    var ok = true;
    if (wsize == 1) {
        ok = try gate2(allocator, &comm) and ok;
    } else {
        if (mink_nz % wsize != 0) {
            std.debug.print("mpi-gates: run with -n dividing {d} (1,2,3,4)\n", .{mink_nz});
            return error.BadRankCount;
        }
        const dc = try koral.comm.decompose(minkGrid(), wsize, comm.rank());
        ok = try gate3Roundtrip(allocator, &comm, dc) and ok;
        ok = try gate3Mink(allocator, &comm, dc) and ok;
        ok = try gate4(allocator, &comm, dc) and ok;
        ok = try gate5(allocator, &comm) and ok;
    }
    if (!ok) return error.GateFailed;
    if (comm.rank() == 0) std.debug.print("mpi-gates: all gates green at {d} rank(s)\n", .{wsize});
}
