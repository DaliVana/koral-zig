//! Self-golden regression gates: re-run the pinned scenarios and compare them
//! against the committed baselines in `tests/selfgolden/`.
//!
//! **What a failure here means.** Not "the physics is wrong"; that is the
//! C-oracle goldens' job (`tests/golden/`, `koral/tests/golden/`). It means the
//! numbers this repository produces have moved since the baseline was recorded,
//! and only the author can say whether that was intended. A refactor that is
//! supposed to be behaviour-preserving failing this test is a bug; a deliberate
//! algorithm change failing it is expected, and the fix is to review the
//! reported deviation and then run `zig build update-self-goldens`.
//!
//! This is the net that has to hold once the Zig side stops matching koral_lite
//! bit-for-bit: at that point `tools/gen_golden.sh` can no longer produce an
//! authoritative C baseline, and the analytic theory tests, which check
//! identities and known solutions, not the assembled pipeline; are all that is
//! left of end-to-end coverage. See koral/testing/selfgolden.zig.
//!
//! The scenario definitions live in koral/testing/selfscenarios.zig and are
//! shared with the generator, so what is checked is by construction what was
//! recorded.

const std = @import("std");
const build_options = @import("build_options");
const selfgolden = @import("../testing/selfgolden.zig");
const scenarios = @import("../testing/selfscenarios.zig");

/// These are DIAGNOSIS thresholds, not pass/fail lines; the gate is bit
/// identity (see `selfgolden.Cmp.check`). Generator and checker build against
/// the same `koral` module at the same -Doptimize, so equality is achievable;
/// the threshold only separates "the numbers really changed" from "only the
/// last bits moved", which is what a toolchain or optimize-mode change looks
/// like. Both still fail; the author is told which.
const field_bound: f64 = 1e-12;
const scalar_bound: f64 = 1e-12;

fn checkScenario(comptime sc: scenarios.Scenario) !void {
    const a = std.testing.allocator;

    var base = try selfgolden.read(a, sc.file);
    defer base.deinit();

    var fresh = try scenarios.run(a, sc);
    defer fresh.deinit();

    std.debug.print("self-golden [{s}] vs baseline \"{s}\"\n", .{ sc.file, base.label });

    // (1) Shape. A mismatch means the scenario itself was edited without
    //     regenerating — report that plainly instead of as a numeric diff.
    try std.testing.expectEqual(base.hdr.nx, fresh.hdr.nx);
    try std.testing.expectEqual(base.hdr.ny, fresh.hdr.ny);
    try std.testing.expectEqual(base.hdr.nz, fresh.hdr.nz);
    try std.testing.expectEqual(base.hdr.ng, fresh.hdr.ng);
    try std.testing.expectEqual(base.hdr.nv, fresh.hdr.nv);
    try std.testing.expectEqual(base.hdr.ncell, fresh.hdr.ncell);
    try std.testing.expectEqual(base.hdr.nflags, fresh.hdr.nflags);
    try std.testing.expectEqual(base.hdr.n_scalar_recs, fresh.hdr.n_scalar_recs);
    try std.testing.expectEqual(base.hdr.n_field_recs, fresh.hdr.n_field_recs);

    // (2) Per-step scalars — the cheap "when did it diverge" signal. The two
    //     counters are integer-valued and compared exactly: a differing
    //     implicit-failure count is a branch change, not FP noise.
    for (0..base.hdr.n_scalar_recs) |istep| {
        const b = base.scalarsAt(istep);
        const f = fresh.scalars.items[istep * selfgolden.Scalars.count ..][0..selfgolden.Scalars.count];
        const bv = b.toArray();
        for (bv, f, 0..) |bval, fval, i| {
            const name = selfgolden.Scalars.name(i);
            if (std.mem.startsWith(u8, name, "n_")) {
                if (bval != fval) {
                    std.debug.print(
                        "self-golden [{s}] step {d}: {s} {d} -> {d}\n",
                        .{ sc.file, istep, name, bval, fval },
                    );
                    return error.SelfGoldenMismatch;
                }
                continue;
            }
            if (bval == fval) continue;
            const scale = @max(@abs(bval), 1e-300);
            const dev = @abs(bval - fval) / scale;
            std.debug.print(
                "self-golden {s} [{s}] step {d}: {s} {e:.17} -> {e:.17} (rel {e:.3})\n" ++
                    "  If intended, regenerate: zig build update-self-goldens\n",
                .{ if (dev > scalar_bound) "REGRESSION" else "DRIFT", sc.file, istep, name, bval, fval, dev },
            );
            return error.SelfGoldenMismatch;
        }
    }

    // (3) Full fields at the endpoints, incl. ghosts.
    for (0..base.hdr.n_field_recs) |i| {
        const bs = base.snapshot(i);
        const fs_base = i * selfgolden.File.fieldLen(fresh.hdr);
        const ncell: usize = @intCast(base.hdr.ncell);
        const nu = ncell * base.hdr.nv;
        const fu = fresh.fields.items[fs_base .. fs_base + nu];
        const fp = fresh.fields.items[fs_base + nu .. fs_base + 2 * nu];
        const ff = fresh.fields.items[fs_base + 2 * nu .. fs_base + selfgolden.File.fieldLen(fresh.hdr)];

        var cu = selfgolden.Cmp{};
        cu.addStrided(bs.u, fu, base.hdr.nv);
        var cp = selfgolden.Cmp{};
        cp.addStrided(bs.p, fp, base.hdr.nv);

        var label_u: [64]u8 = undefined;
        var label_p: [64]u8 = undefined;
        try cu.check(field_bound, try std.fmt.bufPrint(&label_u, "{s} step {d} u", .{ sc.file, bs.step }));
        try cp.check(field_bound, try std.fmt.bufPrint(&label_p, "{s} step {d} p", .{ sc.file, bs.step }));

        // Flags are integer branch outcomes (entropy / hd- / rad- / radimp-
        // fixup): any difference is a different code path taken, never noise.
        var nflag_diff: usize = 0;
        for (bs.flags, ff) |b, f| {
            if (b != f) nflag_diff += 1;
        }
        if (nflag_diff != 0) {
            std.debug.print(
                "self-golden [{s} step {d} flags]: {d}/{d} cells took a different branch\n" ++
                    "  If intended, regenerate: zig build update-self-goldens\n",
                .{ sc.file, bs.step, nflag_diff, bs.flags.len },
            );
            return error.SelfGoldenMismatch;
        }
    }
}

test "self-golden: PUFFY 32x30, 3 CFL steps matches the committed baseline" {
    try checkScenario(scenarios.puffy_fast);
}

test "self-golden: PUFFY 48x44, 8 CFL steps matches the committed baseline" {
    if (!build_options.slow_tests) return error.SkipZigTest;
    try checkScenario(scenarios.puffy_full);
}
