const std = @import("std");

/// What `-Dmpi` resolved to at configure time (MPI plan §3.3): the comptime
/// ABI family baked into build_options, and the lib dirs to link/rpath.
/// No include paths are ever fed to the compiler — the bindings are
/// hand-written externs (comm/mpi/core.zig); mpi.h is only *read* here to
/// detect the family.
const MpiConfig = struct {
    family: []const u8,
    lib_dirs: []const []const u8,
};

/// Probe order (plan §3.3): explicit -Dmpi-* overrides → `mpicc`
/// (`--showme:*` for Open MPI, `-show` for MPICH/Intel) → error with
/// instructions. Family: -Dmpi-family, else grep the found mpi.h for
/// MPICH_NUMVERSION / OMPI_MAJOR_VERSION.
fn probeMpi(
    b: *std.Build,
    family_opt: ?[]const u8,
    lib_opt: ?[]const u8,
    inc_opt: ?[]const u8,
) !MpiConfig {
    const io = b.graph.io;
    var lib_dirs: std.ArrayList([]const u8) = .empty;
    var inc_dirs: std.ArrayList([]const u8) = .empty;
    if (lib_opt) |d| try lib_dirs.append(b.allocator, d);
    if (inc_opt) |d| try inc_dirs.append(b.allocator, d);

    if (lib_dirs.items.len == 0 or (family_opt == null and inc_dirs.items.len == 0)) {
        const mpicc_found = b.findProgram(&.{"mpicc"}, &.{}) catch {
            // Say which half is actually missing. `-Dmpi-lib` alone still
            // lands here (the ABI family is undetectable without mpi.h or an
            // explicit family), and reporting it as "no -Dmpi-lib given"
            // sends the user to re-pass the flag they already passed.
            if (lib_dirs.items.len > 0) {
                std.debug.print(
                    "build.zig: -Dmpi-lib given, but the MPI ABI family could not be determined:\n" ++
                        "  no mpicc on PATH to probe, and neither -Dmpi-family nor -Dmpi-include was passed.\n" ++
                        "  Add -Dmpi-family=mpich|ompi (or -Dmpi-include=<dir-with-mpi.h> to detect it).\n",
                    .{},
                );
            } else {
                std.debug.print(
                    "build.zig: -Dmpi set but no mpicc found and no -Dmpi-lib given.\n" ++
                        "  Load the cluster's MPI module first, or pass -Dmpi-lib=<dir-with-libmpi> -Dmpi-family=mpich|ompi.\n",
                    .{},
                );
            }
            return error.MpiNotFound;
        };
        // findProgram resolves symlinks — but Open MPI's mpicc is a symlink
        // to opal_wrapper, which picks its personality from argv[0] and
        // errors when invoked under its real name. Re-append the basename.
        const mpicc = if (std.fs.path.dirname(mpicc_found)) |dir|
            b.pathJoin(&.{ dir, "mpicc" })
        else
            mpicc_found;
        // Open MPI answers --showme:*; MPICH/Intel answer -show. Each prints
        // a gcc-style flag line; we only harvest -L (link+rpath) and -I
        // (family detection).
        var outputs: std.ArrayList([]const u8) = .empty;
        for ([_][]const u8{ "--showme:link", "--showme:compile", "-show" }) |flag| {
            var code: u8 = 0;
            const out = b.runAllowFail(&.{ mpicc, flag }, &code, .ignore) catch continue;
            try outputs.append(b.allocator, out);
        }
        for (outputs.items) |out| {
            var it = std.mem.tokenizeAny(u8, out, " \t\r\n");
            while (it.next()) |tok| {
                if (tok.len > 2 and std.mem.startsWith(u8, tok, "-L"))
                    try lib_dirs.append(b.allocator, tok[2..]);
                if (tok.len > 2 and std.mem.startsWith(u8, tok, "-I"))
                    try inc_dirs.append(b.allocator, tok[2..]);
            }
        }
        // last-resort include guess for family detection: <mpicc>/../include
        if (inc_dirs.items.len == 0) {
            if (std.fs.path.dirname(mpicc)) |bindir| {
                if (std.fs.path.dirname(bindir)) |prefix| {
                    try inc_dirs.append(b.allocator, b.pathJoin(&.{ prefix, "include" }));
                }
            }
        }
    }
    if (lib_dirs.items.len == 0) {
        std.debug.print("build.zig: -Dmpi: could not determine libmpi's directory — pass -Dmpi-lib=<dir>.\n", .{});
        return error.MpiNotFound;
    }

    const family = family_opt orelse blk: {
        for (inc_dirs.items) |dir| {
            const hpath = b.pathJoin(&.{ dir, "mpi.h" });
            const src = std.Io.Dir.cwd().readFileAlloc(io, hpath, b.allocator, .limited(8 << 20)) catch continue;
            if (std.mem.indexOf(u8, src, "MPICH_NUMVERSION") != null) break :blk "mpich";
            if (std.mem.indexOf(u8, src, "OMPI_MAJOR_VERSION") != null) break :blk "ompi";
        }
        std.debug.print("build.zig: -Dmpi: could not detect the ABI family from mpi.h — pass -Dmpi-family=mpich|ompi.\n", .{});
        return error.MpiFamilyUnknown;
    };
    if (!std.mem.eql(u8, family, "mpich") and !std.mem.eql(u8, family, "ompi")) {
        std.debug.print("build.zig: -Dmpi-family must be 'mpich' or 'ompi' (got '{s}').\n", .{family});
        return error.MpiFamilyUnknown;
    }
    return .{ .family = family, .lib_dirs = lib_dirs.items };
}

/// Wire MPI into a module: libc (MPI headers' ABI expects it), the lib
/// dirs (+rpath so binaries find the module-loaded libmpi at runtime), and
/// -lmpi. Propagates transitively into every exe that imports the module.
fn linkMpi(m: *std.Build.Module, mc: MpiConfig) void {
    m.link_libc = true;
    for (mc.lib_dirs) |d| {
        m.addLibraryPath(.{ .cwd_relative = d });
        m.addRPath(.{ .cwd_relative = d });
    }
    m.linkSystemLibrary("mpi", .{ .use_pkg_config = .no });
}

pub fn build(b: *std.Build) !void {
    const io = b.graph.io;
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const use_mpi = b.option(bool, "mpi", "link system MPI (probes mpicc; see -Dmpi-family/-Dmpi-lib/-Dmpi-include)") orelse false;
    const mpi_family_opt = b.option([]const u8, "mpi-family", "MPI ABI family: mpich | ompi (default: detected from mpi.h)");
    const mpi_lib_opt = b.option([]const u8, "mpi-lib", "directory containing libmpi (default: parsed from mpicc)");
    const mpi_include_opt = b.option([]const u8, "mpi-include", "directory containing mpi.h — family detection only, never compiled against");
    const mpi_cfg: ?MpiConfig = if (use_mpi) try probeMpi(b, mpi_family_opt, mpi_lib_opt, mpi_include_opt) else null;
    const slow_tests = b.option(bool, "slow-tests", "run slow tests (convergence studies, soaks)") orelse false;

    // `-Dsilo` builds LLNL Silo 4.12 from source (PDB driver, no HDF5) via the
    // silo-zig package and links it into the problem executables so they can
    // export VisIt-openable `.silo` files (koral/io/silo.zig). Default off so
    // `zig build test` and Silo-less checkouts keep building — no external Silo
    // or VisIt install is needed.
    const enable_silo = b.option(bool, "silo", "build .silo export by compiling Silo from source (silo-zig)") orelse false;

    // The `silo` import wired into the koral module: the real from-source
    // wrapper when -Dsilo, else a tiny stub so `@import("silo")` resolves
    // without linking Silo or libc. The wrapper is a lazy dependency, so
    // `lazyDependency` triggers its (hash-pinned) Silo-source fetch only here
    // under -Dsilo; if it isn't cached yet it returns null and Zig fetches +
    // re-runs this build.
    const silo_module = if (enable_silo)
        (b.lazyDependency("silo", .{ .target = target, .optimize = optimize }) orelse return).module("silo")
    else
        b.createModule(.{
            .root_source_file = b.path("koral/io/silo_disabled.zig"),
            .target = target,
            .optimize = optimize,
        });

    const build_opts = b.addOptions();
    build_opts.addOption(bool, "mpi", use_mpi);
    build_opts.addOption([]const u8, "mpi_family", if (mpi_cfg) |mc| mc.family else "");
    build_opts.addOption(bool, "slow_tests", slow_tests);
    build_opts.addOption(bool, "silo", enable_silo);
    build_opts.addOption([]const u8, "golden_dir", b.pathFromRoot("tests/golden"));
    // Zig-generated regression baselines (tests/selfgolden), kept in a separate
    // tree from the C-oracle goldens so the provenance of a failure is never
    // ambiguous: tests/golden says "does this match KORAL C", tests/selfgolden
    // says "did our own numbers move".
    build_opts.addOption([]const u8, "selfgolden_dir", b.pathFromRoot("tests/selfgolden"));

    const koral = b.addModule("koral", .{
        .root_source_file = b.path("koral/koral.zig"),
        .target = target,
        .optimize = optimize,
    });
    koral.addOptions("build_options", build_opts);
    koral.addImport("silo", silo_module);
    if (mpi_cfg) |mc| linkMpi(koral, mc);

    // Library unit tests: `zig build test` (-Dtest-filter=... to select)
    const test_filters = b.option(
        []const []const u8,
        "test-filter",
        "filter unit tests by name substring",
    ) orelse &.{};
    const koral_tests = b.addTest(.{ .root_module = koral, .filters = test_filters });
    const run_koral_tests = b.addRunArtifact(koral_tests);
    const test_step = b.step("test", "run koral library tests");
    test_step.dependOn(&run_koral_tests.step);

    // Configure-time guard: the suite is registered by hand in koral.zig's
    // `test {}` block (there is no auto-registration), so a new, cleanly-
    // compiling `foo_tests.zig` would run *zero* tests unnoticed. Enforce the
    // contract: every theory gate (`koral/tests/*_tests.zig`) and every C-oracle
    // golden (`koral/tests/golden/*_golden_tests.zig`) must appear as an
    // `@import` in koral.zig, plus the one off-naming carrier
    // `testing/tubes.zig`. Fail the build loudly otherwise.
    {
        const koral_src = try b.build_root.handle.readFileAlloc(io, "koral/koral.zig", b.allocator, .limited(1 << 20));
        var any_missing = false;
        const warn = struct {
            fn header(first: *bool) void {
                if (!first.*) std.debug.print("build.zig: test file(s) not registered in koral.zig test block:\n", .{});
                first.* = true;
            }
        };
        for ([_][]const u8{ "tests", "tests/golden" }) |sub| {
            var tdir = try b.build_root.handle.openDir(io, b.fmt("koral/{s}", .{sub}), .{ .iterate = true });
            defer tdir.close(io);
            var tit = tdir.iterate();
            while (try tit.next(io)) |entry| {
                if (entry.kind != .file) continue;
                if (!std.mem.endsWith(u8, entry.name, "_tests.zig")) continue;
                const rel = b.fmt("{s}/{s}", .{ sub, entry.name });
                const needle = b.fmt("@import(\"{s}\")", .{rel});
                if (std.mem.indexOf(u8, koral_src, needle) == null) {
                    warn.header(&any_missing);
                    std.debug.print("  - {s}\n", .{rel});
                }
            }
        }
        if (std.mem.indexOf(u8, koral_src, "@import(\"testing/tubes.zig\")") == null) {
            warn.header(&any_missing);
            std.debug.print("  - testing/tubes.zig (off-naming carrier)\n", .{});
        }
        if (any_missing) {
            std.debug.print("Add `_ = @import(\"<path>\");` to koral.zig's test block.\n", .{});
            return error.UnregisteredTestFile;
        }
    }

    // Implicit-solver benchmark (always ReleaseFast; the dev-mode koral
    // module above keeps whatever -Doptimize the user picked)
    const koral_fast = b.createModule(.{
        .root_source_file = b.path("koral/koral.zig"),
        .target = target,
        .optimize = .ReleaseFast,
        .link_libc = true,
    });
    koral_fast.addOptions("build_options", build_opts);
    koral_fast.addImport("silo", silo_module);
    if (mpi_cfg) |mc| linkMpi(koral_fast, mc);
    const bench = b.addExecutable(.{
        .name = "bench_implicit",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/bench_implicit.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .link_libc = true,
            .imports = &.{.{ .name = "koral", .module = koral_fast }},
        }),
    });
    const run_bench = b.addRunArtifact(bench);
    run_bench.addArg(b.pathFromRoot("tests/golden/rad/rad_implicit.kgld"));
    if (b.args) |args| run_bench.addArgs(args);
    b.step("bench-implicit", "time the implicit solver: scalar vs SIMD Jacobian")
        .dependOn(&run_bench.step);

    // update-self-goldens: rewrite tests/selfgolden/** from THIS build. Needs
    // no C toolchain (unlike tools/gen_golden.sh) — the baseline is this
    // repository's own output.
    //
    // Deliberately built against the SAME `koral` module the test artifact
    // uses, so it honors -Doptimize instead of pinning ReleaseFast: generator
    // and checker then share a codegen path and the comparison can be gated at
    // machine precision. Regenerate with the same -Doptimize you test with
    // (i.e. plain defaults); a ReleaseFast baseline checked in Debug would
    // burn most of the tolerance on FP contraction differences.
    const gen_self = b.addExecutable(.{
        .name = "gen_self_golden",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/gen_self_golden.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{.{ .name = "koral", .module = koral }},
        }),
    });
    const run_gen_self = b.addRunArtifact(gen_self);
    run_gen_self.addArg(b.pathFromRoot("tests/selfgolden"));
    b.step("update-self-goldens", "regenerate tests/selfgolden/** from this build")
        .dependOn(&run_gen_self.step);

    // res2kdmp: convert a C KORAL serial restart (res####.head/.dat) into a Zig
    // KDMP checkpoint so a C-initialized run can be continued by `puffy
    // --restart`. Pure std.Io + byte-shuffling — no libc, honors -Doptimize.
    const res2kdmp = b.addExecutable(.{
        .name = "res2kdmp",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/res2kdmp.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{.{ .name = "koral", .module = koral }},
        }),
    });
    b.installArtifact(res2kdmp);
    const run_res2kdmp = b.addRunArtifact(res2kdmp);
    if (b.args) |args| run_res2kdmp.addArgs(args);
    b.step("res2kdmp", "convert a C restart (res####.head/.dat) to a KDMP checkpoint")
        .dependOn(&run_res2kdmp.step);

    // kdmp2silo: convert KDMP checkpoints to VisIt-openable .silo files
    // (MPI plan §8.2 Phase A — under MPI the run writes KDMP only; this
    // serial tool replays them through io/silo.zig). Useful only with
    // -Dsilo (it builds either way but refuses to run without it).
    const kdmp2silo = b.addExecutable(.{
        .name = "kdmp2silo",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/kdmp2silo.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{.{ .name = "koral", .module = koral }},
        }),
    });
    b.installArtifact(kdmp2silo);
    const run_kdmp2silo = b.addRunArtifact(kdmp2silo);
    if (b.args) |args| run_kdmp2silo.addArgs(args);
    b.step("kdmp2silo", "convert KDMP checkpoints to .silo for VisIt (needs -Dsilo)")
        .dependOn(&run_kdmp2silo.step);

    // qmri: MRI quality factors (Q_r/Q_θ/Q_φ) of KDMP snapshots — the
    // "is this resolution enough?" diagnostic (tools/qmri.zig).
    const qmri = b.addExecutable(.{
        .name = "qmri",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/qmri.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{.{ .name = "koral", .module = koral }},
        }),
    });
    b.installArtifact(qmri);
    const run_qmri = b.addRunArtifact(qmri);
    if (b.args) |args| run_qmri.addArgs(args);
    b.step("qmri", "MRI quality factors of KDMP snapshots")
        .dependOn(&run_qmri.step);

    // kdmp2png: GRRT-render a KDMP checkpoint into a PNG image — null
    // geodesics through the run's Kerr/MKS2 metric + radiative transfer from
    // its own opacities and M1 radiation field (koral/render/render.zig).
    // Build with -Doptimize=ReleaseFast; a 512² frame renders in ~a minute.
    const kdmp2png = b.addExecutable(.{
        .name = "kdmp2png",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/kdmp2png.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{.{ .name = "koral", .module = koral }},
        }),
    });
    b.installArtifact(kdmp2png);
    const run_kdmp2png = b.addRunArtifact(kdmp2png);
    if (b.args) |args| run_kdmp2png.addArgs(args);
    b.step("kdmp2png", "GRRT-render a KDMP checkpoint to PNG")
        .dependOn(&run_kdmp2png.step);

    // mpi-gates: the MPI validation-ladder harness (plan §10 gates 2-6).
    // Build with -Dmpi (ideally -Doptimize=ReleaseSafe) and run under
    // mpiexec at 1..4 ranks; each rank recomputes the serial reference
    // in-process and compares its slab, so no files are involved. A serial
    // build still compiles it (the Serial backend degenerates the gates).
    {
        const gates = b.addExecutable(.{
            .name = "mpi-gates",
            .root_module = b.createModule(.{
                .root_source_file = b.path("tools/mpi_gates.zig"),
                .target = target,
                .optimize = optimize,
                .imports = &.{.{ .name = "koral", .module = koral }},
            }),
        });
        const install_gates = b.addInstallArtifact(gates, .{});
        b.step("mpi-gates", "build the MPI validation gates harness (run via mpiexec)")
            .dependOn(&install_gates.step);
    }

    // One executable per koral/problems/<name>/main.zig
    var dir = try b.build_root.handle.openDir(io, "koral/problems", .{ .iterate = true });
    defer dir.close(io);
    var it = dir.iterate();
    while (try it.next(io)) |entry| {
        if (entry.kind != .directory) continue;
        const exe = b.addExecutable(.{
            .name = entry.name,
            .root_module = b.createModule(.{
                .root_source_file = b.path(b.fmt("koral/problems/{s}/main.zig", .{entry.name})),
                .target = target,
                .optimize = optimize,
                .imports = &.{.{ .name = "koral", .module = koral }},
            }),
        });
        // Silo is linked transitively: the koral module imports the `silo`
        // wrapper module (real when -Dsilo), which links the from-source
        // libsilo.a and libc into whatever imports it — no per-exe wiring.

        // One InstallArtifact shared by the default `install` step and the
        // per-problem `build <name>` step (two separate ones would silently
        // diverge if install options ever differ).
        const install = b.addInstallArtifact(exe, .{});
        b.getInstallStep().dependOn(&install.step);
        b.step(entry.name, b.fmt("build {s}", .{entry.name})).dependOn(&install.step);

        const run = b.addRunArtifact(exe);
        if (b.args) |args| run.addArgs(args);
        b.step(b.fmt("run-{s}", .{entry.name}), b.fmt("build & run {s}", .{entry.name}))
            .dependOn(&run.step);
    }
}
