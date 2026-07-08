const std = @import("std");

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const use_mpi = b.option(bool, "mpi", "link system MPI (not yet implemented)") orelse false;
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
    build_opts.addOption(bool, "slow_tests", slow_tests);
    build_opts.addOption(bool, "silo", enable_silo);
    build_opts.addOption([]const u8, "golden_dir", b.pathFromRoot("tests/golden"));

    const koral = b.addModule("koral", .{
        .root_source_file = b.path("koral/koral.zig"),
        .target = target,
        .optimize = optimize,
    });
    koral.addOptions("build_options", build_opts);
    koral.addImport("silo", silo_module);

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

    // One executable per koral/problems/<name>/main.zig
    const io = b.graph.io;
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
