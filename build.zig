const std = @import("std");

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const use_mpi = b.option(bool, "mpi", "link system MPI (not yet implemented)") orelse false;
    const slow_tests = b.option(bool, "slow-tests", "run slow tests (convergence studies, soaks)") orelse false;

    // `-Dsilo` links LLNL Silo into the built problem executables so they can
    // export VisIt-openable `.silo` files (koral/io/silo.zig). Default off so
    // `zig build test` and Silo-less machines keep building. `-Dsilo-prefix`
    // points at the Silo install root; the default is VisIt 3.5's bundled Silo
    // (its `libsiloh5` — guaranteeing the files match what VisIt 3.5 reads).
    const enable_silo = b.option(bool, "silo", "link Silo for .silo export (needs VisIt / a Silo install)") orelse false;
    const silo_prefix = b.option(
        []const u8,
        "silo-prefix",
        "Silo install root (contains lib/libsiloh5.dylib)",
    ) orelse "/Applications/VisIt.app/Contents/Resources/3.5.0/darwin-arm64";

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

    // One executable per PROBLEMS/<name>/main.zig
    const io = b.graph.io;
    var dir = try b.build_root.handle.openDir(io, "PROBLEMS", .{ .iterate = true });
    defer dir.close(io);
    var it = dir.iterate();
    while (try it.next(io)) |entry| {
        if (entry.kind != .directory) continue;
        const exe = b.addExecutable(.{
            .name = entry.name,
            .root_module = b.createModule(.{
                .root_source_file = b.path(b.fmt("PROBLEMS/{s}/main.zig", .{entry.name})),
                .target = target,
                .optimize = optimize,
                .imports = &.{.{ .name = "koral", .module = koral }},
            }),
        });
        if (enable_silo) {
            const m = exe.root_module;
            m.link_libc = true;
            m.addLibraryPath(.{ .cwd_relative = b.fmt("{s}/lib", .{silo_prefix}) });
            // Silo's install name is `@rpath/lib/libsiloh5.dylib` (and its
            // hdf5/mpi/z deps are `@rpath/lib/...`), so a single rpath at the
            // prefix resolves the whole chain at load time.
            m.addRPath(.{ .cwd_relative = silo_prefix });
            m.linkSystemLibrary("siloh5", .{});
        }
        b.installArtifact(exe);

        const install = b.addInstallArtifact(exe, .{});
        b.step(entry.name, b.fmt("build {s}", .{entry.name})).dependOn(&install.step);

        const run = b.addRunArtifact(exe);
        if (b.args) |args| run.addArgs(args);
        b.step(b.fmt("run-{s}", .{entry.name}), b.fmt("build & run {s}", .{entry.name}))
            .dependOn(&run.step);
    }
}
