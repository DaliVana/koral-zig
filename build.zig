const std = @import("std");

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const use_mpi = b.option(bool, "mpi", "link system MPI (not yet implemented)") orelse false;
    const slow_tests = b.option(bool, "slow-tests", "run slow tests (convergence studies, soaks)") orelse false;

    const build_opts = b.addOptions();
    build_opts.addOption(bool, "mpi", use_mpi);
    build_opts.addOption(bool, "slow_tests", slow_tests);
    build_opts.addOption([]const u8, "golden_dir", b.pathFromRoot("tests/golden"));

    const koral = b.addModule("koral", .{
        .root_source_file = b.path("koral/koral.zig"),
        .target = target,
        .optimize = optimize,
    });
    koral.addOptions("build_options", build_opts);

    // Library unit tests: `zig build test`
    const koral_tests = b.addTest(.{ .root_module = koral });
    const run_koral_tests = b.addRunArtifact(koral_tests);
    const test_step = b.step("test", "run koral library tests");
    test_step.dependOn(&run_koral_tests.step);

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
        b.installArtifact(exe);

        const install = b.addInstallArtifact(exe, .{});
        b.step(entry.name, b.fmt("build {s}", .{entry.name})).dependOn(&install.step);

        const run = b.addRunArtifact(exe);
        if (b.args) |args| run.addArgs(args);
        b.step(b.fmt("run-{s}", .{entry.name}), b.fmt("build & run {s}", .{entry.name}))
            .dependOn(&run.step);
    }
}
