const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const common = b.createModule(.{ .root_source_file = b.path("src/ggml-cpu/ggml-common.zig") });

    // Static library (used by cmake)
    b.installArtifact(b.addLibrary(.{
        .name = "llama_zig",
        .linkage = .static,
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/ggml-cpu/root.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{.{ .name = "ggml-common", .module = common }},
        }),
    }));

    // Tests — build llama via cmake, then link for parity checks
    const build_dir = b.pathFromRoot("../build");
    const cmake_configure = b.addSystemCommand(&.{
        "cmake", "-B", build_dir,
        "-DGGML_CUDA=OFF", "-DGGML_METAL=OFF",
        "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_EXAMPLES=OFF",
        "-DLLAMA_BUILD_TOOLS=OFF", "-DLLAMA_BUILD_SERVER=OFF",
    });
    cmake_configure.setCwd(b.path(".."));
    const cmake_build = b.addSystemCommand(&.{ "cmake", "--build", build_dir, "--target", "llama", "-j" });
    cmake_build.step.dependOn(&cmake_configure.step);

    const t = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("src/ggml-cpu/root.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{.{ .name = "ggml-common", .module = common }},
    }) });
    t.addLibraryPath(.{ .cwd_relative = b.fmt("{s}/bin", .{build_dir}) });
    for ([_][]const u8{ "ggml-cpu", "ggml-base", "ggml" }) |lib| t.linkSystemLibrary(lib);
    t.step.dependOn(&cmake_build.step);
    b.step("test", "Run parity tests + benchmarks").dependOn(&b.addRunArtifact(t).step);
}
