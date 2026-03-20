const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const common = b.createModule(.{ .root_source_file = b.path("src/ggml-cpu/ggml-common.zig") });

    // Static library
    const lib = b.addLibrary(.{
        .name = "llama_zig",
        .linkage = .static,
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/ggml-cpu/root.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "ggml-common", .module = common },
            },
        }),
    });
    b.installArtifact(lib);

    const cmake_build_dir = b.option([]const u8, "cmake-build-dir", "Path to existing cmake build dir");
    const lib_path, const cmake_step = buildLlama(b, cmake_build_dir);

    // Helper to add a test for a source file
    const sources = .{
        .{ "quants", "src/ggml-cpu/quants.zig" },
        .{ "repack", "src/ggml-cpu/repack.zig" },
    };

    const test_step = b.step("test", "Run parity tests + benchmarks");

    inline for (sources) |src| {
        const t = b.addTest(.{
            .root_module = b.createModule(.{
                .root_source_file = b.path(src[1]),
                .target = target,
                .optimize = optimize,
                .link_libc = true,
                .imports = &.{
                    .{ .name = "ggml-common", .module = common },
                },
            }),
        });
        t.addLibraryPath(.{ .cwd_relative = lib_path });
        t.linkSystemLibrary("ggml-cpu");
        t.linkSystemLibrary("ggml-base");
        t.linkSystemLibrary("ggml");
        if (cmake_step) |step| t.step.dependOn(step);
        test_step.dependOn(&b.addRunArtifact(t).step);
    }
}

fn buildLlama(b: *std.Build, cmake_build_dir: ?[]const u8) struct { []const u8, ?*std.Build.Step } {
    if (cmake_build_dir) |dir| {
        return .{ b.fmt("{s}/bin", .{dir}), null };
    }

    const build_dir = b.pathFromRoot("../build");
    const cmake_configure = b.addSystemCommand(&.{
        "cmake", "-B", build_dir,
        "-DGGML_CUDA=OFF", "-DGGML_METAL=OFF",
        "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_EXAMPLES=OFF",
        "-DLLAMA_BUILD_TOOLS=OFF", "-DLLAMA_BUILD_SERVER=OFF",
    });
    cmake_configure.setCwd(b.path(".."));

    const cmake_build = b.addSystemCommand(&.{
        "cmake", "--build", build_dir, "--target", "llama", "-j",
    });
    cmake_build.step.dependOn(&cmake_configure.step);

    return .{ b.fmt("{s}/bin", .{build_dir}), &cmake_build.step };
}
