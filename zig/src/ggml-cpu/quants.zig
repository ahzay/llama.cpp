const std = @import("std");
const T = @import("ggml-common");

pub export fn zig_vec_dot_q4_K_q8_K(
    n: c_int,
    s: *f32,
    bs: usize,
    vx: *const anyopaque,
    bx: usize,
    vy: *const anyopaque,
    by: usize,
    nrc: c_int,
) void {
    _ = bs;
    _ = bx;
    _ = by;
    _ = nrc;

    const nb: usize = @intCast(@divExact(n, T.QK_K));
    const xp: [*]const T.block_q4_K = @ptrCast(@alignCast(vx));
    const yp: [*]const T.block_q8_K = @ptrCast(@alignCast(vy));

    var sumf: f32 = 0;

    for (0..nb) |i| {
        const q4 = &xp[i].qs;
        const q8 = &yp[i].qs;

        var aux8: [T.QK_K]i8 = undefined;
        inline for (0..T.QK_K / 64) |j| {
            const raw: @Vector(32, u8) = q4[j * 32 ..][0..32].*;
            aux8[j * 64 ..][0..32].* = @bitCast(raw & @as(@Vector(32, u8), @splat(0x0F)));
            aux8[j * 64 + 32 ..][0..32].* = @bitCast(raw >> @as(@Vector(32, u8), @splat(4)));
        }

        var utmp: [4]u32 = undefined;
        @memcpy(std.mem.asBytes(&utmp)[0..12], &xp[i].scales);
        utmp[3] = ((utmp[2] >> 4) & T.kmask2) | (((utmp[1] >> 6) & T.kmask3) << 4);
        const uaux = utmp[1] & T.kmask1;
        utmp[1] = (utmp[2] & T.kmask2) | (((utmp[0] >> 6) & T.kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= T.kmask1;

        const scales: *const [8]u8 = @ptrCast(&utmp[0]);
        const mins: *const [8]u8 = @ptrCast(&utmp[2]);

        var sumi: i32 = 0;
        inline for (0..T.QK_K / 16) |j| {
            sumi += @as(i32, yp[i].bsums[j]) * @as(i32, mins[j / 2]);
        }

        var sumk: f32 = 0;
        var a_off: usize = 0;
        var q8_off: usize = 0;
        for (0..T.QK_K / 32) |is| {
            const scale: i32 = @intCast(scales[is]);
            const av: @Vector(32, i32) = @intCast(@as(@Vector(32, i8), aux8[a_off..][0..32].*));
            const qv: @Vector(32, i32) = @intCast(@as(@Vector(32, i8), q8[q8_off..][0..32].*));
            sumk += @as(f32, @floatFromInt(scale * @reduce(.Add, av * qv)));
            a_off += 32;
            q8_off += 32;
        }

        sumf += T.f16f32(xp[i].d) * yp[i].d * sumk;
        sumf -= T.f16f32(xp[i].dmin) * yp[i].d * @as(f32, @floatFromInt(sumi));
    }

    s.* = sumf;
}

//  Tests

extern fn ggml_cpu_init() void;
extern fn ggml_vec_dot_q4_K_q8_K(n: c_int, s: *f32, bs: usize, vx: *const anyopaque, bx: usize, vy: *const anyopaque, by: usize, nrc: c_int) void;
extern fn ggml_vec_dot_q4_K_q8_K_generic(n: c_int, s: *f32, bs: usize, vx: *const anyopaque, bx: usize, vy: *const anyopaque, by: usize, nrc: c_int) void;

fn fmtNs(ns: u64) struct { v: f64, u: []const u8 } {
    return if (ns >= 1_000_000) .{ .v = @as(f64, @floatFromInt(ns)) / 1e6, .u = "ms" } else if (ns >= 1_000) .{ .v = @as(f64, @floatFromInt(ns)) / 1e3, .u = "us" } else .{ .v = @as(f64, @floatFromInt(ns)), .u = "ns" };
}

var bench_sink: f32 = 0;

fn timeCall(comptime f: anytype, n: c_int, vx: *const anyopaque, vy: *const anyopaque) u64 {
    var result: f32 = 0;
    for (0..50) |_| f(n, &result, 0, vx, 0, vy, 0, 1);
    var t: [500]u64 = undefined;
    for (0..500) |i| {
        var timer = std.time.Timer.start() catch unreachable;
        f(n, &result, 0, vx, 0, vy, 0, 1);
        t[i] = timer.read();
    }
    bench_sink += result;
    std.mem.sortUnstable(u64, &t, {}, std.sort.asc(u64));
    return t[250];
}

test "q4_K vec_dot: parity + bench" {
    ggml_cpu_init();
    var prng = std.Random.DefaultPrng.init(0xDEAD_BEEF);
    const rand = prng.random();
    const alloc = std.heap.page_allocator;

    for ([_]usize{ 1, 4, 16, 64, 256 }) |nb| {
        const xb = try alloc.alloc(T.block_q4_K, nb);
        const yb = try alloc.alloc(T.block_q8_K, nb);
        defer alloc.free(xb);
        defer alloc.free(yb);

        for (xb) |*blk| {
            blk.d = @bitCast(@as(f16, @floatCast((rand.float(f32) - 0.5) * 2.0)));
            blk.dmin = @bitCast(@as(f16, @floatCast(rand.float(f32) * 0.5)));
            for (&blk.scales) |*v| v.* = rand.intRangeAtMost(u8, 0, 63);
            for (&blk.qs) |*v| v.* = rand.int(u8);
        }
        for (yb) |*blk| {
            blk.d = (rand.float(f32) - 0.5) * 2.0;
            for (&blk.qs) |*v| v.* = @as(i8, @intCast(@as(i32, rand.intRangeAtMost(u8, 0, 255)) - 128));
            for (&blk.bsums) |*v| v.* = @as(i16, @intCast(@as(i32, rand.intRangeAtMost(u16, 0, 65535)) - 32768));
        }

        const n: c_int = @intCast(nb * T.QK_K);
        var zig_r: f32 = 0;
        var ref_r: f32 = 0;
        zig_vec_dot_q4_K_q8_K(n, &zig_r, 0, @ptrCast(xb.ptr), 0, @ptrCast(yb.ptr), 0, 1);
        ggml_vec_dot_q4_K_q8_K_generic(n, &ref_r, 0, @ptrCast(xb.ptr), 0, @ptrCast(yb.ptr), 0, 1);

        if (@abs(zig_r - ref_r) / @max(@abs(ref_r), 1e-6) > 1e-3) {
            std.debug.print("MISMATCH nb={}: zig={d:.6} ref={d:.6}\n", .{ nb, zig_r, ref_r });
            return error.TestExpectedEqual;
        }
    }

    std.debug.print("\n{s:<12} {s:>10} {s:>10} {s:>10}\n", .{ "", "zig", "generic", "neon" });
    for ([_]usize{ 16, 64, 256, 1024 }) |nb| {
        const xb = try alloc.alloc(T.block_q4_K, nb);
        const yb = try alloc.alloc(T.block_q8_K, nb);
        defer alloc.free(xb);
        defer alloc.free(yb);
        for (xb) |*blk| {
            blk.d = @bitCast(@as(f16, @floatCast((rand.float(f32) - 0.5) * 2.0)));
            blk.dmin = @bitCast(@as(f16, @floatCast(rand.float(f32) * 0.5)));
            for (&blk.scales) |*v| v.* = rand.intRangeAtMost(u8, 0, 63);
            for (&blk.qs) |*v| v.* = rand.int(u8);
        }
        for (yb) |*blk| {
            blk.d = (rand.float(f32) - 0.5) * 2.0;
            for (&blk.qs) |*v| v.* = @as(i8, @intCast(@as(i32, rand.intRangeAtMost(u8, 0, 255)) - 128));
            for (&blk.bsums) |*v| v.* = @as(i16, @intCast(@as(i32, rand.intRangeAtMost(u16, 0, 65535)) - 32768));
        }
        const n: c_int = @intCast(nb * T.QK_K);
        const zt = fmtNs(timeCall(zig_vec_dot_q4_K_q8_K, n, @ptrCast(xb.ptr), @ptrCast(yb.ptr)));
        const gt = fmtNs(timeCall(ggml_vec_dot_q4_K_q8_K_generic, n, @ptrCast(xb.ptr), @ptrCast(yb.ptr)));
        const nt = fmtNs(timeCall(ggml_vec_dot_q4_K_q8_K, n, @ptrCast(xb.ptr), @ptrCast(yb.ptr)));
        std.debug.print("n={d:<8} {d:>7.2} {s} {d:>7.2} {s} {d:>7.2} {s}\n", .{ nb * T.QK_K, zt.v, zt.u, gt.v, gt.u, nt.v, nt.u });
    }
}
