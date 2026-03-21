const std = @import("std");
const T = @import("ggml-common");

pub export fn zig_vec_dot_q2_K_q8_K(
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
    const xp: [*]const T.block_q2_K = @ptrCast(@alignCast(vx));
    const yp: [*]const T.block_q8_K = @ptrCast(@alignCast(vy));

    var sumf: f32 = 0;

    for (0..nb) |i| {
        const q2 = &xp[i].qs;
        const q8 = &yp[i].qs;
        const sc = &xp[i].scales;

        // mins × bsums
        var summs: i32 = 0;
        inline for (0..16) |j| {
            summs += @as(i32, yp[i].bsums[j]) * @as(i32, sc[j] >> 4);
        }

        const dall: f32 = yp[i].d * T.f16f32(xp[i].d);
        const dmin: f32 = yp[i].d * T.f16f32(xp[i].dmin);

        // Main dot: 2-bit quants × q8, with per-group 4-bit scales
        var isum: i32 = 0;
        var is: usize = 0;
        var q2_off: usize = 0;
        var q8_off: usize = 0;

        for (0..T.QK_K / 128) |_| {
            inline for (0..4) |shift_idx| {
                const shift: u3 = shift_idx * 2;
                // First 16 elements
                const d0: i32 = @intCast(sc[is] & 0xF);
                is += 1;
                const q2v0: @Vector(16, u8) = q2[q2_off..][0..16].*;
                const bits0: @Vector(16, i8) = @bitCast((q2v0 >> @as(@Vector(16, u3), @splat(shift))) & @as(@Vector(16, u8), @splat(3)));
                const q8v0: @Vector(16, i32) = @intCast(@as(@Vector(16, i8), q8[q8_off..][0..16].*));
                const av0: @Vector(16, i32) = @intCast(bits0);
                isum += d0 * @reduce(.Add, av0 * q8v0);

                // Second 16 elements
                const d1: i32 = @intCast(sc[is] & 0xF);
                is += 1;
                const q2v1: @Vector(16, u8) = q2[q2_off + 16 ..][0..16].*;
                const bits1: @Vector(16, i8) = @bitCast((q2v1 >> @as(@Vector(16, u3), @splat(shift))) & @as(@Vector(16, u8), @splat(3)));
                const q8v1: @Vector(16, i32) = @intCast(@as(@Vector(16, i8), q8[q8_off + 16 ..][0..16].*));
                const av1: @Vector(16, i32) = @intCast(bits1);
                isum += d1 * @reduce(.Add, av1 * q8v1);

                q8_off += 32;
            }
            q2_off += 32;
        }

        sumf += dall * @as(f32, @floatFromInt(isum)) - dmin * @as(f32, @floatFromInt(summs));
    }

    s.* = sumf;
}

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

        var a_off: usize = 0;
        var q8_off: usize = 0;
        var sumk: i32 = 0;
        for (0..T.QK_K / 32) |is| {
            const scale: i32 = @intCast(scales[is]);
            const av: @Vector(32, i32) = @intCast(@as(@Vector(32, i8), aux8[a_off..][0..32].*));
            const qv: @Vector(32, i32) = @intCast(@as(@Vector(32, i8), q8[q8_off..][0..32].*));
            sumk += scale * @reduce(.Add, av * qv);
            a_off += 32;
            q8_off += 32;
        }

        sumf += T.f16f32(xp[i].d) * yp[i].d * @as(f32, @floatFromInt(sumk));
        sumf -= T.f16f32(xp[i].dmin) * yp[i].d * @as(f32, @floatFromInt(sumi));
    }

    s.* = sumf;
}

// We want to compute: s = dot(X, Y)
// where X is a float vector of length n (the weights)
// and   Y is a float vector of length n (the activations)
//
// But X and Y are not stored as floats. They are quantized:
//   X is stored as n/32 blocks of block_q4_0
//   Y is stored as n/32 blocks of block_q8_0
//
// Each block represents 32 consecutive floats from the original vector.
// To recover the original float value:
//   X[i*32 + j] = (raw_x[j] - 8) * d_x     where raw_x[j] is a 4-bit int (0..15)
//   Y[i*32 + j] = raw_y[j] * d_y            where raw_y[j] is a signed 8-bit int
//
// d_x and d_y are per-block scale factors that control the precision
// of that group of 32 values.
//
// Since d_x and d_y are constant within a group of 32, we can compute:
//   dot(X, Y) = sum over all groups of:
//       d_x * d_y * sum_j( (raw_x[j] - 8) * raw_y[j] )
//
// That inner sum is pure integer arithmetic.
//
// One more trick: instead of subtracting 8 from each raw_x[j], we expand:
//   sum( (raw_x[j] - 8) * raw_y[j] ) = sum( raw_x[j] * raw_y[j] ) - 8 * sum( raw_y[j] )
// This avoids 32 subtractions per group.

pub export fn zig_vec_dot_q4_0_q8_0(
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

    const nb: usize = @intCast(@divExact(n, T.QK8_0));
    const xp: [*]const T.block_q4_0 = @ptrCast(@alignCast(vx)); // quantized X
    const yp: [*]const T.block_q8_0 = @ptrCast(@alignCast(vy)); // quantized Y

    var sum: f32 = 0;

    const cat_idx = comptime blk: {
        var idx: [32]i32 = undefined;
        for (0..16) |j| idx[j] = @intCast(j);
        for (0..16) |j| idx[16 + j] = ~@as(i32, @intCast(j));
        break :blk idx;
    };

    for (0..nb) |i| {
        const raw: @Vector(16, u8) = xp[i].qs;
        // Pre-bias nibbles to signed: 0..15 → -8..+7 via wrapping add of 0xF8
        // This lets us use signed×signed dot (sdot) and skip the bias subtraction entirely.
        const bias: @Vector(16, u8) = @splat(0xF8);
        const lo: @Vector(16, i8) = @bitCast((raw & @as(@Vector(16, u8), @splat(0x0F))) +% bias);
        const hi: @Vector(16, i8) = @bitCast((raw >> @as(@Vector(16, u8), @splat(4))) +% bias);

        // Concat lo/hi into 32-wide signed vector, widen to i32 for dot product.
        const q4: @Vector(32, i32) = @intCast(@shuffle(i8, lo, hi, cat_idx));
        const q8: @Vector(32, i32) = @intCast(@as(@Vector(32, i8), yp[i].qs));

        // Single signed dot product — no bias subtraction needed.
        const int_dot = @reduce(.Add, q4 * q8);

        sum += T.f16f32(xp[i].d) * T.f16f32(yp[i].d) * @as(f32, @floatFromInt(int_dot));
    }

    s.* = sum;
}

pub export fn zig_vec_dot_q8_0_q8_0(
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

    const nb: usize = @intCast(@divExact(n, T.QK8_0));
    const xp: [*]const T.block_q8_0 = @ptrCast(@alignCast(vx));
    const yp: [*]const T.block_q8_0 = @ptrCast(@alignCast(vy));

    var sum: f32 = 0;

    for (0..nb) |i| {
        const qx: @Vector(32, i32) = @intCast(@as(@Vector(32, i8), xp[i].qs));
        const qy: @Vector(32, i32) = @intCast(@as(@Vector(32, i8), yp[i].qs));
        const int_dot = @reduce(.Add, qx * qy);
        sum += T.f16f32(xp[i].d) * T.f16f32(yp[i].d) * @as(f32, @floatFromInt(int_dot));
    }

    s.* = sum;
}

// ── Test infrastructure ──

const VecDotFn = *const fn (c_int, *f32, usize, *const anyopaque, usize, *const anyopaque, usize, c_int) callconv(.c) void;

fn benchmark(comptime f: VecDotFn, n: c_int, vx: *const anyopaque, vy: *const anyopaque) u64 {
    var r: f32 = 0;
    for (0..50) |_| f(n, &r, 0, vx, 0, vy, 0, 1);
    var t: [500]u64 = undefined;
    for (0..500) |i| {
        var timer = std.time.Timer.start() catch unreachable;
        f(n, &r, 0, vx, 0, vy, 0, 1);
        t[i] = timer.read();
    }
    sink += r;
    std.mem.sortUnstable(u64, &t, {}, std.sort.asc(u64));
    return t[250];
}

var sink: f32 = 0;

const FmtResult = struct { v: f64, u: []const u8 };

fn fmtNs(ns: u64) FmtResult {
    return if (ns >= 1_000_000)
        .{ .v = @as(f64, @floatFromInt(ns)) / 1e6, .u = "ms" }
    else if (ns >= 1_000)
        .{ .v = @as(f64, @floatFromInt(ns)) / 1e3, .u = "us" }
    else
        .{ .v = @as(f64, @floatFromInt(ns)), .u = "ns" };
}

fn checkParity(comptime zig_fn: VecDotFn, comptime ref_fn: VecDotFn, n: c_int, vx: *const anyopaque, vy: *const anyopaque) !void {
    var zig_r: f32 = 0;
    var ref_r: f32 = 0;
    zig_fn(n, &zig_r, 0, vx, 0, vy, 0, 1);
    ref_fn(n, &ref_r, 0, vx, 0, vy, 0, 1);
    if (@abs(zig_r - ref_r) / @max(@abs(ref_r), 1e-6) > 1e-3)
        return error.TestExpectedEqual;
}

fn printBench(label: []const u8, n: usize, comptime fns: anytype, vx: *const anyopaque, vy: *const anyopaque) void {
    const ci: c_int = @intCast(n);
    var vals: [fns.len]FmtResult = undefined;
    inline for (fns, 0..) |entry, i| vals[i] = fmtNs(benchmark(entry[1], ci, vx, vy));
    std.debug.print("n={d:<8}", .{n});
    inline for (0..fns.len) |i| std.debug.print(" {d:>7.2} {s}", .{ vals[i].v, vals[i].u });
    std.debug.print(" {s}\n", .{label});
}

extern fn ggml_cpu_init() void;

// ── Q2_K tests ──

extern fn ggml_vec_dot_q2_K_q8_K(c_int, *f32, usize, *const anyopaque, usize, *const anyopaque, usize, c_int) void;
extern fn ggml_vec_dot_q2_K_q8_K_generic(c_int, *f32, usize, *const anyopaque, usize, *const anyopaque, usize, c_int) void;

test "q2_K vec_dot" {
    ggml_cpu_init();
    var prng = std.Random.DefaultPrng.init(0xFACE_FEED);
    const rand = prng.random();
    const a = std.heap.page_allocator;

    for ([_]usize{ 1, 4, 16, 64, 256 }) |nb| {
        const x = try a.alloc(T.block_q2_K, nb);
        const y = try a.alloc(T.block_q8_K, nb);
        defer a.free(x);
        defer a.free(y);
        fillQ2K(x, rand);
        fillQ8K(y, rand);
        try checkParity(zig_vec_dot_q2_K_q8_K, ggml_vec_dot_q2_K_q8_K_generic, @intCast(nb * T.QK_K), @ptrCast(x.ptr), @ptrCast(y.ptr));
    }

    std.debug.print("\nq2_K:        {s:>10} {s:>10} {s:>10}\n", .{ "zig", "generic", "neon" });
    for ([_]usize{ 16, 64, 256, 1024 }) |nb| {
        const x = try a.alloc(T.block_q2_K, nb);
        const y = try a.alloc(T.block_q8_K, nb);
        defer a.free(x);
        defer a.free(y);
        fillQ2K(x, rand);
        fillQ8K(y, rand);
        printBench("", nb * T.QK_K, .{
            .{ "zig", zig_vec_dot_q2_K_q8_K },
            .{ "generic", ggml_vec_dot_q2_K_q8_K_generic },
            .{ "neon", ggml_vec_dot_q2_K_q8_K },
        }, @ptrCast(x.ptr), @ptrCast(y.ptr));
    }
}

fn fillQ2K(xb: []T.block_q2_K, rand: std.Random) void {
    for (xb) |*blk| {
        blk.d = @bitCast(@as(f16, @floatCast((rand.float(f32) - 0.5) * 2.0)));
        blk.dmin = @bitCast(@as(f16, @floatCast(rand.float(f32) * 0.5)));
        for (&blk.scales) |*v| v.* = rand.int(u8);
        for (&blk.qs) |*v| v.* = rand.int(u8);
    }
}

// ── Q4_K tests ──

extern fn ggml_vec_dot_q4_K_q8_K(c_int, *f32, usize, *const anyopaque, usize, *const anyopaque, usize, c_int) void;
extern fn ggml_vec_dot_q4_K_q8_K_generic(c_int, *f32, usize, *const anyopaque, usize, *const anyopaque, usize, c_int) void;

test "q4_K vec_dot" {
    ggml_cpu_init();
    var prng = std.Random.DefaultPrng.init(0xDEAD_BEEF);
    const rand = prng.random();
    const a = std.heap.page_allocator;

    for ([_]usize{ 1, 4, 16, 64, 256 }) |nb| {
        const x = try a.alloc(T.block_q4_K, nb);
        const y = try a.alloc(T.block_q8_K, nb);
        defer a.free(x);
        defer a.free(y);
        fillQ4K(x, rand);
        fillQ8K(y, rand);
        try checkParity(zig_vec_dot_q4_K_q8_K, ggml_vec_dot_q4_K_q8_K_generic, @intCast(nb * T.QK_K), @ptrCast(x.ptr), @ptrCast(y.ptr));
    }

    std.debug.print("\nq4_K:        {s:>10} {s:>10} {s:>10}\n", .{ "zig", "generic", "neon" });
    for ([_]usize{ 16, 64, 256, 1024 }) |nb| {
        const x = try a.alloc(T.block_q4_K, nb);
        const y = try a.alloc(T.block_q8_K, nb);
        defer a.free(x);
        defer a.free(y);
        fillQ4K(x, rand);
        fillQ8K(y, rand);
        printBench("", nb * T.QK_K, .{
            .{ "zig", zig_vec_dot_q4_K_q8_K },
            .{ "generic", ggml_vec_dot_q4_K_q8_K_generic },
            .{ "neon", ggml_vec_dot_q4_K_q8_K },
        }, @ptrCast(x.ptr), @ptrCast(y.ptr));
    }
}

fn fillQ4K(xb: []T.block_q4_K, rand: std.Random) void {
    for (xb) |*blk| {
        blk.d = @bitCast(@as(f16, @floatCast((rand.float(f32) - 0.5) * 2.0)));
        blk.dmin = @bitCast(@as(f16, @floatCast(rand.float(f32) * 0.5)));
        for (&blk.scales) |*v| v.* = rand.intRangeAtMost(u8, 0, 63);
        for (&blk.qs) |*v| v.* = rand.int(u8);
    }
}

fn fillQ8K(yb: []T.block_q8_K, rand: std.Random) void {
    for (yb) |*blk| {
        blk.d = (rand.float(f32) - 0.5) * 2.0;
        for (&blk.qs) |*v| v.* = @as(i8, @intCast(@as(i32, rand.intRangeAtMost(u8, 0, 255)) - 128));
        for (&blk.bsums) |*v| v.* = @as(i16, @intCast(@as(i32, rand.intRangeAtMost(u16, 0, 65535)) - 32768));
    }
}

// ── Q4_0 tests ──

extern fn ggml_vec_dot_q4_0_q8_0(c_int, *f32, usize, *const anyopaque, usize, *const anyopaque, usize, c_int) void;
extern fn ggml_vec_dot_q4_0_q8_0_generic(c_int, *f32, usize, *const anyopaque, usize, *const anyopaque, usize, c_int) void;

test "q4_0 vec_dot" {
    ggml_cpu_init();
    var prng = std.Random.DefaultPrng.init(0xCAFE_BABE);
    const rand = prng.random();
    const a = std.heap.page_allocator;

    for ([_]usize{ 1, 4, 16, 64, 256 }) |nb| {
        const x = try a.alloc(T.block_q4_0, nb);
        const y = try a.alloc(T.block_q8_0, nb);
        defer a.free(x);
        defer a.free(y);
        fillQ4_0(x, rand);
        fillQ8_0(y, rand);
        try checkParity(zig_vec_dot_q4_0_q8_0, ggml_vec_dot_q4_0_q8_0_generic, @intCast(nb * T.QK4_0), @ptrCast(x.ptr), @ptrCast(y.ptr));
    }

    std.debug.print("\nq4_0:        {s:>10} {s:>10} {s:>10}\n", .{ "zig", "generic", "neon" });
    for ([_]usize{ 16, 64, 256, 1024 }) |nb| {
        const x = try a.alloc(T.block_q4_0, nb);
        const y = try a.alloc(T.block_q8_0, nb);
        defer a.free(x);
        defer a.free(y);
        fillQ4_0(x, rand);
        fillQ8_0(y, rand);
        printBench("", nb * T.QK4_0, .{
            .{ "zig", zig_vec_dot_q4_0_q8_0 },
            .{ "generic", ggml_vec_dot_q4_0_q8_0_generic },
            .{ "neon", ggml_vec_dot_q4_0_q8_0 },
        }, @ptrCast(x.ptr), @ptrCast(y.ptr));
    }
}

fn fillQ4_0(xb: []T.block_q4_0, rand: std.Random) void {
    for (xb) |*blk| {
        blk.d = @bitCast(@as(f16, @floatCast((rand.float(f32) - 0.5) * 2.0)));
        for (&blk.qs) |*v| v.* = rand.int(u8);
    }
}

fn fillQ8_0(yb: []T.block_q8_0, rand: std.Random) void {
    for (yb) |*blk| {
        blk.d = @bitCast(@as(f16, @floatCast((rand.float(f32) - 0.5) * 2.0)));
        for (&blk.qs) |*v| v.* = @as(i8, @intCast(@as(i32, rand.intRangeAtMost(u8, 0, 255)) - 128));
    }
}

// ── Q8_0 tests ──

extern fn ggml_vec_dot_q8_0_q8_0(c_int, *f32, usize, *const anyopaque, usize, *const anyopaque, usize, c_int) void;
extern fn ggml_vec_dot_q8_0_q8_0_generic(c_int, *f32, usize, *const anyopaque, usize, *const anyopaque, usize, c_int) void;

test "q8_0 vec_dot" {
    ggml_cpu_init();
    var prng = std.Random.DefaultPrng.init(0xBEEF_F00D);
    const rand = prng.random();
    const a = std.heap.page_allocator;

    for ([_]usize{ 1, 4, 16, 64, 256 }) |nb| {
        const x = try a.alloc(T.block_q8_0, nb);
        const y = try a.alloc(T.block_q8_0, nb);
        defer a.free(x);
        defer a.free(y);
        fillQ8_0(x, rand);
        fillQ8_0(y, rand);
        try checkParity(zig_vec_dot_q8_0_q8_0, ggml_vec_dot_q8_0_q8_0_generic, @intCast(nb * T.QK8_0), @ptrCast(x.ptr), @ptrCast(y.ptr));
    }

    std.debug.print("\nq8_0:        {s:>10} {s:>10} {s:>10}\n", .{ "zig", "generic", "neon" });
    for ([_]usize{ 16, 64, 256, 1024 }) |nb| {
        const x = try a.alloc(T.block_q8_0, nb);
        const y = try a.alloc(T.block_q8_0, nb);
        defer a.free(x);
        defer a.free(y);
        fillQ8_0(x, rand);
        fillQ8_0(y, rand);
        printBench("", nb * T.QK8_0, .{
            .{ "zig", zig_vec_dot_q8_0_q8_0 },
            .{ "generic", ggml_vec_dot_q8_0_q8_0_generic },
            .{ "neon", ggml_vec_dot_q8_0_q8_0 },
        }, @ptrCast(x.ptr), @ptrCast(y.ptr));
    }
}
