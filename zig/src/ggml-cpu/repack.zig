const std = @import("std");
const T = @import("ggml-common");

const NCOLS = 8;
const BLOCKLEN = 8;

pub export fn zig_gemm_q4_K_8x8_q8_K(
    n: c_int,
    s: [*]f32,
    bs: usize,
    vx: *const anyopaque,
    vy: *const anyopaque,
    nr: c_int,
    nc: c_int,
) void {
    const nb: usize = @intCast(@divExact(n, T.QK_K));
    const nrows: usize = @intCast(nr);
    const ncols: usize = @intCast(nc);

    const b_base: [*]const T.block_q4_Kx8 = @ptrCast(@alignCast(vx));
    const a_base: [*]const T.block_q8_Kx4 = @ptrCast(@alignCast(vy));

    var y: usize = 0;
    while (y < nrows / 4) : (y += 1) {
        const a_row = a_base + y * nb;

        var x: usize = 0;
        while (x < ncols / NCOLS) : (x += 1) {
            const b_col = b_base + x * nb;

            var sumf: [4][8]f32 = .{.{0} ** 8} ** 4;
            var sum_minf: [4][8]f32 = .{.{0} ** 8} ** 4;

            for (0..nb) |l| {
                var utmp: [32]u32 = undefined;
                inline for (0..8) |sb| {
                    const off = sb * 4;
                    @memcpy(std.mem.asBytes(&utmp)[off * 4 ..][0..12], b_col[l].scales[sb * 12 ..][0..12]);
                    utmp[off + 3] = ((utmp[off + 2] >> 4) & T.kmask2) | (((utmp[off + 1] >> 6) & T.kmask3) << 4);
                    const uaux_0 = utmp[off + 1] & T.kmask1;
                    utmp[off + 1] = (utmp[off + 2] & T.kmask2) | (((utmp[off + 0] >> 6) & T.kmask3) << 4);
                    utmp[off + 2] = uaux_0;
                    utmp[off + 0] &= T.kmask1;
                }

                for (0..T.QK_K / (2 * BLOCKLEN)) |k| {
                    const sc0: [*]const u8 = @ptrCast(&utmp[(k / 4) * 8]);
                    const sc1: [*]const u8 = @ptrCast(&utmp[(k / 4) * 8 + 4]);

                    for (0..4) |m| {
                        const q8_base = (k >> 2) * 256 + (k % 4) * 4 * BLOCKLEN + m * BLOCKLEN;

                        // Load q8 lo and hi as separate 8-wide vectors
                        const q8_lo: @Vector(BLOCKLEN, i32) = @intCast(@as(@Vector(BLOCKLEN, i8), a_row[l].qs[q8_base..][0..BLOCKLEN].*));
                        const q8_hi: @Vector(BLOCKLEN, i32) = @intCast(@as(@Vector(BLOCKLEN, i8), a_row[l].qs[q8_base + 128 ..][0..BLOCKLEN].*));

                        inline for (0..NCOLS) |j| {
                            const qs_base = k * NCOLS * BLOCKLEN + j * BLOCKLEN;
                            const raw: @Vector(BLOCKLEN, u8) = b_col[l].qs[qs_base..][0..BLOCKLEN].*;
                            const v0: @Vector(BLOCKLEN, i32) = @intCast(@as(@Vector(BLOCKLEN, i8), @bitCast(raw & @as(@Vector(BLOCKLEN, u8), @splat(0x0F)))));
                            const v1: @Vector(BLOCKLEN, i32) = @intCast(@as(@Vector(BLOCKLEN, i8), @bitCast(raw >> @as(@Vector(BLOCKLEN, u8), @splat(4)))));

                            const dot_lo = @reduce(.Add, v0 * q8_lo);
                            const dot_hi = @reduce(.Add, v1 * q8_hi);
                            const sumi = dot_lo * @as(i32, sc0[j]) + dot_hi * @as(i32, sc1[j]);

                            sumf[m][j] += @as(f32, @floatFromInt(sumi)) * T.f16f32(b_col[l].d[j]) * a_row[l].d[m];
                        }
                    }
                }

                for (0..8) |sb| {
                    const mins: [*]const u8 = @as([*]const u8, @ptrCast(&utmp)) + 8 + sb * 16;
                    for (0..4) |m| {
                        const bsums_base = sb * 8 + m * 4 -| (sb % 2) * 6;
                        const bsum: i32 = @as(i32, a_row[l].bsums[bsums_base]) + @as(i32, a_row[l].bsums[bsums_base + 1]);
                        for (0..NCOLS) |j| {
                            sum_minf[m][j] += @as(f32, @floatFromInt(@as(i32, mins[j]) * bsum)) * T.f16f32(b_col[l].dmin[j]) * a_row[l].d[m];
                        }
                    }
                }
            }

            for (0..4) |m| {
                for (0..NCOLS) |j| {
                    s[(y * 4 + m) * bs + x * NCOLS + j] = sumf[m][j] - sum_minf[m][j];
                }
            }
        }
    }
}

//  Tests

extern fn ggml_cpu_init() void;
extern fn ggml_gemm_q4_K_8x8_q8_K_generic(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) void;
extern fn ggml_gemm_q4_K_8x8_q8_K(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) void;

fn fmtNs(ns: u64) struct { v: f64, u: []const u8 } {
    return if (ns >= 1_000_000) .{ .v = @as(f64, @floatFromInt(ns)) / 1e6, .u = "ms" } else if (ns >= 1_000) .{ .v = @as(f64, @floatFromInt(ns)) / 1e3, .u = "us" } else .{ .v = @as(f64, @floatFromInt(ns)), .u = "ns" };
}

var bench_sink: f32 = 0;

fn fillRandBytes(buf: []u8, rand: std.Random) void {
    for (buf) |*b| b.* = rand.int(u8);
}

test "q4_K gemm: parity + bench" {
    ggml_cpu_init();
    var prng = std.Random.DefaultPrng.init(0xFACE_CAFE);
    const rand = prng.random();
    const alloc = std.heap.page_allocator;

    // -- parity --
    const nb: usize = 4;
    const nr: c_int = 4;
    const nc: c_int = 8;
    const n: c_int = @intCast(nb * T.QK_K);

    const xb = try alloc.alloc(T.block_q4_Kx8, nb);
    const yb = try alloc.alloc(T.block_q8_Kx4, nb);
    defer alloc.free(xb);
    defer alloc.free(yb);

    fillRandBytes(std.mem.sliceAsBytes(xb), rand);
    fillRandBytes(std.mem.sliceAsBytes(yb), rand);
    for (xb) |*blk| inline for (0..8) |j| {
        blk.d[j] = @bitCast(@as(f16, @floatCast((rand.float(f32) - 0.5) * 2.0)));
        blk.dmin[j] = @bitCast(@as(f16, @floatCast(rand.float(f32) * 0.5)));
    };
    for (yb) |*blk| inline for (0..4) |j| {
        blk.d[j] = (rand.float(f32) - 0.5) * 2.0;
    };

    var zig_out: [32]f32 = .{0} ** 32;
    var ref_out: [32]f32 = .{0} ** 32;
    zig_gemm_q4_K_8x8_q8_K(n, &zig_out, @intCast(nc), @ptrCast(xb.ptr), @ptrCast(yb.ptr), nr, nc);
    ggml_gemm_q4_K_8x8_q8_K_generic(n, &ref_out, @intCast(nc), @ptrCast(xb.ptr), @ptrCast(yb.ptr), nr, nc);

    for (0..32) |i| {
        if (@abs(zig_out[i] - ref_out[i]) / @max(@abs(ref_out[i]), 1e-6) > 1e-2) {
            std.debug.print("GEMM MISMATCH i={}: zig={d:.6} ref={d:.6}\n", .{ i, zig_out[i], ref_out[i] });
            return error.TestExpectedEqual;
        }
    }

    // -- bench --
    const bnb: usize = 16;
    const bxb = try alloc.alloc(T.block_q4_Kx8, bnb);
    const byb = try alloc.alloc(T.block_q8_Kx4, bnb);
    const out = try alloc.alloc(f32, 32);
    defer alloc.free(bxb);
    defer alloc.free(byb);
    defer alloc.free(out);

    fillRandBytes(std.mem.sliceAsBytes(bxb), rand);
    fillRandBytes(std.mem.sliceAsBytes(byb), rand);
    for (bxb) |*blk| inline for (0..8) |j| {
        blk.d[j] = @bitCast(@as(f16, @floatCast((rand.float(f32) - 0.5) * 2.0)));
        blk.dmin[j] = @bitCast(@as(f16, @floatCast(rand.float(f32) * 0.5)));
    };
    for (byb) |*blk| inline for (0..4) |j| {
        blk.d[j] = (rand.float(f32) - 0.5) * 2.0;
    };

    const bn: c_int = @intCast(bnb * T.QK_K);
    std.debug.print("\ngemm:\n{s:<12} {s:>10}\n", .{ "", "median" });
    inline for (.{ .{ "zig", zig_gemm_q4_K_8x8_q8_K }, .{ "generic", ggml_gemm_q4_K_8x8_q8_K_generic }, .{ "neon", ggml_gemm_q4_K_8x8_q8_K } }) |entry| {
        for (0..50) |_| entry[1](bn, out.ptr, 8, @ptrCast(bxb.ptr), @ptrCast(byb.ptr), 4, 8);
        var t: [500]u64 = undefined;
        for (0..500) |i| {
            var timer = std.time.Timer.start() catch unreachable;
            entry[1](bn, out.ptr, 8, @ptrCast(bxb.ptr), @ptrCast(byb.ptr), 4, 8);
            t[i] = timer.read();
        }
        bench_sink += out[0];
        std.mem.sortUnstable(u64, &t, {}, std.sort.asc(u64));
        const med = fmtNs(t[250]);
        std.debug.print("  {s:<10} {d:>7.2} {s}\n", .{ entry[0], med.v, med.u });
    }
}
