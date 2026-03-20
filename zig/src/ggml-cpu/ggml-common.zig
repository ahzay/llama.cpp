pub const QK_K = 256;
pub const K_SCALE_SIZE = 12;

pub const block_q4_K = extern struct {
    d: u16,
    dmin: u16,
    scales: [K_SCALE_SIZE]u8,
    qs: [QK_K / 2]u8,
};
comptime {
    if (@sizeOf(block_q4_K) != 144) @compileError("block_q4_K layout mismatch");
}

pub const block_q8_K = extern struct {
    d: f32,
    qs: [QK_K]i8,
    bsums: [QK_K / 16]i16,
};
comptime {
    if (@sizeOf(block_q8_K) != 292) @compileError("block_q8_K layout mismatch");
}

pub const block_q4_Kx8 = extern struct {
    d: [8]u16,
    dmin: [8]u16,
    scales: [96]u8,
    qs: [1024]u8,
};
comptime {
    if (@sizeOf(block_q4_Kx8) != 1152) @compileError("block_q4_Kx8 layout mismatch");
}

pub const block_q8_Kx4 = extern struct {
    d: [4]f32,
    qs: [QK_K * 4]i8,
    bsums: [QK_K / 4]i16,
};
comptime {
    if (@sizeOf(block_q8_Kx4) != 1168) @compileError("block_q8_Kx4 layout mismatch");
}

pub inline fn f16f32(h: u16) f32 {
    return @floatCast(@as(f16, @bitCast(h)));
}

pub const kmask1: u32 = 0x3f3f3f3f;
pub const kmask2: u32 = 0x0f0f0f0f;
pub const kmask3: u32 = 0x03030303;
