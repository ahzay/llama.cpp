pub const QK4_0 = 32;
pub const QK8_0 = 32;

pub const block_q4_0 = extern struct {
    d: u16,
    qs: [QK4_0 / 2]u8,
};
comptime {
    if (@sizeOf(block_q4_0) != 18) @compileError("block_q4_0 layout mismatch");
}

pub const block_q8_0 = extern struct {
    d: u16,
    qs: [QK8_0]i8,
};
comptime {
    if (@sizeOf(block_q8_0) != 34) @compileError("block_q8_0 layout mismatch");
}

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


pub inline fn f16f32(h: u16) f32 {
    return @floatCast(@as(f16, @bitCast(h)));
}

pub const kmask1: u32 = 0x3f3f3f3f;
pub const kmask2: u32 = 0x0f0f0f0f;
pub const kmask3: u32 = 0x03030303;
