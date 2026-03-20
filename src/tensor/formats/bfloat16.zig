const std = @import("std");

pub fn toF32(bits: u16) f32 {
    const raw: u32 = @as(u32, bits) << 16;
    return @bitCast(raw);
}

pub fn fromF32(value: f32) u16 {
    const raw: u32 = @bitCast(value);
    const lsb = (raw >> 16) & 1;
    const rounding_bias: u32 = 0x7fff + lsb;
    return @truncate((raw + rounding_bias) >> 16);
}

test "bfloat16 roundtrip keeps simple values" {
    const testing = std.testing;

    try testing.expectEqual(@as(f32, 0.0), toF32(fromF32(0.0)));
    try testing.expectEqual(@as(f32, 1.0), toF32(fromF32(1.0)));
    try testing.expectEqual(@as(f32, -2.5), toF32(fromF32(-2.5)));
}

test "bfloat16 decodes known bit patterns" {
    const testing = std.testing;

    try testing.expectEqual(@as(f32, 1.0), toF32(0x3f80));
    try testing.expectEqual(@as(f32, -1.0), toF32(0xbf80));
    try testing.expectEqual(@as(f32, 2.0), toF32(0x4000));
}
