const std = @import("std");
const bfloat16 = @import("../../../tensor/formats/bfloat16.zig");
const types = @import("types.zig");

pub fn quantizeQ8Slice(output: []i8, input: []const f32) u16 {
    var max_abs: f32 = 0.0;
    for (input) |value| {
        max_abs = @max(max_abs, @abs(value));
    }
    const scale: f32 = if (max_abs == 0.0) 1.0 else max_abs / 127.0;
    const inv_scale = 1.0 / scale;
    for (input, 0..) |value, idx| {
        const quantized = std.math.clamp(@as(i32, @intFromFloat(@round(value * inv_scale))), -127, 127);
        output[idx] = @intCast(quantized);
    }
    return bfloat16.fromF32(scale);
}

pub fn scaleGroupsPerToken(num_key_value_heads: usize, head_dim: usize) usize {
    return num_key_value_heads * (std.math.divCeil(usize, head_dim, types.q8_group_size) catch unreachable);
}
