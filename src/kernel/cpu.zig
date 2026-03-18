const std = @import("std");

pub fn dot(lhs: []const f32, rhs: []const f32) !f32 {
    if (lhs.len != rhs.len) return error.SizeMismatch;

    var sum: f32 = 0.0;
    for (lhs, rhs) |a, b| {
        sum += a * b;
    }
    return sum;
}

pub fn matmulVec(
    output: []f32,
    weights_row_major: []const f32,
    input: []const f32,
    rows: usize,
    cols: usize,
) !void {
    if (output.len != rows) return error.SizeMismatch;
    if (input.len != cols) return error.SizeMismatch;
    if (weights_row_major.len != rows * cols) return error.SizeMismatch;

    for (0..rows) |row| {
        const start = row * cols;
        output[row] = try dot(weights_row_major[start .. start + cols], input);
    }
}

pub fn rmsNorm(
    output: []f32,
    input: []const f32,
    weight: []const f32,
    eps: f32,
) !void {
    if (output.len != input.len or input.len != weight.len) return error.SizeMismatch;

    var mean_square: f32 = 0.0;
    for (input) |value| {
        mean_square += value * value;
    }
    mean_square /= @as(f32, @floatFromInt(input.len));

    const inv_rms = 1.0 / @sqrt(mean_square + eps);
    for (output, input, weight) |*out, x, w| {
        out.* = x * inv_rms * w;
    }
}

pub fn rmsNormRepeated(
    output: []f32,
    input: []const f32,
    repeat_count: usize,
    slice_len: usize,
    weight: []const f32,
    eps: f32,
) !void {
    if (weight.len != slice_len) return error.SizeMismatch;
    if (input.len != repeat_count * slice_len) return error.SizeMismatch;
    if (output.len != input.len) return error.SizeMismatch;

    for (0..repeat_count) |idx| {
        const start = idx * slice_len;
        try rmsNorm(
            output[start .. start + slice_len],
            input[start .. start + slice_len],
            weight,
            eps,
        );
    }
}

pub fn silu(x: f32) f32 {
    return x / (1.0 + std.math.exp(-x));
}

pub fn swiglu(output: []f32, gate: []const f32, up: []const f32) !void {
    if (output.len != gate.len or gate.len != up.len) return error.SizeMismatch;

    for (output, gate, up) |*out, gate_value, up_value| {
        out.* = silu(gate_value) * up_value;
    }
}

test "dot computes expected result" {
    const testing = std.testing;

    const lhs = [_]f32{ 1.0, 2.0, 3.0 };
    const rhs = [_]f32{ 4.0, 5.0, 6.0 };
    try testing.expectEqual(@as(f32, 32.0), try dot(&lhs, &rhs));
}

test "matmulVec multiplies row-major matrix by vector" {
    const testing = std.testing;

    const weights = [_]f32{
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    };
    const input = [_]f32{ 1.0, 0.5, -1.0 };
    var output = [_]f32{ 0.0, 0.0 };

    try matmulVec(&output, &weights, &input, 2, 3);

    try testing.expectApproxEqAbs(@as(f32, -1.0), output[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.5), output[1], 1e-6);
}

test "rmsNorm matches manual calculation" {
    const testing = std.testing;

    const input = [_]f32{ 3.0, 4.0 };
    const weight = [_]f32{ 1.0, 2.0 };
    var output = [_]f32{ 0.0, 0.0 };

    try rmsNorm(&output, &input, &weight, 0.0);

    try testing.expectApproxEqAbs(@as(f32, 0.84852814), output[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 2.2627418), output[1], 1e-6);
}

test "swiglu applies silu gate then multiplies up branch" {
    const testing = std.testing;

    const gate = [_]f32{ 0.0, 1.0, -1.0 };
    const up = [_]f32{ 1.0, 2.0, 3.0 };
    var output = [_]f32{ 0.0, 0.0, 0.0 };

    try swiglu(&output, &gate, &up);

    try testing.expectApproxEqAbs(@as(f32, 0.0), output[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 1.4621172), output[1], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, -0.8068243), output[2], 1e-6);
}

test "rmsNormRepeated applies same norm weight to multiple slices" {
    const testing = std.testing;

    const input = [_]f32{ 3.0, 4.0, 6.0, 8.0 };
    const weight = [_]f32{ 1.0, 2.0 };
    var output = [_]f32{ 0.0, 0.0, 0.0, 0.0 };

    try rmsNormRepeated(&output, &input, 2, 2, &weight, 0.0);

    try testing.expectApproxEqAbs(@as(f32, 0.84852814), output[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 2.2627418), output[1], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.84852814), output[2], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 2.2627418), output[3], 1e-6);
}
