const std = @import("std");

pub fn dot(lhs: []const f32, rhs: []const f32) !f32 {
    if (lhs.len != rhs.len) return error.SizeMismatch;

    var sum: f32 = 0.0;
    for (lhs, rhs) |a, b| {
        sum += a * b;
    }
    return sum;
}

pub fn axpyInPlace(output: []f32, alpha: f32, input: []const f32) !void {
    if (output.len != input.len) return error.SizeMismatch;

    for (output, input) |*out, value| {
        out.* += alpha * value;
    }
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

    const inv_rms = computeInvRms(input, eps);
    applyRmsNormScaled(output, input, weight, inv_rms);
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

fn computeInvRms(input: []const f32, eps: f32) f32 {
    var sum_vec0: @Vector(16, f32) = @splat(0.0);
    var sum_vec1: @Vector(16, f32) = @splat(0.0);
    var index: usize = 0;
    while (index + 32 <= input.len) : (index += 32) {
        const v0: @Vector(16, f32) = input[index..][0..16].*;
        const v1: @Vector(16, f32) = input[index + 16 ..][0..16].*;
        sum_vec0 += v0 * v0;
        sum_vec1 += v1 * v1;
    }

    var mean_square = @reduce(.Add, sum_vec0 + sum_vec1);

    while (index + 16 <= input.len) : (index += 16) {
        const v: @Vector(16, f32) = input[index..][0..16].*;
        mean_square += @reduce(.Add, v * v);
    }
    while (index < input.len) : (index += 1) {
        mean_square += input[index] * input[index];
    }

    mean_square /= @as(f32, @floatFromInt(input.len));
    return 1.0 / @sqrt(mean_square + eps);
}

fn applyRmsNormScaled(output: []f32, input: []const f32, weight: []const f32, inv_rms: f32) void {
    const inv_rms_vec: @Vector(16, f32) = @splat(inv_rms);

    var index: usize = 0;
    while (index + 16 <= output.len) : (index += 16) {
        const in_vec: @Vector(16, f32) = input[index..][0..16].*;
        const weight_vec: @Vector(16, f32) = weight[index..][0..16].*;
        output[index..][0..16].* = in_vec * weight_vec * inv_rms_vec;
    }
    while (index < output.len) : (index += 1) {
        output[index] = input[index] * inv_rms * weight[index];
    }
}

fn rmsNormScalarReference(
    output: []f32,
    input: []const f32,
    weight: []const f32,
    eps: f32,
) void {
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

test "axpyInPlace accumulates scaled input into output" {
    const testing = std.testing;

    var output = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const input = [_]f32{ 0.5, -1.0, 2.0, 1.5 };

    try axpyInPlace(&output, 2.0, &input);

    try testing.expectApproxEqAbs(@as(f32, 2.0), output[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.0), output[1], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 7.0), output[2], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 7.0), output[3], 1e-6);
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

test "wide rmsNorm matches scalar reference" {
    const testing = std.testing;

    inline for (.{ 128, 1024, 3072 }) |len| {
        const input = try testing.allocator.alloc(f32, len);
        defer testing.allocator.free(input);
        const weight = try testing.allocator.alloc(f32, len);
        defer testing.allocator.free(weight);
        const output = try testing.allocator.alloc(f32, len);
        defer testing.allocator.free(output);
        const expected = try testing.allocator.alloc(f32, len);
        defer testing.allocator.free(expected);

        for (input, 0..) |*value, idx| {
            value.* = @as(f32, @floatFromInt(@as(i32, @intCast((idx * 17 + 3) % 41)) - 20)) / 8.0;
            weight[idx] = @as(f32, @floatFromInt(@as(i32, @intCast((idx * 11 + 7) % 37)) - 18)) / 9.0;
        }

        rmsNormScalarReference(expected, input, weight, 1e-5);
        try rmsNorm(output, input, weight, 1e-5);

        for (expected, output) |want, got| {
            try testing.expectApproxEqAbs(want, got, 1e-5);
        }
    }
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
