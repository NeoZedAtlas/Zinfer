const std = @import("std");
const bfloat16 = @import("../tensor/bfloat16.zig");
const cpu = @import("cpu.zig");

pub fn applyRoPEToHeadInPlace(
    head: []f32,
    position: usize,
    rope_theta: f32,
) !void {
    if (head.len % 2 != 0) return error.InvalidHeadDim;

    const dim_f = @as(f32, @floatFromInt(head.len));
    const pos_f = @as(f32, @floatFromInt(position));
    const half = head.len / 2;

    for (0..half) |i| {
        const exponent = @as(f32, @floatFromInt(i * 2)) / dim_f;
        const inv_freq = 1.0 / std.math.pow(f32, rope_theta, exponent);
        const angle = pos_f * inv_freq;
        const cos_angle = std.math.cos(angle);
        const sin_angle = std.math.sin(angle);

        const x0 = head[i];
        const x1 = head[i + half];
        head[i] = x0 * cos_angle - x1 * sin_angle;
        head[i + half] = x1 * cos_angle + x0 * sin_angle;
    }
}

pub fn applyRoPEToHeadsInPlace(
    heads: []f32,
    num_heads: usize,
    head_dim: usize,
    position: usize,
    rope_theta: f32,
) !void {
    if (heads.len != num_heads * head_dim) return error.SizeMismatch;

    for (0..num_heads) |head_idx| {
        const start = head_idx * head_dim;
        try applyRoPEToHeadInPlace(heads[start .. start + head_dim], position, rope_theta);
    }
}

pub fn softmaxInPlace(values: []f32) !void {
    if (values.len == 0) return error.EmptyInput;

    var max_value = values[0];
    for (values[1..]) |value| {
        max_value = @max(max_value, value);
    }

    var sum: f32 = 0.0;
    for (values, 0..) |value, idx| {
        const exp_value = std.math.exp(value - max_value);
        values[idx] = exp_value;
        sum += exp_value;
    }

    if (sum == 0.0) return error.InvalidSoftmaxSum;
    const inv_sum = 1.0 / sum;
    for (values) |*value| {
        value.* *= inv_sum;
    }
}

pub fn scaledDotProductAttentionSingleQuery(
    output: []f32,
    query: []const f32,
    key_cache: []const f32,
    value_cache: []const f32,
    seq_len: usize,
    num_query_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    scores_scratch: []f32,
) !void {
    if (num_query_heads == 0 or num_key_value_heads == 0 or head_dim == 0) {
        return error.InvalidDimensions;
    }
    if (num_query_heads % num_key_value_heads != 0) return error.InvalidGrouping;
    if (seq_len == 0) return error.InvalidSequenceLength;
    if (output.len != num_query_heads * head_dim) return error.SizeMismatch;
    if (query.len != num_query_heads * head_dim) return error.SizeMismatch;
    if (key_cache.len != seq_len * num_key_value_heads * head_dim) return error.SizeMismatch;
    if (value_cache.len != seq_len * num_key_value_heads * head_dim) return error.SizeMismatch;
    if (scores_scratch.len < seq_len) return error.InsufficientScratchSpace;

    const group_size = num_query_heads / num_key_value_heads;
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

    @memset(output, 0.0);

    for (0..num_query_heads) |q_head_idx| {
        const q_start = q_head_idx * head_dim;
        const q_slice = query[q_start .. q_start + head_dim];
        const kv_head_idx = q_head_idx / group_size;
        const scores = scores_scratch[0..seq_len];

        for (0..seq_len) |pos| {
            const cache_start = (pos * num_key_value_heads + kv_head_idx) * head_dim;
            const k_slice = key_cache[cache_start .. cache_start + head_dim];
            scores[pos] = (try cpu.dot(q_slice, k_slice)) * scale;
        }

        try softmaxInPlace(scores);

        const out_slice = output[q_start .. q_start + head_dim];
        for (0..seq_len) |pos| {
            const weight = scores[pos];
            const cache_start = (pos * num_key_value_heads + kv_head_idx) * head_dim;
            const v_slice = value_cache[cache_start .. cache_start + head_dim];
            try cpu.axpyInPlace(out_slice, weight, v_slice);
        }
    }
}

pub fn scaledDotProductAttentionSingleQueryBf16Cache(
    output: []f32,
    query: []const f32,
    key_cache: []const u16,
    value_cache: []const u16,
    seq_len: usize,
    num_query_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    scores_scratch: []f32,
) !void {
    if (num_query_heads == 0 or num_key_value_heads == 0 or head_dim == 0) {
        return error.InvalidDimensions;
    }
    if (num_query_heads % num_key_value_heads != 0) return error.InvalidGrouping;
    if (seq_len == 0) return error.InvalidSequenceLength;
    if (output.len != num_query_heads * head_dim) return error.SizeMismatch;
    if (query.len != num_query_heads * head_dim) return error.SizeMismatch;
    if (key_cache.len != seq_len * num_key_value_heads * head_dim) return error.SizeMismatch;
    if (value_cache.len != seq_len * num_key_value_heads * head_dim) return error.SizeMismatch;
    if (scores_scratch.len < seq_len) return error.InsufficientScratchSpace;

    const group_size = num_query_heads / num_key_value_heads;
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

    @memset(output, 0.0);

    for (0..num_query_heads) |q_head_idx| {
        const q_start = q_head_idx * head_dim;
        const q_slice = query[q_start .. q_start + head_dim];
        const kv_head_idx = q_head_idx / group_size;
        const scores = scores_scratch[0..seq_len];

        for (0..seq_len) |pos| {
            const cache_start = (pos * num_key_value_heads + kv_head_idx) * head_dim;
            var sum: f32 = 0.0;
            for (0..head_dim) |dim_idx| {
                sum += q_slice[dim_idx] * bfloat16.toF32(key_cache[cache_start + dim_idx]);
            }
            scores[pos] = sum * scale;
        }

        try softmaxInPlace(scores);

        const out_slice = output[q_start .. q_start + head_dim];
        for (0..seq_len) |pos| {
            const weight = scores[pos];
            const cache_start = (pos * num_key_value_heads + kv_head_idx) * head_dim;
            for (0..head_dim) |dim_idx| {
                out_slice[dim_idx] += weight * bfloat16.toF32(value_cache[cache_start + dim_idx]);
            }
        }
    }
}

pub fn scaledDotProductAttentionSingleQueryQ8Cache(
    output: []f32,
    query: []const f32,
    key_cache: []const i8,
    key_scales: []const u16,
    value_cache: []const i8,
    value_scales: []const u16,
    seq_len: usize,
    num_query_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    scores_scratch: []f32,
) !void {
    if (num_query_heads == 0 or num_key_value_heads == 0 or head_dim == 0) {
        return error.InvalidDimensions;
    }
    if (num_query_heads % num_key_value_heads != 0) return error.InvalidGrouping;
    if (seq_len == 0) return error.InvalidSequenceLength;
    if (output.len != num_query_heads * head_dim) return error.SizeMismatch;
    if (query.len != num_query_heads * head_dim) return error.SizeMismatch;
    if (key_cache.len != seq_len * num_key_value_heads * head_dim) return error.SizeMismatch;
    if (value_cache.len != seq_len * num_key_value_heads * head_dim) return error.SizeMismatch;
    if (key_scales.len != seq_len * num_key_value_heads) return error.SizeMismatch;
    if (value_scales.len != seq_len * num_key_value_heads) return error.SizeMismatch;
    if (scores_scratch.len < seq_len) return error.InsufficientScratchSpace;

    const group_size = num_query_heads / num_key_value_heads;
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

    @memset(output, 0.0);

    for (0..num_query_heads) |q_head_idx| {
        const q_start = q_head_idx * head_dim;
        const q_slice = query[q_start .. q_start + head_dim];
        const kv_head_idx = q_head_idx / group_size;
        const scores = scores_scratch[0..seq_len];

        for (0..seq_len) |pos| {
            const scale_index = pos * num_key_value_heads + kv_head_idx;
            const cache_start = scale_index * head_dim;
            const key_scale = bfloat16.toF32(key_scales[scale_index]);
            scores[pos] = (dotQ8Slice(q_slice, key_cache[cache_start .. cache_start + head_dim]) * key_scale) * scale;
        }

        try softmaxInPlace(scores);

        const out_slice = output[q_start .. q_start + head_dim];
        for (0..seq_len) |pos| {
            const weight = scores[pos];
            const scale_index = pos * num_key_value_heads + kv_head_idx;
            const cache_start = scale_index * head_dim;
            const value_scale = bfloat16.toF32(value_scales[scale_index]);
            axpyQ8SliceInPlace(out_slice, weight * value_scale, value_cache[cache_start .. cache_start + head_dim]);
        }
    }
}

fn dotQ8Slice(lhs: []const f32, rhs_q8: []const i8) f32 {
    std.debug.assert(lhs.len == rhs_q8.len);

    var sum: f32 = 0.0;
    var index: usize = 0;
    while (index + 16 <= lhs.len) : (index += 16) {
        const lhs_vec: @Vector(16, f32) = lhs[index..][0..16].*;
        var rhs_arr: [16]f32 = undefined;
        inline for (0..16) |lane| {
            rhs_arr[lane] = @floatFromInt(rhs_q8[index + lane]);
        }
        const rhs_vec: @Vector(16, f32) = rhs_arr;
        sum += @reduce(.Add, lhs_vec * rhs_vec);
    }
    while (index < lhs.len) : (index += 1) {
        sum += lhs[index] * @as(f32, @floatFromInt(rhs_q8[index]));
    }
    return sum;
}

fn axpyQ8SliceInPlace(output: []f32, alpha: f32, input_q8: []const i8) void {
    std.debug.assert(output.len == input_q8.len);

    const alpha_vec: @Vector(16, f32) = @splat(alpha);
    var index: usize = 0;
    while (index + 16 <= output.len) : (index += 16) {
        const out_vec: @Vector(16, f32) = output[index..][0..16].*;
        var in_arr: [16]f32 = undefined;
        inline for (0..16) |lane| {
            in_arr[lane] = @floatFromInt(input_q8[index + lane]);
        }
        const in_vec: @Vector(16, f32) = in_arr;
        output[index..][0..16].* = out_vec + alpha_vec * in_vec;
    }
    while (index < output.len) : (index += 1) {
        output[index] += alpha * @as(f32, @floatFromInt(input_q8[index]));
    }
}

test "rope leaves head unchanged at position zero" {
    const testing = std.testing;

    var head = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    try applyRoPEToHeadInPlace(&head, 0, 10000.0);

    try testing.expectEqual(@as(f32, 1.0), head[0]);
    try testing.expectEqual(@as(f32, 2.0), head[1]);
    try testing.expectEqual(@as(f32, 3.0), head[2]);
    try testing.expectEqual(@as(f32, 4.0), head[3]);
}

test "rope rotates first and second half pairs at nonzero position" {
    const testing = std.testing;

    var head = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    try applyRoPEToHeadInPlace(&head, 1, 10000.0);

    const cos0 = std.math.cos(@as(f32, 1.0));
    const sin0 = std.math.sin(@as(f32, 1.0));
    const inv_freq1 = 1.0 / std.math.pow(f32, @as(f32, 10000.0), 0.5);
    const cos1 = std.math.cos(inv_freq1);
    const sin1 = std.math.sin(inv_freq1);

    try testing.expectApproxEqAbs(@as(f32, 1.0) * cos0 - @as(f32, 3.0) * sin0, head[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 2.0) * cos1 - @as(f32, 4.0) * sin1, head[1], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 3.0) * cos0 + @as(f32, 1.0) * sin0, head[2], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 4.0) * cos1 + @as(f32, 2.0) * sin1, head[3], 1e-6);
}

test "softmax normalizes values" {
    const testing = std.testing;

    var values = [_]f32{ 1.0, 2.0, 3.0 };
    try softmaxInPlace(&values);

    const sum = values[0] + values[1] + values[2];
    try testing.expectApproxEqAbs(@as(f32, 1.0), sum, 1e-6);
    try testing.expect(values[2] > values[1]);
    try testing.expect(values[1] > values[0]);
}

test "single-query attention attends over one kv head" {
    const testing = std.testing;

    const query = [_]f32{ 1.0, 0.0 };
    const key_cache = [_]f32{
        1.0, 0.0,
        0.0, 1.0,
    };
    const value_cache = [_]f32{
        10.0, 1.0,
        20.0, 2.0,
    };
    var output = [_]f32{ 0.0, 0.0 };
    var scores = [_]f32{ 0.0, 0.0 };

    try scaledDotProductAttentionSingleQuery(
        &output,
        &query,
        &key_cache,
        &value_cache,
        2,
        1,
        1,
        2,
        &scores,
    );

    try testing.expect(output[0] > 10.0);
    try testing.expect(output[0] < 20.0);
    try testing.expect(output[1] > 1.0);
    try testing.expect(output[1] < 2.0);
}

test "single-query attention supports grouped query attention" {
    const testing = std.testing;

    const query = [_]f32{
        1.0, 0.0,
        0.0, 1.0,
    };
    const key_cache = [_]f32{
        1.0, 0.0,
    };
    const value_cache = [_]f32{
        5.0, 6.0,
    };
    var output = [_]f32{ 0.0, 0.0, 0.0, 0.0 };
    var scores = [_]f32{0.0};

    try scaledDotProductAttentionSingleQuery(
        &output,
        &query,
        &key_cache,
        &value_cache,
        1,
        2,
        1,
        2,
        &scores,
    );

    try testing.expectApproxEqAbs(@as(f32, 5.0), output[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 6.0), output[1], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 5.0), output[2], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 6.0), output[3], 1e-6);
}

test "single-query attention maps grouped query heads to different kv heads" {
    const testing = std.testing;

    const query = [_]f32{
        1.0, 0.0,
        1.0, 0.0,
        0.0, 1.0,
        0.0, 1.0,
    };
    const key_cache = [_]f32{
        1.0, 0.0,
        0.0, 1.0,
    };
    const value_cache = [_]f32{
        11.0, 12.0,
        21.0, 22.0,
    };
    var output = [_]f32{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    var scores = [_]f32{0.0};

    try scaledDotProductAttentionSingleQuery(
        &output,
        &query,
        &key_cache,
        &value_cache,
        1,
        4,
        2,
        2,
        &scores,
    );

    try testing.expectApproxEqAbs(@as(f32, 11.0), output[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 12.0), output[1], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 11.0), output[2], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 12.0), output[3], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 21.0), output[4], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 22.0), output[5], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 21.0), output[6], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 22.0), output[7], 1e-6);
}

test "single-query attention supports bf16 kv cache" {
    const testing = std.testing;

    const query = [_]f32{ 1.0, 0.0 };
    const key_cache = [_]u16{
        bfloat16.fromF32(1.0), bfloat16.fromF32(0.0),
        bfloat16.fromF32(0.0), bfloat16.fromF32(1.0),
    };
    const value_cache = [_]u16{
        bfloat16.fromF32(10.0), bfloat16.fromF32(1.0),
        bfloat16.fromF32(20.0), bfloat16.fromF32(2.0),
    };
    var output = [_]f32{ 0.0, 0.0 };
    var scores = [_]f32{ 0.0, 0.0 };

    try scaledDotProductAttentionSingleQueryBf16Cache(
        &output,
        &query,
        &key_cache,
        &value_cache,
        2,
        1,
        1,
        2,
        &scores,
    );

    try testing.expect(output[0] > 10.0);
    try testing.expect(output[0] < 20.0);
    try testing.expect(output[1] > 1.0);
    try testing.expect(output[1] < 2.0);
}

test "single-query attention supports q8 kv cache" {
    const testing = std.testing;

    const query = [_]f32{ 1.0, 0.0 };
    const key_cache = [_]i8{ 127, 0, 0, 127 };
    const value_cache = [_]i8{ 64, 6, 127, 13 };
    const key_scales = [_]u16{ bfloat16.fromF32(1.0 / 127.0), bfloat16.fromF32(1.0 / 127.0) };
    const value_scales = [_]u16{ bfloat16.fromF32(20.0 / 127.0), bfloat16.fromF32(20.0 / 127.0) };
    var output = [_]f32{ 0.0, 0.0 };
    var scores = [_]f32{ 0.0, 0.0 };

    try scaledDotProductAttentionSingleQueryQ8Cache(
        &output,
        &query,
        &key_cache,
        &key_scales,
        &value_cache,
        &value_scales,
        2,
        1,
        1,
        2,
        &scores,
    );

    try testing.expect(output[0] > 9.0);
    try testing.expect(output[0] < 20.5);
    try testing.expect(output[1] > 0.5);
    try testing.expect(output[1] < 2.5);
}
