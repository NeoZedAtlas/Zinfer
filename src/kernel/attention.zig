const std = @import("std");
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
