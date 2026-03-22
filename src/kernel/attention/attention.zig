const std = @import("std");
const bfloat16 = @import("../../tensor/formats/bfloat16.zig");
const kernel_registry = @import("../registry.zig");
const cpu = @import("../core/cpu.zig");

pub const q8_cache_group_size: usize = 16;
const handwritten_q8_head_dim: usize = 128;
const handwritten_q8_scale_groups: usize = handwritten_q8_head_dim / q8_cache_group_size;

pub const RoPETable = struct {
    allocator: std.mem.Allocator,
    max_seq_len: usize,
    head_dim: usize,
    cos: []f32,
    sin: []f32,

    pub fn init(
        allocator: std.mem.Allocator,
        max_seq_len: usize,
        head_dim: usize,
        rope_theta: f32,
    ) !RoPETable {
        if (head_dim == 0 or head_dim % 2 != 0) return error.InvalidHeadDim;
        const half = head_dim / 2;
        const total = try std.math.mul(usize, max_seq_len, half);

        const cos = try allocator.alloc(f32, total);
        errdefer allocator.free(cos);
        const sin = try allocator.alloc(f32, total);
        errdefer allocator.free(sin);

        const dim_f = @as(f32, @floatFromInt(head_dim));
        for (0..max_seq_len) |position| {
            const pos_f = @as(f32, @floatFromInt(position));
            const base = position * half;
            for (0..half) |i| {
                const exponent = @as(f32, @floatFromInt(i * 2)) / dim_f;
                const inv_freq = 1.0 / std.math.pow(f32, rope_theta, exponent);
                const angle = pos_f * inv_freq;
                cos[base + i] = std.math.cos(angle);
                sin[base + i] = std.math.sin(angle);
            }
        }

        return .{
            .allocator = allocator,
            .max_seq_len = max_seq_len,
            .head_dim = head_dim,
            .cos = cos,
            .sin = sin,
        };
    }

    pub fn deinit(self: *RoPETable) void {
        self.allocator.free(self.cos);
        self.allocator.free(self.sin);
    }

    pub fn cosForPosition(self: *const RoPETable, position: usize) []const f32 {
        std.debug.assert(position < self.max_seq_len);
        const half = self.head_dim / 2;
        const start = position * half;
        return self.cos[start .. start + half];
    }

    pub fn sinForPosition(self: *const RoPETable, position: usize) []const f32 {
        std.debug.assert(position < self.max_seq_len);
        const half = self.head_dim / 2;
        const start = position * half;
        return self.sin[start .. start + half];
    }
};

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

pub fn applyRoPEToHeadWithTableInPlace(
    head: []f32,
    cos_table: []const f32,
    sin_table: []const f32,
) !void {
    if (head.len % 2 != 0) return error.InvalidHeadDim;
    const half = head.len / 2;
    if (cos_table.len != half or sin_table.len != half) return error.SizeMismatch;

    for (0..half) |i| {
        const x0 = head[i];
        const x1 = head[i + half];
        const cos_angle = cos_table[i];
        const sin_angle = sin_table[i];
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

pub fn applyRoPEToHeadsWithTableInPlace(
    heads: []f32,
    num_heads: usize,
    head_dim: usize,
    table: *const RoPETable,
    position: usize,
) !void {
    if (heads.len != num_heads * head_dim) return error.SizeMismatch;
    if (table.head_dim != head_dim) return error.SizeMismatch;
    if (position >= table.max_seq_len) return error.PositionOutOfBounds;

    const cos = table.cosForPosition(position);
    const sin = table.sinForPosition(position);
    for (0..num_heads) |head_idx| {
        const start = head_idx * head_dim;
        try applyRoPEToHeadWithTableInPlace(heads[start .. start + head_dim], cos, sin);
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
    const scale_groups_per_head = std.math.divCeil(usize, head_dim, q8_cache_group_size) catch return error.InvalidDimensions;
    if (key_scales.len != seq_len * num_key_value_heads * scale_groups_per_head) return error.SizeMismatch;
    if (value_scales.len != seq_len * num_key_value_heads * scale_groups_per_head) return error.SizeMismatch;
    if (scores_scratch.len < seq_len) return error.InsufficientScratchSpace;

    const entry = kernel_registry.resolve(.{ .attention_q8_decode = .{
        .head_dim = head_dim,
        .layout = .token_major,
    } });
    if (entry.shape == .qwen3_head_dim_128 and scale_groups_per_head == handwritten_q8_scale_groups) {
        return scaledDotProductAttentionSingleQueryQ8Cache128(
            output,
            query,
            key_cache,
            key_scales,
            value_cache,
            value_scales,
            seq_len,
            num_query_heads,
            num_key_value_heads,
            scores_scratch,
        );
    }

    return scaledDotProductAttentionSingleQueryQ8CacheGeneric(
        output,
        query,
        key_cache,
        key_scales,
        value_cache,
        value_scales,
        seq_len,
        num_query_heads,
        num_key_value_heads,
        head_dim,
        scale_groups_per_head,
        scores_scratch,
    );
}

pub fn scaledDotProductAttentionSingleQueryQ8CacheHeadMajor(
    output: []f32,
    query: []const f32,
    key_cache: []const i8,
    key_scales: []const u16,
    value_cache: []const i8,
    value_scales: []const u16,
    data_head_stride: usize,
    scale_head_stride: usize,
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
    const scale_groups_per_head = std.math.divCeil(usize, head_dim, q8_cache_group_size) catch return error.InvalidDimensions;
    if (data_head_stride < seq_len * head_dim) return error.SizeMismatch;
    if (scale_head_stride < seq_len * scale_groups_per_head) return error.SizeMismatch;
    if (key_cache.len < num_key_value_heads * data_head_stride) return error.SizeMismatch;
    if (value_cache.len < num_key_value_heads * data_head_stride) return error.SizeMismatch;
    if (key_scales.len < num_key_value_heads * scale_head_stride) return error.SizeMismatch;
    if (value_scales.len < num_key_value_heads * scale_head_stride) return error.SizeMismatch;
    if (scores_scratch.len < seq_len) return error.InsufficientScratchSpace;

    const entry = kernel_registry.resolve(.{ .attention_q8_decode = .{
        .head_dim = head_dim,
        .layout = .head_major,
    } });
    if (entry.shape == .qwen3_head_dim_128 and scale_groups_per_head == handwritten_q8_scale_groups) {
        return scaledDotProductAttentionSingleQueryQ8CacheHeadMajor128(
            output,
            query,
            key_cache,
            key_scales,
            value_cache,
            value_scales,
            data_head_stride,
            scale_head_stride,
            seq_len,
            num_query_heads,
            num_key_value_heads,
            scores_scratch,
        );
    }

    return scaledDotProductAttentionSingleQueryQ8CacheHeadMajorGeneric(
        output,
        query,
        key_cache,
        key_scales,
        value_cache,
        value_scales,
        data_head_stride,
        scale_head_stride,
        seq_len,
        num_query_heads,
        num_key_value_heads,
        head_dim,
        scale_groups_per_head,
        scores_scratch,
    );
}

fn scaledDotProductAttentionSingleQueryQ8CacheGeneric(
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
    scale_groups_per_head: usize,
    scores_scratch: []f32,
) !void {
    const group_size = num_query_heads / num_key_value_heads;
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

    @memset(output, 0.0);

    for (0..num_query_heads) |q_head_idx| {
        const q_start = q_head_idx * head_dim;
        const q_slice = query[q_start .. q_start + head_dim];
        const kv_head_idx = q_head_idx / group_size;
        const scores = scores_scratch[0..seq_len];

        for (0..seq_len) |pos| {
            const cache_head_index = pos * num_key_value_heads + kv_head_idx;
            const cache_start = cache_head_index * head_dim;
            const scale_start = cache_head_index * scale_groups_per_head;
            scores[pos] = dotQ8GroupedSlice(
                q_slice,
                key_cache[cache_start .. cache_start + head_dim],
                key_scales[scale_start .. scale_start + scale_groups_per_head],
            ) * scale;
        }

        try softmaxInPlace(scores);

        const out_slice = output[q_start .. q_start + head_dim];
        for (0..seq_len) |pos| {
            const weight = scores[pos];
            const cache_head_index = pos * num_key_value_heads + kv_head_idx;
            const cache_start = cache_head_index * head_dim;
            const scale_start = cache_head_index * scale_groups_per_head;
            axpyQ8GroupedSliceInPlace(
                out_slice,
                weight,
                value_cache[cache_start .. cache_start + head_dim],
                value_scales[scale_start .. scale_start + scale_groups_per_head],
            );
        }
    }
}

fn scaledDotProductAttentionSingleQueryQ8CacheHeadMajorGeneric(
    output: []f32,
    query: []const f32,
    key_cache: []const i8,
    key_scales: []const u16,
    value_cache: []const i8,
    value_scales: []const u16,
    data_head_stride: usize,
    scale_head_stride: usize,
    seq_len: usize,
    num_query_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    scale_groups_per_head: usize,
    scores_scratch: []f32,
) !void {
    const group_size = num_query_heads / num_key_value_heads;
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

    @memset(output, 0.0);

    for (0..num_query_heads) |q_head_idx| {
        const q_start = q_head_idx * head_dim;
        const q_slice = query[q_start .. q_start + head_dim];
        const kv_head_idx = q_head_idx / group_size;
        const head_data_start = kv_head_idx * data_head_stride;
        const head_scale_start = kv_head_idx * scale_head_stride;
        const scores = scores_scratch[0..seq_len];

        for (0..seq_len) |pos| {
            const cache_start = head_data_start + pos * head_dim;
            const scale_start = head_scale_start + pos * scale_groups_per_head;
            scores[pos] = dotQ8GroupedSlice(
                q_slice,
                key_cache[cache_start .. cache_start + head_dim],
                key_scales[scale_start .. scale_start + scale_groups_per_head],
            ) * scale;
        }

        try softmaxInPlace(scores);

        const out_slice = output[q_start .. q_start + head_dim];
        for (0..seq_len) |pos| {
            const weight = scores[pos];
            const cache_start = head_data_start + pos * head_dim;
            const scale_start = head_scale_start + pos * scale_groups_per_head;
            axpyQ8GroupedSliceInPlace(
                out_slice,
                weight,
                value_cache[cache_start .. cache_start + head_dim],
                value_scales[scale_start .. scale_start + scale_groups_per_head],
            );
        }
    }
}

fn scaledDotProductAttentionSingleQueryQ8Cache128(
    output: []f32,
    query: []const f32,
    key_cache: []const i8,
    key_scales: []const u16,
    value_cache: []const i8,
    value_scales: []const u16,
    seq_len: usize,
    num_query_heads: usize,
    num_key_value_heads: usize,
    scores_scratch: []f32,
) !void {
    const head_dim = handwritten_q8_head_dim;
    const scale_groups_per_head = handwritten_q8_scale_groups;
    const group_size = num_query_heads / num_key_value_heads;
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

    @memset(output, 0.0);

    for (0..num_query_heads) |q_head_idx| {
        const q_start = q_head_idx * head_dim;
        const q_slice = query[q_start .. q_start + head_dim];
        const kv_head_idx = q_head_idx / group_size;
        const scores = scores_scratch[0..seq_len];

        for (0..seq_len) |pos| {
            const cache_head_index = pos * num_key_value_heads + kv_head_idx;
            const cache_start = cache_head_index * head_dim;
            const scale_start = cache_head_index * scale_groups_per_head;
            scores[pos] = dotQ8GroupedSlice128Exact(
                q_slice,
                key_cache[cache_start .. cache_start + head_dim],
                key_scales[scale_start .. scale_start + scale_groups_per_head],
            ) * scale;
        }

        try softmaxInPlace(scores);

        const out_slice = output[q_start .. q_start + head_dim];
        accumulateQ8ValueHead128(
            out_slice,
            scores,
            value_cache,
            value_scales,
            num_key_value_heads,
            kv_head_idx,
        );
    }
}

fn scaledDotProductAttentionSingleQueryQ8CacheHeadMajor128(
    output: []f32,
    query: []const f32,
    key_cache: []const i8,
    key_scales: []const u16,
    value_cache: []const i8,
    value_scales: []const u16,
    data_head_stride: usize,
    scale_head_stride: usize,
    seq_len: usize,
    num_query_heads: usize,
    num_key_value_heads: usize,
    scores_scratch: []f32,
) !void {
    const head_dim = handwritten_q8_head_dim;
    const group_size = num_query_heads / num_key_value_heads;
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

    @memset(output, 0.0);

    for (0..num_query_heads) |q_head_idx| {
        const q_start = q_head_idx * head_dim;
        const q_slice = query[q_start .. q_start + head_dim];
        const kv_head_idx = q_head_idx / group_size;
        const head_data_start = kv_head_idx * data_head_stride;
        const head_scale_start = kv_head_idx * scale_head_stride;
        const scores = scores_scratch[0..seq_len];

        for (0..seq_len) |pos| {
            const cache_start = head_data_start + pos * head_dim;
            const scale_start = head_scale_start + pos * handwritten_q8_scale_groups;
            scores[pos] = dotQ8GroupedSlice128Exact(
                q_slice,
                key_cache[cache_start .. cache_start + head_dim],
                key_scales[scale_start .. scale_start + handwritten_q8_scale_groups],
            ) * scale;
        }

        try softmaxInPlace(scores);

        const out_slice = output[q_start .. q_start + head_dim];
        accumulateQ8ValueHead128HeadMajor(
            out_slice,
            scores,
            value_cache[head_data_start .. head_data_start + seq_len * head_dim],
            value_scales[head_scale_start .. head_scale_start + seq_len * handwritten_q8_scale_groups],
        );
    }
}

pub fn dotQ8GroupedSlice(lhs: []const f32, rhs_q8: []const i8, scales: []const u16) f32 {
    std.debug.assert(lhs.len == rhs_q8.len);
    if (lhs.len == scales.len * q8_cache_group_size) {
        if (lhs.len == handwritten_q8_head_dim and scales.len == handwritten_q8_scale_groups) {
            return dotQ8GroupedSlice128Exact(lhs, rhs_q8, scales);
        }
        return dotQ8GroupedSliceExact(lhs, rhs_q8, scales);
    }

    var sum: f32 = 0.0;
    var index: usize = 0;
    for (scales) |scale_bits| {
        if (index >= lhs.len) break;
        const end = @min(lhs.len, index + q8_cache_group_size);
        const scale = bfloat16.toF32(scale_bits);
        if (end - index == 16) {
            const lhs_vec: @Vector(16, f32) = lhs[index..][0..16].*;
            const rhs_i8: @Vector(16, i8) = rhs_q8[index..][0..16].*;
            const rhs_vec: @Vector(16, f32) = @floatFromInt(rhs_i8);
            sum += @reduce(.Add, lhs_vec * rhs_vec) * scale;
        } else {
            var local: f32 = 0.0;
            var local_index = index;
            while (local_index < end) : (local_index += 1) {
                local += lhs[local_index] * @as(f32, @floatFromInt(rhs_q8[local_index]));
            }
            sum += local * scale;
        }
        index = end;
    }
    return sum;
}

pub fn axpyQ8GroupedSliceInPlace(output: []f32, alpha: f32, input_q8: []const i8, scales: []const u16) void {
    std.debug.assert(output.len == input_q8.len);
    if (output.len == scales.len * q8_cache_group_size) {
        axpyQ8GroupedSliceExactInPlace(output, alpha, input_q8, scales);
        return;
    }

    var index: usize = 0;
    for (scales) |scale_bits| {
        if (index >= output.len) break;
        const end = @min(output.len, index + q8_cache_group_size);
        const scaled_alpha = alpha * bfloat16.toF32(scale_bits);
        if (end - index == 16) {
            const alpha_vec: @Vector(16, f32) = @splat(scaled_alpha);
            const out_vec: @Vector(16, f32) = output[index..][0..16].*;
            const in_i8: @Vector(16, i8) = input_q8[index..][0..16].*;
            const in_vec: @Vector(16, f32) = @floatFromInt(in_i8);
            output[index..][0..16].* = out_vec + alpha_vec * in_vec;
        } else {
            var local_index = index;
            while (local_index < end) : (local_index += 1) {
                output[local_index] += scaled_alpha * @as(f32, @floatFromInt(input_q8[local_index]));
            }
        }
        index = end;
    }
}

fn dotQ8GroupedSliceExact(lhs: []const f32, rhs_q8: []const i8, scales: []const u16) f32 {
    var sum: f32 = 0.0;
    var index: usize = 0;
    for (scales) |scale_bits| {
        const scale = bfloat16.toF32(scale_bits);
        const lhs_vec: @Vector(16, f32) = lhs[index..][0..16].*;
        const rhs_i8: @Vector(16, i8) = rhs_q8[index..][0..16].*;
        const rhs_vec: @Vector(16, f32) = @floatFromInt(rhs_i8);
        sum += @reduce(.Add, lhs_vec * rhs_vec) * scale;
        index += 16;
    }
    return sum;
}

fn dotQ8GroupedSlice128Exact(lhs: []const f32, rhs_q8: []const i8, scales: []const u16) f32 {
    std.debug.assert(lhs.len == handwritten_q8_head_dim);
    std.debug.assert(rhs_q8.len == handwritten_q8_head_dim);
    std.debug.assert(scales.len == handwritten_q8_scale_groups);

    var acc0: @Vector(16, f32) = @splat(0.0);
    var acc1: @Vector(16, f32) = @splat(0.0);
    var acc2: @Vector(16, f32) = @splat(0.0);
    var acc3: @Vector(16, f32) = @splat(0.0);
    var acc4: @Vector(16, f32) = @splat(0.0);
    var acc5: @Vector(16, f32) = @splat(0.0);
    var acc6: @Vector(16, f32) = @splat(0.0);
    var acc7: @Vector(16, f32) = @splat(0.0);

    acc0 += lhs[0..16].* * scaledQ8Vector16(rhs_q8, 0, bfloat16.toF32(scales[0]));
    acc1 += lhs[16..32].* * scaledQ8Vector16(rhs_q8, 16, bfloat16.toF32(scales[1]));
    acc2 += lhs[32..48].* * scaledQ8Vector16(rhs_q8, 32, bfloat16.toF32(scales[2]));
    acc3 += lhs[48..64].* * scaledQ8Vector16(rhs_q8, 48, bfloat16.toF32(scales[3]));
    acc4 += lhs[64..80].* * scaledQ8Vector16(rhs_q8, 64, bfloat16.toF32(scales[4]));
    acc5 += lhs[80..96].* * scaledQ8Vector16(rhs_q8, 80, bfloat16.toF32(scales[5]));
    acc6 += lhs[96..112].* * scaledQ8Vector16(rhs_q8, 96, bfloat16.toF32(scales[6]));
    acc7 += lhs[112..128].* * scaledQ8Vector16(rhs_q8, 112, bfloat16.toF32(scales[7]));

    return @reduce(.Add, acc0 + acc1 + acc2 + acc3 + acc4 + acc5 + acc6 + acc7);
}

fn axpyQ8GroupedSliceExactInPlace(output: []f32, alpha: f32, input_q8: []const i8, scales: []const u16) void {
    var index: usize = 0;
    for (scales) |scale_bits| {
        const alpha_vec: @Vector(16, f32) = @splat(alpha * bfloat16.toF32(scale_bits));
        const out_vec: @Vector(16, f32) = output[index..][0..16].*;
        const in_i8: @Vector(16, i8) = input_q8[index..][0..16].*;
        const in_vec: @Vector(16, f32) = @floatFromInt(in_i8);
        output[index..][0..16].* = out_vec + alpha_vec * in_vec;
        index += 16;
    }
}

fn accumulateQ8ValueHead128(
    output: []f32,
    scores: []const f32,
    value_cache: []const i8,
    value_scales: []const u16,
    num_key_value_heads: usize,
    kv_head_idx: usize,
) void {
    std.debug.assert(output.len == handwritten_q8_head_dim);

    var acc0: @Vector(16, f32) = @splat(0.0);
    var acc1: @Vector(16, f32) = @splat(0.0);
    var acc2: @Vector(16, f32) = @splat(0.0);
    var acc3: @Vector(16, f32) = @splat(0.0);
    var acc4: @Vector(16, f32) = @splat(0.0);
    var acc5: @Vector(16, f32) = @splat(0.0);
    var acc6: @Vector(16, f32) = @splat(0.0);
    var acc7: @Vector(16, f32) = @splat(0.0);

    for (scores, 0..) |weight, pos| {
        const cache_head_index = pos * num_key_value_heads + kv_head_idx;
        const cache_start = cache_head_index * handwritten_q8_head_dim;
        const scale_start = cache_head_index * handwritten_q8_scale_groups;

        acc0 += scaledQ8Vector16(value_cache, cache_start + 0, weight * bfloat16.toF32(value_scales[scale_start + 0]));
        acc1 += scaledQ8Vector16(value_cache, cache_start + 16, weight * bfloat16.toF32(value_scales[scale_start + 1]));
        acc2 += scaledQ8Vector16(value_cache, cache_start + 32, weight * bfloat16.toF32(value_scales[scale_start + 2]));
        acc3 += scaledQ8Vector16(value_cache, cache_start + 48, weight * bfloat16.toF32(value_scales[scale_start + 3]));
        acc4 += scaledQ8Vector16(value_cache, cache_start + 64, weight * bfloat16.toF32(value_scales[scale_start + 4]));
        acc5 += scaledQ8Vector16(value_cache, cache_start + 80, weight * bfloat16.toF32(value_scales[scale_start + 5]));
        acc6 += scaledQ8Vector16(value_cache, cache_start + 96, weight * bfloat16.toF32(value_scales[scale_start + 6]));
        acc7 += scaledQ8Vector16(value_cache, cache_start + 112, weight * bfloat16.toF32(value_scales[scale_start + 7]));
    }

    output[0..16].* = acc0;
    output[16..32].* = acc1;
    output[32..48].* = acc2;
    output[48..64].* = acc3;
    output[64..80].* = acc4;
    output[80..96].* = acc5;
    output[96..112].* = acc6;
    output[112..128].* = acc7;
}

fn accumulateQ8ValueHead128HeadMajor(
    output: []f32,
    scores: []const f32,
    value_cache_head: []const i8,
    value_scales_head: []const u16,
) void {
    std.debug.assert(output.len == handwritten_q8_head_dim);

    var acc0: @Vector(16, f32) = @splat(0.0);
    var acc1: @Vector(16, f32) = @splat(0.0);
    var acc2: @Vector(16, f32) = @splat(0.0);
    var acc3: @Vector(16, f32) = @splat(0.0);
    var acc4: @Vector(16, f32) = @splat(0.0);
    var acc5: @Vector(16, f32) = @splat(0.0);
    var acc6: @Vector(16, f32) = @splat(0.0);
    var acc7: @Vector(16, f32) = @splat(0.0);

    for (scores, 0..) |weight, pos| {
        const cache_start = pos * handwritten_q8_head_dim;
        const scale_start = pos * handwritten_q8_scale_groups;
        acc0 += scaledQ8Vector16(value_cache_head, cache_start + 0, weight * bfloat16.toF32(value_scales_head[scale_start + 0]));
        acc1 += scaledQ8Vector16(value_cache_head, cache_start + 16, weight * bfloat16.toF32(value_scales_head[scale_start + 1]));
        acc2 += scaledQ8Vector16(value_cache_head, cache_start + 32, weight * bfloat16.toF32(value_scales_head[scale_start + 2]));
        acc3 += scaledQ8Vector16(value_cache_head, cache_start + 48, weight * bfloat16.toF32(value_scales_head[scale_start + 3]));
        acc4 += scaledQ8Vector16(value_cache_head, cache_start + 64, weight * bfloat16.toF32(value_scales_head[scale_start + 4]));
        acc5 += scaledQ8Vector16(value_cache_head, cache_start + 80, weight * bfloat16.toF32(value_scales_head[scale_start + 5]));
        acc6 += scaledQ8Vector16(value_cache_head, cache_start + 96, weight * bfloat16.toF32(value_scales_head[scale_start + 6]));
        acc7 += scaledQ8Vector16(value_cache_head, cache_start + 112, weight * bfloat16.toF32(value_scales_head[scale_start + 7]));
    }

    output[0..16].* = acc0;
    output[16..32].* = acc1;
    output[32..48].* = acc2;
    output[48..64].* = acc3;
    output[64..80].* = acc4;
    output[80..96].* = acc5;
    output[96..112].* = acc6;
    output[112..128].* = acc7;
}

fn scaledQ8Vector16(input_q8: []const i8, start: usize, scale: f32) @Vector(16, f32) {
    const scale_vec: @Vector(16, f32) = @splat(scale);
    const input_i8: @Vector(16, i8) = input_q8[start..][0..16].*;
    const input_f32: @Vector(16, f32) = @floatFromInt(input_i8);
    return input_f32 * scale_vec;
}

test "q8 attention handwritten 128 full path matches generic path" {
    const testing = std.testing;

    const seq_len = 5;
    const num_query_heads = 4;
    const num_key_value_heads = 2;
    const head_dim = handwritten_q8_head_dim;
    const total_query = num_query_heads * head_dim;
    const total_cache = seq_len * num_key_value_heads * head_dim;
    const total_scales = seq_len * num_key_value_heads * handwritten_q8_scale_groups;

    var query: [total_query]f32 = undefined;
    var key_cache: [total_cache]i8 = undefined;
    var value_cache: [total_cache]i8 = undefined;
    var key_scales: [total_scales]u16 = undefined;
    var value_scales: [total_scales]u16 = undefined;
    var scores_generic: [seq_len]f32 = undefined;
    var scores_handwritten: [seq_len]f32 = undefined;
    var output_generic: [total_query]f32 = undefined;
    var output_handwritten: [total_query]f32 = undefined;

    for (&query, 0..) |*value, idx| {
        value.* = @as(f32, @floatFromInt(@as(i32, @intCast((idx * 7 + 3) % 37)) - 18)) / 9.0;
    }
    for (&key_cache, 0..) |*value, idx| {
        value.* = @intCast(@as(i16, @intCast((idx * 11 + 5) % 255)) - 127);
        value_cache[idx] = @intCast(@as(i16, @intCast((idx * 13 + 9) % 255)) - 127);
    }
    for (&key_scales, 0..) |*value, idx| {
        value.* = bfloat16.fromF32(@as(f32, @floatFromInt((idx % handwritten_q8_scale_groups) + 1)) / 127.0);
        value_scales[idx] = bfloat16.fromF32(@as(f32, @floatFromInt((idx % handwritten_q8_scale_groups) + 2)) / 127.0);
    }

    try scaledDotProductAttentionSingleQueryQ8CacheGeneric(
        &output_generic,
        &query,
        &key_cache,
        &key_scales,
        &value_cache,
        &value_scales,
        seq_len,
        num_query_heads,
        num_key_value_heads,
        head_dim,
        handwritten_q8_scale_groups,
        &scores_generic,
    );

    try scaledDotProductAttentionSingleQueryQ8Cache128(
        &output_handwritten,
        &query,
        &key_cache,
        &key_scales,
        &value_cache,
        &value_scales,
        seq_len,
        num_query_heads,
        num_key_value_heads,
        &scores_handwritten,
    );

    for (output_generic, output_handwritten) |expected, actual| {
        try testing.expectApproxEqAbs(expected, actual, 1e-5);
    }
}

test "q8 attention head-major path matches token-major path" {
    const testing = std.testing;

    const seq_len = 5;
    const num_query_heads = 4;
    const num_key_value_heads = 2;
    const head_dim = handwritten_q8_head_dim;
    const scale_groups_per_head = handwritten_q8_scale_groups;
    const total_query = num_query_heads * head_dim;
    const total_cache = seq_len * num_key_value_heads * head_dim;
    const total_scales = seq_len * num_key_value_heads * scale_groups_per_head;
    const head_data_stride = seq_len * head_dim;
    const head_scale_stride = seq_len * scale_groups_per_head;

    var query: [total_query]f32 = undefined;
    var key_cache_token_major: [total_cache]i8 = undefined;
    var value_cache_token_major: [total_cache]i8 = undefined;
    var key_scales_token_major: [total_scales]u16 = undefined;
    var value_scales_token_major: [total_scales]u16 = undefined;
    var key_cache_head_major: [total_cache]i8 = undefined;
    var value_cache_head_major: [total_cache]i8 = undefined;
    var key_scales_head_major: [total_scales]u16 = undefined;
    var value_scales_head_major: [total_scales]u16 = undefined;
    var scores_token_major: [seq_len]f32 = undefined;
    var scores_head_major: [seq_len]f32 = undefined;
    var output_token_major: [total_query]f32 = undefined;
    var output_head_major: [total_query]f32 = undefined;

    for (&query, 0..) |*value, idx| {
        value.* = @as(f32, @floatFromInt(@as(i32, @intCast((idx * 7 + 3) % 37)) - 18)) / 9.0;
    }
    for (&key_cache_token_major, 0..) |*value, idx| {
        value.* = @intCast(@as(i16, @intCast((idx * 11 + 5) % 255)) - 127);
        value_cache_token_major[idx] = @intCast(@as(i16, @intCast((idx * 13 + 9) % 255)) - 127);
    }
    for (&key_scales_token_major, 0..) |*value, idx| {
        value.* = bfloat16.fromF32(@as(f32, @floatFromInt((idx % scale_groups_per_head) + 1)) / 127.0);
        value_scales_token_major[idx] = bfloat16.fromF32(@as(f32, @floatFromInt((idx % scale_groups_per_head) + 2)) / 127.0);
    }

    for (0..num_key_value_heads) |head_idx| {
        for (0..seq_len) |pos| {
            const token_major_data_start = (pos * num_key_value_heads + head_idx) * head_dim;
            const token_major_scale_start = (pos * num_key_value_heads + head_idx) * scale_groups_per_head;
            const head_major_data_start = head_idx * head_data_stride + pos * head_dim;
            const head_major_scale_start = head_idx * head_scale_stride + pos * scale_groups_per_head;
            @memcpy(key_cache_head_major[head_major_data_start .. head_major_data_start + head_dim], key_cache_token_major[token_major_data_start .. token_major_data_start + head_dim]);
            @memcpy(value_cache_head_major[head_major_data_start .. head_major_data_start + head_dim], value_cache_token_major[token_major_data_start .. token_major_data_start + head_dim]);
            @memcpy(key_scales_head_major[head_major_scale_start .. head_major_scale_start + scale_groups_per_head], key_scales_token_major[token_major_scale_start .. token_major_scale_start + scale_groups_per_head]);
            @memcpy(value_scales_head_major[head_major_scale_start .. head_major_scale_start + scale_groups_per_head], value_scales_token_major[token_major_scale_start .. token_major_scale_start + scale_groups_per_head]);
        }
    }

    try scaledDotProductAttentionSingleQueryQ8Cache(
        &output_token_major,
        &query,
        &key_cache_token_major,
        &key_scales_token_major,
        &value_cache_token_major,
        &value_scales_token_major,
        seq_len,
        num_query_heads,
        num_key_value_heads,
        head_dim,
        &scores_token_major,
    );

    try scaledDotProductAttentionSingleQueryQ8CacheHeadMajor(
        &output_head_major,
        &query,
        &key_cache_head_major,
        &key_scales_head_major,
        &value_cache_head_major,
        &value_scales_head_major,
        head_data_stride,
        head_scale_stride,
        seq_len,
        num_query_heads,
        num_key_value_heads,
        head_dim,
        &scores_head_major,
    );

    for (output_token_major, output_head_major) |expected, actual| {
        try testing.expectApproxEqAbs(expected, actual, 1e-5);
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

test "rope table path matches direct rope" {
    const testing = std.testing;

    var table = try RoPETable.init(testing.allocator, 4, 4, 10000.0);
    defer table.deinit();

    var direct = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var cached = direct;

    try applyRoPEToHeadInPlace(&direct, 2, 10000.0);
    try applyRoPEToHeadWithTableInPlace(&cached, table.cosForPosition(2), table.sinForPosition(2));

    for (direct, cached) |lhs, rhs| {
        try testing.expectApproxEqAbs(lhs, rhs, 1e-6);
    }
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
