const std = @import("std");
const attention = @import("../../../../kernel/attention/attention.zig");
const bfloat16 = @import("../../../../tensor/formats/bfloat16.zig");
const optimized_kv_cache = @import("../../../../model/runtime/optimized_kv_cache.zig");
const decoder_family = @import("../../../../model/runtime/decoder_family.zig");

pub fn estimateKvCacheBytes(
    cfg: decoder_family.DecoderConfig,
    max_seq_len: usize,
    kv_cache_scheme: optimized_kv_cache.Scheme,
    q8_layout: optimized_kv_cache.Q8Layout,
) u64 {
    return optimized_kv_cache.estimateBytesWithLayout(
        cfg.num_hidden_layers,
        max_seq_len,
        cfg.num_key_value_heads,
        cfg.head_dim,
        kv_cache_scheme,
        q8_layout,
    );
}

pub fn nsToMs(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1_000_000.0;
}

pub fn bytesToMiB(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / (1024.0 * 1024.0);
}

pub fn tokensPerSecond(token_count: usize, elapsed_ns: u64) f64 {
    if (token_count == 0 or elapsed_ns == 0) return 0.0;
    return @as(f64, @floatFromInt(token_count)) * 1_000_000_000.0 / @as(f64, @floatFromInt(elapsed_ns));
}

pub fn fillSyntheticF32(output: []f32, salt: usize) void {
    for (output, 0..) |*value, idx| {
        const bucket = @as(i32, @intCast((idx * 17 + salt) % 31)) - 15;
        value.* = @as(f32, @floatFromInt(bucket)) / 8.0;
    }
}

pub fn fillSyntheticQ8Cache(
    values: []i8,
    scales: []u16,
    head_dim: usize,
    salt: usize,
) void {
    const groups_per_head = std.math.divCeil(usize, head_dim, attention.q8_cache_group_size) catch unreachable;
    for (values, 0..) |*value, idx| {
        const bucket = @as(i16, @intCast((idx * 11 + salt) % 255)) - 127;
        value.* = @intCast(bucket);
    }
    for (scales, 0..) |*scale, idx| {
        const group_idx = idx % groups_per_head;
        const magnitude = @as(f32, @floatFromInt(@as(u32, @intCast((group_idx + salt) % 13 + 1))));
        scale.* = bfloat16.fromF32(magnitude / 127.0);
    }
}

pub fn transposeQ8CacheTokenToHeadMajor(
    dst_values: []i8,
    dst_scales: []u16,
    src_values: []const i8,
    src_scales: []const u16,
    seq_len: usize,
    num_key_value_heads: usize,
    head_dim: usize,
) void {
    const groups_per_head = std.math.divCeil(usize, head_dim, attention.q8_cache_group_size) catch unreachable;
    const head_data_stride = seq_len * head_dim;
    const head_scale_stride = seq_len * groups_per_head;

    for (0..num_key_value_heads) |head_idx| {
        for (0..seq_len) |pos| {
            const token_major_data_start = (pos * num_key_value_heads + head_idx) * head_dim;
            const token_major_scale_start = (pos * num_key_value_heads + head_idx) * groups_per_head;
            const head_major_data_start = head_idx * head_data_stride + pos * head_dim;
            const head_major_scale_start = head_idx * head_scale_stride + pos * groups_per_head;

            @memcpy(
                dst_values[head_major_data_start .. head_major_data_start + head_dim],
                src_values[token_major_data_start .. token_major_data_start + head_dim],
            );
            @memcpy(
                dst_scales[head_major_scale_start .. head_major_scale_start + groups_per_head],
                src_scales[token_major_scale_start .. token_major_scale_start + groups_per_head],
            );
        }
    }
}

pub fn transposeQ8CacheTokenToPagedHeadMajor(
    dst_values: []i8,
    dst_scales: []u16,
    src_values: []const i8,
    src_scales: []const u16,
    seq_len: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    page_len: usize,
) void {
    const groups_per_head = std.math.divCeil(usize, head_dim, attention.q8_cache_group_size) catch unreachable;
    const pages_per_head = std.math.divCeil(usize, seq_len, page_len) catch unreachable;
    const page_data_stride = page_len * head_dim;
    const page_scale_stride = page_len * groups_per_head;
    const head_data_stride = pages_per_head * page_data_stride;
    const head_scale_stride = pages_per_head * page_scale_stride;

    @memset(dst_values, 0);
    @memset(dst_scales, 0);

    for (0..num_key_value_heads) |head_idx| {
        for (0..seq_len) |pos| {
            const token_major_data_start = (pos * num_key_value_heads + head_idx) * head_dim;
            const token_major_scale_start = (pos * num_key_value_heads + head_idx) * groups_per_head;
            const page_idx = pos / page_len;
            const page_offset = pos % page_len;
            const paged_data_start = head_idx * head_data_stride + page_idx * page_data_stride + page_offset * head_dim;
            const paged_scale_start = head_idx * head_scale_stride + page_idx * page_scale_stride + page_offset * groups_per_head;

            @memcpy(
                dst_values[paged_data_start .. paged_data_start + head_dim],
                src_values[token_major_data_start .. token_major_data_start + head_dim],
            );
            @memcpy(
                dst_scales[paged_scale_start .. paged_scale_start + groups_per_head],
                src_scales[token_major_scale_start .. token_major_scale_start + groups_per_head],
            );
        }
    }
}

pub fn encodeF32Row(output: []u8, values: []const f32) void {
    for (values, 0..) |value, idx| {
        const start = idx * 4;
        std.mem.writeInt(u32, output[start .. start + 4][0..4], @bitCast(value), .little);
    }
}

pub fn encodeBf16Row(output: []u8, values: []const f32) void {
    for (values, 0..) |value, idx| {
        const start = idx * 2;
        std.mem.writeInt(u16, output[start .. start + 2][0..2], bfloat16.fromF32(value), .little);
    }
}

pub fn runAttentionDotSweep(
    query: []const f32,
    key_cache: []const i8,
    key_scales: []const u16,
    head_dim: usize,
    scale_groups_per_head: usize,
) f32 {
    var sum: f32 = 0.0;
    var cache_start: usize = 0;
    var scale_start: usize = 0;
    while (cache_start < key_cache.len) : ({
        cache_start += head_dim;
        scale_start += scale_groups_per_head;
    }) {
        sum += attention.dotQ8GroupedSlice(
            query,
            key_cache[cache_start .. cache_start + head_dim],
            key_scales[scale_start .. scale_start + scale_groups_per_head],
        );
    }
    return sum;
}

pub fn runAttentionAxpySweep(
    output: []f32,
    value_cache: []const i8,
    value_scales: []const u16,
    head_dim: usize,
    scale_groups_per_head: usize,
) f32 {
    var cache_start: usize = 0;
    var scale_start: usize = 0;
    while (cache_start < value_cache.len) : ({
        cache_start += head_dim;
        scale_start += scale_groups_per_head;
    }) {
        attention.axpyQ8GroupedSliceInPlace(
            output,
            1.0,
            value_cache[cache_start .. cache_start + head_dim],
            value_scales[scale_start .. scale_start + scale_groups_per_head],
        );
    }
    return output[0] + output[output.len - 1];
}

pub fn resolveBenchIterations(requested_iterations: usize, work_per_iteration: usize, target_elements: usize) usize {
    if (requested_iterations != 0) return requested_iterations;
    const safe_work = @max(work_per_iteration, 1);
    const auto_iterations = target_elements / safe_work;
    return @max(@as(usize, 8), @min(@as(usize, 4096), auto_iterations));
}

pub fn nsPerIteration(elapsed_ns: u64, iterations: usize) f64 {
    if (iterations == 0) return 0.0;
    return @as(f64, @floatFromInt(elapsed_ns)) / @as(f64, @floatFromInt(iterations));
}

pub fn millionElementsPerSecond(work_per_iteration: usize, iterations: usize, elapsed_ns: u64) f64 {
    if (work_per_iteration == 0 or iterations == 0 or elapsed_ns == 0) return 0.0;
    const total_elements = @as(f64, @floatFromInt(work_per_iteration)) * @as(f64, @floatFromInt(iterations));
    return total_elements / @as(f64, @floatFromInt(elapsed_ns)) * 1_000.0;
}
