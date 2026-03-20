const std = @import("std");
const bfloat16 = @import("../tensor/bfloat16.zig");

pub const Scheme = enum {
    bf16,
    q8,

    pub fn name(self: Scheme) []const u8 {
        return switch (self) {
            .bf16 => "bf16",
            .q8 => "q8",
        };
    }
};

pub const LayerKVCache = struct {
    allocator: std.mem.Allocator,
    scheme: Scheme,
    max_seq_len: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    keys_bf16: []u16,
    values_bf16: []u16,
    keys_q8: []i8,
    values_q8: []i8,
    key_scales_q8: []u16,
    value_scales_q8: []u16,
    len: usize,

    pub fn init(
        allocator: std.mem.Allocator,
        max_seq_len: usize,
        num_key_value_heads: usize,
        head_dim: usize,
        scheme: Scheme,
    ) !LayerKVCache {
        const total = try std.math.mul(usize, max_seq_len, try std.math.mul(usize, num_key_value_heads, head_dim));
        const scale_total = try std.math.mul(usize, max_seq_len, num_key_value_heads);

        const keys_bf16 = try allocator.alloc(u16, if (scheme == .bf16) total else 0);
        errdefer allocator.free(keys_bf16);
        const values_bf16 = try allocator.alloc(u16, if (scheme == .bf16) total else 0);
        errdefer allocator.free(values_bf16);
        const keys_q8 = try allocator.alloc(i8, if (scheme == .q8) total else 0);
        errdefer allocator.free(keys_q8);
        const values_q8 = try allocator.alloc(i8, if (scheme == .q8) total else 0);
        errdefer allocator.free(values_q8);
        const key_scales_q8 = try allocator.alloc(u16, if (scheme == .q8) scale_total else 0);
        errdefer allocator.free(key_scales_q8);
        const value_scales_q8 = try allocator.alloc(u16, if (scheme == .q8) scale_total else 0);
        errdefer allocator.free(value_scales_q8);

        @memset(keys_bf16, 0);
        @memset(values_bf16, 0);
        @memset(keys_q8, 0);
        @memset(values_q8, 0);
        @memset(key_scales_q8, 0);
        @memset(value_scales_q8, 0);

        return .{
            .allocator = allocator,
            .scheme = scheme,
            .max_seq_len = max_seq_len,
            .num_key_value_heads = num_key_value_heads,
            .head_dim = head_dim,
            .keys_bf16 = keys_bf16,
            .values_bf16 = values_bf16,
            .keys_q8 = keys_q8,
            .values_q8 = values_q8,
            .key_scales_q8 = key_scales_q8,
            .value_scales_q8 = value_scales_q8,
            .len = 0,
        };
    }

    pub fn deinit(self: *LayerKVCache) void {
        self.allocator.free(self.keys_bf16);
        self.allocator.free(self.values_bf16);
        self.allocator.free(self.keys_q8);
        self.allocator.free(self.values_q8);
        self.allocator.free(self.key_scales_q8);
        self.allocator.free(self.value_scales_q8);
    }

    pub fn append(self: *LayerKVCache, key: []const f32, value: []const f32) !void {
        if (self.len >= self.max_seq_len) return error.CacheFull;

        const token_width = self.numKeyValueElementsPerToken();
        if (key.len != token_width or value.len != token_width) return error.SizeMismatch;

        switch (self.scheme) {
            .bf16 => self.appendBf16(key, value),
            .q8 => self.appendQ8(key, value),
        }
        self.len += 1;
    }

    pub fn currentBf16Keys(self: *const LayerKVCache) []const u16 {
        std.debug.assert(self.scheme == .bf16);
        return self.keys_bf16[0 .. self.len * self.numKeyValueElementsPerToken()];
    }

    pub fn currentBf16Values(self: *const LayerKVCache) []const u16 {
        std.debug.assert(self.scheme == .bf16);
        return self.values_bf16[0 .. self.len * self.numKeyValueElementsPerToken()];
    }

    pub fn currentQ8Keys(self: *const LayerKVCache) []const i8 {
        std.debug.assert(self.scheme == .q8);
        return self.keys_q8[0 .. self.len * self.numKeyValueElementsPerToken()];
    }

    pub fn currentQ8Values(self: *const LayerKVCache) []const i8 {
        std.debug.assert(self.scheme == .q8);
        return self.values_q8[0 .. self.len * self.numKeyValueElementsPerToken()];
    }

    pub fn currentQ8KeyScales(self: *const LayerKVCache) []const u16 {
        std.debug.assert(self.scheme == .q8);
        return self.key_scales_q8[0 .. self.len * self.num_key_value_heads];
    }

    pub fn currentQ8ValueScales(self: *const LayerKVCache) []const u16 {
        std.debug.assert(self.scheme == .q8);
        return self.value_scales_q8[0 .. self.len * self.num_key_value_heads];
    }

    pub fn numKeyValueElementsPerToken(self: *const LayerKVCache) usize {
        return self.num_key_value_heads * self.head_dim;
    }

    fn appendBf16(self: *LayerKVCache, key: []const f32, value: []const f32) void {
        const start = self.len * self.numKeyValueElementsPerToken();
        for (key, 0..) |element, idx| {
            self.keys_bf16[start + idx] = bfloat16.fromF32(element);
        }
        for (value, 0..) |element, idx| {
            self.values_bf16[start + idx] = bfloat16.fromF32(element);
        }
    }

    fn appendQ8(self: *LayerKVCache, key: []const f32, value: []const f32) void {
        const token_start = self.len * self.numKeyValueElementsPerToken();
        const scale_start = self.len * self.num_key_value_heads;

        for (0..self.num_key_value_heads) |head_idx| {
            const head_start = head_idx * self.head_dim;
            const key_slice = key[head_start .. head_start + self.head_dim];
            const value_slice = value[head_start .. head_start + self.head_dim];
            const key_out = self.keys_q8[token_start + head_start .. token_start + head_start + self.head_dim];
            const value_out = self.values_q8[token_start + head_start .. token_start + head_start + self.head_dim];

            self.key_scales_q8[scale_start + head_idx] = quantizeQ8Slice(key_out, key_slice);
            self.value_scales_q8[scale_start + head_idx] = quantizeQ8Slice(value_out, value_slice);
        }
    }
};

pub const ModelCache = struct {
    allocator: std.mem.Allocator,
    scheme: Scheme,
    layers: []LayerKVCache,

    pub fn init(
        allocator: std.mem.Allocator,
        num_layers: usize,
        max_seq_len: usize,
        num_key_value_heads: usize,
        head_dim: usize,
        scheme: Scheme,
    ) !ModelCache {
        const layers = try allocator.alloc(LayerKVCache, num_layers);
        errdefer allocator.free(layers);

        var initialized: usize = 0;
        errdefer {
            for (layers[0..initialized]) |*layer| layer.deinit();
        }

        for (layers, 0..) |*layer, idx| {
            _ = idx;
            layer.* = try LayerKVCache.init(allocator, max_seq_len, num_key_value_heads, head_dim, scheme);
            initialized += 1;
        }

        return .{
            .allocator = allocator,
            .scheme = scheme,
            .layers = layers,
        };
    }

    pub fn deinit(self: *ModelCache) void {
        for (self.layers) |*layer| layer.deinit();
        self.allocator.free(self.layers);
    }
};

pub fn estimateBytes(
    num_layers: usize,
    max_seq_len: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    scheme: Scheme,
) u64 {
    return switch (scheme) {
        .bf16 => blk: {
            const total = @as(u128, num_layers) *
                @as(u128, max_seq_len) *
                @as(u128, num_key_value_heads) *
                @as(u128, head_dim) *
                2 *
                @sizeOf(u16);
            break :blk @intCast(total);
        },
        .q8 => blk: {
            const quantized = @as(u128, num_layers) *
                @as(u128, max_seq_len) *
                @as(u128, num_key_value_heads) *
                @as(u128, head_dim) *
                2 *
                @sizeOf(i8);
            const scales = @as(u128, num_layers) *
                @as(u128, max_seq_len) *
                @as(u128, num_key_value_heads) *
                2 *
                @sizeOf(u16);
            break :blk @intCast(quantized + scales);
        },
    };
}

fn quantizeQ8Slice(output: []i8, input: []const f32) u16 {
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

test "optimized kv cache stores bf16 values in order" {
    const testing = std.testing;

    var cache = try LayerKVCache.init(testing.allocator, 2, 1, 2, .bf16);
    defer cache.deinit();

    try cache.append(&[_]f32{ 1.0, 2.0 }, &[_]f32{ 3.0, 4.0 });
    try cache.append(&[_]f32{ 5.0, 6.0 }, &[_]f32{ 7.0, 8.0 });

    try testing.expectEqual(@as(usize, 2), cache.len);
    try testing.expectEqual(@as(f32, 1.0), bfloat16.toF32(cache.currentBf16Keys()[0]));
    try testing.expectEqual(@as(f32, 6.0), bfloat16.toF32(cache.currentBf16Keys()[3]));
    try testing.expectEqual(@as(f32, 3.0), bfloat16.toF32(cache.currentBf16Values()[0]));
    try testing.expectEqual(@as(f32, 8.0), bfloat16.toF32(cache.currentBf16Values()[3]));
}

test "optimized kv cache stores q8 values with per-head scale" {
    const testing = std.testing;

    var cache = try LayerKVCache.init(testing.allocator, 1, 2, 2, .q8);
    defer cache.deinit();

    try cache.append(
        &[_]f32{ 1.0, -2.0, 0.25, 0.5 },
        &[_]f32{ 3.0, -4.0, 0.75, -1.0 },
    );

    try testing.expectEqual(@as(usize, 1), cache.len);
    try testing.expectEqual(@as(usize, 4), cache.currentQ8Keys().len);
    try testing.expectEqual(@as(usize, 2), cache.currentQ8KeyScales().len);
    try testing.expectApproxEqAbs(@as(f32, 2.0 / 127.0), bfloat16.toF32(cache.currentQ8KeyScales()[0]), 1e-3);
    try testing.expectApproxEqAbs(@as(f32, 0.5 / 127.0), bfloat16.toF32(cache.currentQ8KeyScales()[1]), 1e-3);
}
