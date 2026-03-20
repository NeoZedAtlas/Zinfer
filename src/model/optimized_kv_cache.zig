const std = @import("std");
const bfloat16 = @import("../tensor/bfloat16.zig");

pub const LayerKVCache = struct {
    allocator: std.mem.Allocator,
    max_seq_len: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    keys: []u16,
    values: []u16,
    len: usize,

    pub fn init(
        allocator: std.mem.Allocator,
        max_seq_len: usize,
        num_key_value_heads: usize,
        head_dim: usize,
    ) !LayerKVCache {
        const total = try std.math.mul(usize, max_seq_len, try std.math.mul(usize, num_key_value_heads, head_dim));
        const keys = try allocator.alloc(u16, total);
        errdefer allocator.free(keys);
        const values = try allocator.alloc(u16, total);
        errdefer allocator.free(values);

        @memset(keys, 0);
        @memset(values, 0);

        return .{
            .allocator = allocator,
            .max_seq_len = max_seq_len,
            .num_key_value_heads = num_key_value_heads,
            .head_dim = head_dim,
            .keys = keys,
            .values = values,
            .len = 0,
        };
    }

    pub fn deinit(self: *LayerKVCache) void {
        self.allocator.free(self.keys);
        self.allocator.free(self.values);
    }

    pub fn append(self: *LayerKVCache, key: []const f32, value: []const f32) !void {
        if (self.len >= self.max_seq_len) return error.CacheFull;

        const token_width = self.numKeyValueElementsPerToken();
        if (key.len != token_width or value.len != token_width) return error.SizeMismatch;

        const start = self.len * token_width;
        for (key, 0..) |element, idx| {
            self.keys[start + idx] = bfloat16.fromF32(element);
        }
        for (value, 0..) |element, idx| {
            self.values[start + idx] = bfloat16.fromF32(element);
        }
        self.len += 1;
    }

    pub fn currentKeys(self: *const LayerKVCache) []const u16 {
        return self.keys[0 .. self.len * self.numKeyValueElementsPerToken()];
    }

    pub fn currentValues(self: *const LayerKVCache) []const u16 {
        return self.values[0 .. self.len * self.numKeyValueElementsPerToken()];
    }

    pub fn numKeyValueElementsPerToken(self: *const LayerKVCache) usize {
        return self.num_key_value_heads * self.head_dim;
    }
};

pub const ModelCache = struct {
    allocator: std.mem.Allocator,
    layers: []LayerKVCache,

    pub fn init(
        allocator: std.mem.Allocator,
        num_layers: usize,
        max_seq_len: usize,
        num_key_value_heads: usize,
        head_dim: usize,
    ) !ModelCache {
        const layers = try allocator.alloc(LayerKVCache, num_layers);
        errdefer allocator.free(layers);

        var initialized: usize = 0;
        errdefer {
            for (layers[0..initialized]) |*layer| layer.deinit();
        }

        for (layers, 0..) |*layer, idx| {
            _ = idx;
            layer.* = try LayerKVCache.init(allocator, max_seq_len, num_key_value_heads, head_dim);
            initialized += 1;
        }

        return .{
            .allocator = allocator,
            .layers = layers,
        };
    }

    pub fn deinit(self: *ModelCache) void {
        for (self.layers) |*layer| layer.deinit();
        self.allocator.free(self.layers);
    }
};

pub fn estimateBytes(num_layers: usize, max_seq_len: usize, num_key_value_heads: usize, head_dim: usize) u64 {
    const total = @as(u128, num_layers) *
        @as(u128, max_seq_len) *
        @as(u128, num_key_value_heads) *
        @as(u128, head_dim) *
        2 *
        @sizeOf(u16);
    return @intCast(total);
}

test "optimized kv cache stores bf16 values in order" {
    const testing = std.testing;

    var cache = try LayerKVCache.init(testing.allocator, 2, 1, 2);
    defer cache.deinit();

    try cache.append(&[_]f32{ 1.0, 2.0 }, &[_]f32{ 3.0, 4.0 });
    try cache.append(&[_]f32{ 5.0, 6.0 }, &[_]f32{ 7.0, 8.0 });

    try testing.expectEqual(@as(usize, 2), cache.len);
    try testing.expectEqual(@as(f32, 1.0), bfloat16.toF32(cache.currentKeys()[0]));
    try testing.expectEqual(@as(f32, 6.0), bfloat16.toF32(cache.currentKeys()[3]));
    try testing.expectEqual(@as(f32, 3.0), bfloat16.toF32(cache.currentValues()[0]));
    try testing.expectEqual(@as(f32, 8.0), bfloat16.toF32(cache.currentValues()[3]));
}
