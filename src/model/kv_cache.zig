const std = @import("std");

pub const LayerKVCache = struct {
    allocator: std.mem.Allocator,
    max_seq_len: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    keys: []f32,
    values: []f32,
    len: usize,

    pub fn init(
        allocator: std.mem.Allocator,
        max_seq_len: usize,
        num_key_value_heads: usize,
        head_dim: usize,
    ) !LayerKVCache {
        const total = try std.math.mul(usize, max_seq_len, try std.math.mul(usize, num_key_value_heads, head_dim));
        const keys = try allocator.alloc(f32, total);
        errdefer allocator.free(keys);
        const values = try allocator.alloc(f32, total);
        errdefer allocator.free(values);

        @memset(keys, 0.0);
        @memset(values, 0.0);

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
        @memcpy(self.keys[start .. start + token_width], key);
        @memcpy(self.values[start .. start + token_width], value);
        self.len += 1;
    }

    pub fn currentKeys(self: *const LayerKVCache) []const f32 {
        return self.keys[0 .. self.len * self.numKeyValueElementsPerToken()];
    }

    pub fn currentValues(self: *const LayerKVCache) []const f32 {
        return self.values[0 .. self.len * self.numKeyValueElementsPerToken()];
    }

    pub fn numKeyValueElementsPerToken(self: *const LayerKVCache) usize {
        return self.num_key_value_heads * self.head_dim;
    }
};

test "layer kv cache appends tokens in order" {
    const testing = std.testing;

    var cache = try LayerKVCache.init(testing.allocator, 2, 1, 2);
    defer cache.deinit();

    try cache.append(&[_]f32{ 1.0, 2.0 }, &[_]f32{ 3.0, 4.0 });
    try cache.append(&[_]f32{ 5.0, 6.0 }, &[_]f32{ 7.0, 8.0 });

    try testing.expectEqual(@as(usize, 2), cache.len);
    try testing.expectEqualSlices(f32, &[_]f32{ 1.0, 2.0, 5.0, 6.0 }, cache.currentKeys());
    try testing.expectEqualSlices(f32, &[_]f32{ 3.0, 4.0, 7.0, 8.0 }, cache.currentValues());
}
