const std = @import("std");
const kv_cache = @import("kv_cache.zig");

pub const ModelCache = struct {
    allocator: std.mem.Allocator,
    layers: []kv_cache.LayerKVCache,

    pub fn init(
        allocator: std.mem.Allocator,
        num_layers: usize,
        max_seq_len: usize,
        num_key_value_heads: usize,
        head_dim: usize,
    ) !ModelCache {
        const layers = try allocator.alloc(kv_cache.LayerKVCache, num_layers);
        errdefer allocator.free(layers);

        for (layers, 0..) |*layer, idx| {
            _ = idx;
            layer.* = try kv_cache.LayerKVCache.init(allocator, max_seq_len, num_key_value_heads, head_dim);
        }

        return .{
            .allocator = allocator,
            .layers = layers,
        };
    }

    pub fn deinit(self: *ModelCache) void {
        for (self.layers) |*layer| {
            layer.deinit();
        }
        self.allocator.free(self.layers);
    }
};
