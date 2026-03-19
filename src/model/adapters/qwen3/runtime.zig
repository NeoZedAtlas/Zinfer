const std = @import("std");
const cpu = @import("../../../kernel/cpu.zig");
const kv_cache = @import("../../kv_cache.zig");
const adapter_block = @import("block.zig");
const adapter_config = @import("config.zig");
const adapter_spec = @import("spec.zig");
const adapter_weights = @import("weights.zig");
const tensor_store = @import("../../../tensor/store.zig");

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

pub const TopLogit = struct {
    token_id: usize,
    logit: f32,
};

pub fn forwardTokenId(
    allocator: std.mem.Allocator,
    store: *const tensor_store.TensorStore,
    cfg: adapter_config.Config,
    cache: *ModelCache,
    token_id: usize,
) ![]f32 {
    if (token_id >= cfg.vocab_size) return error.TokenIdOutOfBounds;
    if (cache.layers.len != cfg.num_hidden_layers) return error.CacheLayerMismatch;

    var hidden = try store.readRowAsF32Alloc(adapter_weights.common_weights.embed_tokens_weight, token_id);
    defer allocator.free(hidden);

    var scratch = try allocator.alloc(f32, cfg.hidden_size);
    defer allocator.free(scratch);

    for (0..cfg.num_hidden_layers) |layer_index| {
        const spec = adapter_spec.blockSpecFromConfig(cfg, layer_index);
        try adapter_block.forwardSingleToken(allocator, store, spec, &cache.layers[layer_index], hidden, scratch);
        std.mem.swap([]f32, &hidden, &scratch);
    }

    const final_norm_weight = try store.readElementsAsF32Alloc(adapter_weights.common_weights.final_norm_weight, 0, cfg.hidden_size);
    defer allocator.free(final_norm_weight);

    const final_hidden = try allocator.alloc(f32, cfg.hidden_size);
    defer allocator.free(final_hidden);
    try cpu.rmsNorm(final_hidden, hidden, final_norm_weight, @floatCast(cfg.rms_norm_eps));

    const logits = try allocator.alloc(f32, cfg.vocab_size);
    errdefer allocator.free(logits);
    try store.matmulVecByName(logits, adapter_weights.common_weights.lm_head_weight, final_hidden);
    return logits;
}

pub fn topKLogitsAlloc(
    allocator: std.mem.Allocator,
    logits: []const f32,
    k: usize,
) ![]TopLogit {
    const actual_k = @min(k, logits.len);
    const top = try allocator.alloc(TopLogit, actual_k);
    errdefer allocator.free(top);

    for (top, 0..) |*entry, idx| {
        entry.* = .{
            .token_id = idx,
            .logit = logits[idx],
        };
    }

    var cursor = actual_k;
    while (cursor < logits.len) : (cursor += 1) {
        var min_index: usize = 0;
        for (top[1..], 1..) |entry, idx| {
            if (entry.logit < top[min_index].logit) min_index = idx;
        }
        if (logits[cursor] > top[min_index].logit) {
            top[min_index] = .{
                .token_id = cursor,
                .logit = logits[cursor],
            };
        }
    }

    std.sort.block(TopLogit, top, {}, struct {
        fn lessThan(_: void, lhs: TopLogit, rhs: TopLogit) bool {
            return lhs.logit > rhs.logit;
        }
    }.lessThan);

    return top;
}

pub fn argMaxLogit(logits: []const f32) !usize {
    if (logits.len == 0) return error.EmptyLogits;

    var best_index: usize = 0;
    var best_value = logits[0];
    for (logits[1..], 1..) |value, idx| {
        if (value > best_value) {
            best_value = value;
            best_index = idx;
        }
    }
    return best_index;
}

test "topKLogitsAlloc returns logits in descending order" {
    const testing = std.testing;

    const logits = [_]f32{ 0.5, -1.0, 2.0, 1.5 };
    const top = try topKLogitsAlloc(testing.allocator, &logits, 3);
    defer testing.allocator.free(top);

    try testing.expectEqual(@as(usize, 2), top[0].token_id);
    try testing.expectEqual(@as(usize, 3), top[1].token_id);
    try testing.expectEqual(@as(usize, 0), top[2].token_id);
}

test "argMaxLogit returns highest index" {
    const testing = std.testing;

    try testing.expectEqual(@as(usize, 2), try argMaxLogit(&[_]f32{ -1.0, 0.0, 3.0, 2.0 }));
}
