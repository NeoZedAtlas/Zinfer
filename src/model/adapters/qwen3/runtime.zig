const std = @import("std");
const cpu = @import("../../../kernel/cpu.zig");
const decoder_cache = @import("../../decoder_cache.zig");
const logits_util = @import("../../logits.zig");
const adapter_block = @import("block.zig");
const adapter_config = @import("config.zig");
const adapter_spec = @import("spec.zig");
const adapter_weights = @import("weights.zig");
const tensor_store = @import("../../../tensor/store.zig");

pub const ModelCache = decoder_cache.ModelCache;
pub const TopLogit = logits_util.TopLogit;

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
    values: []const f32,
    k: usize,
) ![]TopLogit {
    return try logits_util.topKLogitsAlloc(allocator, values, k);
}

pub fn argMaxLogit(values: []const f32) !usize {
    return try logits_util.argMaxLogit(values);
}
