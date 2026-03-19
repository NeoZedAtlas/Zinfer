const std = @import("std");
const decoder_cache = @import("../../decoder_cache.zig");
const decoder_only_stack = @import("../../decoder_only_stack.zig");
const logits_util = @import("../../logits.zig");
const adapter_config = @import("config.zig");
const adapter_layout = @import("layout.zig");
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
    return try decoder_only_stack.forwardTokenId(
        allocator,
        store,
        stackConfigFromAdapter(cfg),
        adapter_weights.common_weights,
        adapter_layout.layer_layout,
        adapter_weights.layerTensorNameAlloc,
        cache,
        token_id,
    );
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

fn stackConfigFromAdapter(cfg: adapter_config.Config) decoder_only_stack.Config {
    return .{
        .hidden_size = cfg.hidden_size,
        .intermediate_size = cfg.intermediate_size,
        .num_hidden_layers = cfg.num_hidden_layers,
        .num_attention_heads = cfg.num_attention_heads,
        .num_key_value_heads = cfg.num_key_value_heads,
        .head_dim = cfg.head_dim,
        .vocab_size = cfg.vocab_size,
        .rope_theta = cfg.rope_theta,
        .rms_norm_eps = cfg.rms_norm_eps,
    };
}
