const std = @import("std");
const cpu = @import("../../kernel/cpu.zig");
const decoder_cache = @import("../runtime/decoder_cache.zig");
const decoder_types = @import("../runtime/decoder_types.zig");
const generic_block = @import("rmsnorm_gqa_swiglu_block.zig");
const tensor_store = @import("../../tensor/store.zig");
const weights_layout = @import("weights_layout.zig");

pub const Config = struct {
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    vocab_size: usize,
    rope_theta: f64,
    rms_norm_eps: f64,

    pub fn blockSpec(self: Config, layer_index: usize) generic_block.Spec {
        return .{
            .layer_index = layer_index,
            .hidden_size = self.hidden_size,
            .intermediate_size = self.intermediate_size,
            .num_attention_heads = self.num_attention_heads,
            .num_key_value_heads = self.num_key_value_heads,
            .head_dim = self.head_dim,
            .rope_theta = @floatCast(self.rope_theta),
            .rms_norm_eps = @floatCast(self.rms_norm_eps),
        };
    }

    pub fn fromDecoderConfig(cfg: decoder_types.DecoderConfig) Config {
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
};

pub fn forwardTokenId(
    allocator: std.mem.Allocator,
    store: *const tensor_store.TensorStore,
    cfg: Config,
    common_weights: weights_layout.CommonWeights,
    layer_layout: generic_block.LayerLayout,
    layer_tensor_name_alloc: generic_block.LayerTensorNameFn,
    cache: *decoder_cache.ModelCache,
    token_id: usize,
) ![]f32 {
    if (token_id >= cfg.vocab_size) return error.TokenIdOutOfBounds;
    if (cache.layers.len != cfg.num_hidden_layers) return error.CacheLayerMismatch;

    var hidden = try store.readRowAsF32Alloc(common_weights.embed_tokens_weight, token_id);
    defer allocator.free(hidden);

    var scratch = try allocator.alloc(f32, cfg.hidden_size);
    defer allocator.free(scratch);

    for (0..cfg.num_hidden_layers) |layer_index| {
        const spec = cfg.blockSpec(layer_index);
        try generic_block.forwardSingleToken(
            allocator,
            store,
            spec,
            layer_layout,
            layer_tensor_name_alloc,
            &cache.layers[layer_index],
            hidden,
            scratch,
        );
        std.mem.swap([]f32, &hidden, &scratch);
    }

    const final_norm_weight = try store.readElementsAsF32Alloc(common_weights.final_norm_weight, 0, cfg.hidden_size);
    defer allocator.free(final_norm_weight);

    const final_hidden = try allocator.alloc(f32, cfg.hidden_size);
    defer allocator.free(final_hidden);
    try cpu.rmsNorm(final_hidden, hidden, final_norm_weight, @floatCast(cfg.rms_norm_eps));

    const logits = try allocator.alloc(f32, cfg.vocab_size);
    errdefer allocator.free(logits);
    try store.matmulVecByName(logits, common_weights.lm_head_weight, final_hidden);
    return logits;
}

test "decoder stack config builds block spec from layer index" {
    const testing = std.testing;

    const cfg = Config{
        .hidden_size = 1024,
        .intermediate_size = 3072,
        .num_hidden_layers = 28,
        .num_attention_heads = 16,
        .num_key_value_heads = 8,
        .head_dim = 64,
        .vocab_size = 151936,
        .rope_theta = 1_000_000.0,
        .rms_norm_eps = 1e-6,
    };

    const spec = cfg.blockSpec(5);
    try testing.expectEqual(@as(usize, 5), spec.layer_index);
    try testing.expectEqual(@as(usize, 1024), spec.hidden_size);
    try testing.expectEqual(@as(usize, 3072), spec.intermediate_size);
    try testing.expectEqual(@as(usize, 16), spec.num_attention_heads);
    try testing.expectEqual(@as(usize, 8), spec.num_key_value_heads);
    try testing.expectEqual(@as(usize, 64), spec.head_dim);
}
