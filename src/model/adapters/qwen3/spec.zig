const std = @import("std");
const block = @import("block.zig");
const config = @import("config.zig");

pub fn blockSpecFromConfig(cfg: config.Config, layer_index: usize) block.BlockSpec {
    return blockSpecFromFields(
        layer_index,
        cfg.hidden_size,
        cfg.intermediate_size,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.head_dim,
        cfg.rope_theta,
        cfg.rms_norm_eps,
    );
}

pub fn blockSpecFromFields(
    layer_index: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    rope_theta: f64,
    rms_norm_eps: f64,
) block.BlockSpec {
    return .{
        .layer_index = layer_index,
        .hidden_size = hidden_size,
        .intermediate_size = intermediate_size,
        .num_attention_heads = num_attention_heads,
        .num_key_value_heads = num_key_value_heads,
        .head_dim = head_dim,
        .rope_theta = @floatCast(rope_theta),
        .rms_norm_eps = @floatCast(rms_norm_eps),
    };
}

test "adapter spec builds block spec from config" {
    const testing = std.testing;

    const cfg = config.Config{
        .head_dim = 128,
        .hidden_size = 1024,
        .intermediate_size = 3072,
        .max_position_embeddings = 40960,
        .model_type = "qwen3",
        .num_attention_heads = 16,
        .num_hidden_layers = 28,
        .num_key_value_heads = 8,
        .rms_norm_eps = 1e-6,
        .rope_theta = 1_000_000.0,
        .tie_word_embeddings = true,
        .torch_dtype = "bfloat16",
        .vocab_size = 151936,
    };

    const spec = blockSpecFromConfig(cfg, 3);
    try testing.expectEqual(@as(usize, 3), spec.layer_index);
    try testing.expectEqual(@as(usize, 1024), spec.hidden_size);
    try testing.expectEqual(@as(usize, 3072), spec.intermediate_size);
    try testing.expectEqual(@as(usize, 16), spec.num_attention_heads);
    try testing.expectEqual(@as(usize, 8), spec.num_key_value_heads);
    try testing.expectEqual(@as(usize, 128), spec.head_dim);
}

test "adapter spec builds block spec from scalar fields" {
    const testing = std.testing;

    const spec = blockSpecFromFields(4, 512, 1536, 8, 4, 64, 10_000.0, 1e-5);
    try testing.expectEqual(@as(usize, 4), spec.layer_index);
    try testing.expectEqual(@as(usize, 512), spec.hidden_size);
    try testing.expectEqual(@as(usize, 1536), spec.intermediate_size);
    try testing.expectEqual(@as(usize, 8), spec.num_attention_heads);
    try testing.expectEqual(@as(usize, 4), spec.num_key_value_heads);
    try testing.expectEqual(@as(usize, 64), spec.head_dim);
}
