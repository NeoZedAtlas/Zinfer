const std = @import("std");
const weights_layout = @import("../../layers/weights_layout.zig");

pub const common_weights = weights_layout.CommonWeights{
    .embed_tokens_weight = "model.embed_tokens.weight",
    .final_norm_weight = "model.norm.weight",
    .lm_head_weight = "lm_head.weight",
};

pub fn layerTensorNameAlloc(
    allocator: std.mem.Allocator,
    layer_index: usize,
    kind: weights_layout.LayerTensorKind,
) ![]u8 {
    return std.fmt.allocPrint(allocator, "model.layers.{d}.{s}", .{
        layer_index,
        suffixForKind(kind),
    });
}

pub fn suffixForKind(kind: weights_layout.LayerTensorKind) []const u8 {
    return switch (kind) {
        .input_layernorm_weight => "input_layernorm.weight",
        .self_attn_q_norm_weight => "self_attn.q_norm.weight",
        .self_attn_k_norm_weight => "self_attn.k_norm.weight",
        .self_attn_q_proj_weight => "self_attn.q_proj.weight",
        .self_attn_k_proj_weight => "self_attn.k_proj.weight",
        .self_attn_v_proj_weight => "self_attn.v_proj.weight",
        .self_attn_o_proj_weight => "self_attn.o_proj.weight",
        .post_attention_layernorm_weight => "post_attention_layernorm.weight",
        .mlp_gate_proj_weight => "mlp.gate_proj.weight",
        .mlp_up_proj_weight => "mlp.up_proj.weight",
        .mlp_down_proj_weight => "mlp.down_proj.weight",
    };
}

test "adapter weights produce expected layer tensor name" {
    const testing = std.testing;

    const name = try layerTensorNameAlloc(testing.allocator, 7, .self_attn_q_proj_weight);
    defer testing.allocator.free(name);

    try testing.expectEqualStrings("model.layers.7.self_attn.q_proj.weight", name);
}
