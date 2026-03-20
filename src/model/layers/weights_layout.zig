pub const CommonWeights = struct {
    embed_tokens_weight: []const u8,
    final_norm_weight: []const u8,
    lm_head_weight: []const u8,
};

pub const LayerTensorKind = enum {
    input_layernorm_weight,
    self_attn_q_norm_weight,
    self_attn_k_norm_weight,
    self_attn_q_proj_weight,
    self_attn_k_proj_weight,
    self_attn_v_proj_weight,
    self_attn_o_proj_weight,
    post_attention_layernorm_weight,
    mlp_gate_proj_weight,
    mlp_up_proj_weight,
    mlp_down_proj_weight,
};
