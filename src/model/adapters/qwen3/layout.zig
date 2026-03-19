const generic_block = @import("../../rmsnorm_gqa_swiglu_block.zig");

pub const layer_layout = generic_block.LayerLayout{
    .q_norm_kind = .self_attn_q_norm_weight,
    .k_norm_kind = .self_attn_k_norm_weight,
};
