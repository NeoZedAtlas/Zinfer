const std = @import("std");
const generic_block = @import("../../rmsnorm_gqa_swiglu_block.zig");
const tensor_store = @import("../../../tensor/store.zig");
const kv_cache_mod = @import("../../kv_cache.zig");
const adapter_weights = @import("weights.zig");

pub const BlockSpec = generic_block.Spec;

pub const layer_layout = generic_block.LayerLayout{
    .q_norm_kind = .self_attn_q_norm_weight,
    .k_norm_kind = .self_attn_k_norm_weight,
};

pub fn forwardSingleToken(
    allocator: std.mem.Allocator,
    store: *const tensor_store.TensorStore,
    spec: BlockSpec,
    cache: *kv_cache_mod.LayerKVCache,
    hidden_in: []const f32,
    hidden_out: []f32,
) !void {
    return try generic_block.forwardSingleToken(
        allocator,
        store,
        spec,
        layer_layout,
        adapter_weights.layerTensorNameAlloc,
        cache,
        hidden_in,
        hidden_out,
    );
}
