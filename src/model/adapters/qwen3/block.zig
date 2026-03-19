const std = @import("std");
const cpu = @import("../../../kernel/cpu.zig");
const tensor_store = @import("../../../tensor/store.zig");
const kv_cache_mod = @import("../../kv_cache.zig");
const adapter_attention = @import("attention.zig");
const adapter_weights = @import("weights.zig");

pub const BlockSpec = struct {
    layer_index: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    rope_theta: f32,
    rms_norm_eps: f32,

    pub fn validate(self: BlockSpec) !void {
        if (self.hidden_size == 0) return error.InvalidHiddenSize;
        if (self.intermediate_size == 0) return error.InvalidIntermediateSize;
        if (self.num_attention_heads == 0) return error.InvalidAttentionHeads;
        if (self.num_key_value_heads == 0) return error.InvalidKeyValueHeads;
        if (self.num_attention_heads % self.num_key_value_heads != 0) return error.InvalidGrouping;
    }

    pub fn attentionSpec(self: BlockSpec) adapter_attention.AttentionSpec {
        return .{
            .hidden_size = self.num_attention_heads * self.head_dim,
            .num_attention_heads = self.num_attention_heads,
            .num_key_value_heads = self.num_key_value_heads,
            .head_dim = self.head_dim,
            .rope_theta = self.rope_theta,
        };
    }
};

pub fn forwardSingleToken(
    allocator: std.mem.Allocator,
    store: *const tensor_store.TensorStore,
    spec: BlockSpec,
    cache: *kv_cache_mod.LayerKVCache,
    hidden_in: []const f32,
    hidden_out: []f32,
) !void {
    try spec.validate();
    if (hidden_in.len != spec.hidden_size or hidden_out.len != spec.hidden_size) return error.SizeMismatch;
    if (cache.num_key_value_heads != spec.num_key_value_heads or cache.head_dim != spec.head_dim) {
        return error.CacheSpecMismatch;
    }

    const input_ln_weight = try loadVectorWeight(allocator, store, spec, .input_layernorm_weight, spec.hidden_size);
    defer allocator.free(input_ln_weight);
    const q_norm_weight = try loadVectorWeight(allocator, store, spec, .self_attn_q_norm_weight, spec.head_dim);
    defer allocator.free(q_norm_weight);
    const k_norm_weight = try loadVectorWeight(allocator, store, spec, .self_attn_k_norm_weight, spec.head_dim);
    defer allocator.free(k_norm_weight);
    const post_ln_weight = try loadVectorWeight(allocator, store, spec, .post_attention_layernorm_weight, spec.hidden_size);
    defer allocator.free(post_ln_weight);

    const normed = try allocator.alloc(f32, spec.hidden_size);
    defer allocator.free(normed);
    try cpu.rmsNorm(normed, hidden_in, input_ln_weight, spec.rms_norm_eps);

    const q_proj = try allocator.alloc(f32, spec.num_attention_heads * spec.head_dim);
    defer allocator.free(q_proj);
    try matmulWeight(allocator, store, spec, .self_attn_q_proj_weight, q_proj, normed);

    const kv_width = spec.num_key_value_heads * spec.head_dim;
    const k_proj = try allocator.alloc(f32, kv_width);
    defer allocator.free(k_proj);
    try matmulWeight(allocator, store, spec, .self_attn_k_proj_weight, k_proj, normed);

    const v_proj = try allocator.alloc(f32, kv_width);
    defer allocator.free(v_proj);
    try matmulWeight(allocator, store, spec, .self_attn_v_proj_weight, v_proj, normed);

    const q_normed = try allocator.alloc(f32, q_proj.len);
    defer allocator.free(q_normed);
    try cpu.rmsNormRepeated(
        q_normed,
        q_proj,
        spec.num_attention_heads,
        spec.head_dim,
        q_norm_weight,
        spec.rms_norm_eps,
    );

    const k_normed = try allocator.alloc(f32, k_proj.len);
    defer allocator.free(k_normed);
    try cpu.rmsNormRepeated(
        k_normed,
        k_proj,
        spec.num_key_value_heads,
        spec.head_dim,
        k_norm_weight,
        spec.rms_norm_eps,
    );

    const position = cache.len;
    try adapter_attention.applyRoPEToProjectedHeadsInPlace(spec.attentionSpec(), q_normed, k_normed, position);
    try cache.append(k_normed, v_proj);

    const attn_flat = try allocator.alloc(f32, spec.num_attention_heads * spec.head_dim);
    defer allocator.free(attn_flat);
    const scores = try allocator.alloc(f32, cache.len);
    defer allocator.free(scores);
    try adapter_attention.forwardProjectedSingleToken(
        spec.attentionSpec(),
        attn_flat,
        q_normed,
        cache.currentKeys(),
        cache.currentValues(),
        cache.len,
        scores,
    );

    const attn_out = try allocator.alloc(f32, spec.hidden_size);
    defer allocator.free(attn_out);
    try matmulWeight(allocator, store, spec, .self_attn_o_proj_weight, attn_out, attn_flat);

    const post_attn = try allocator.alloc(f32, spec.hidden_size);
    defer allocator.free(post_attn);
    for (post_attn, hidden_in, attn_out) |*out, residual, attn_value| {
        out.* = residual + attn_value;
    }

    const post_normed = try allocator.alloc(f32, spec.hidden_size);
    defer allocator.free(post_normed);
    try cpu.rmsNorm(post_normed, post_attn, post_ln_weight, spec.rms_norm_eps);

    const gate = try allocator.alloc(f32, spec.intermediate_size);
    defer allocator.free(gate);
    try matmulWeight(allocator, store, spec, .mlp_gate_proj_weight, gate, post_normed);

    const up = try allocator.alloc(f32, spec.intermediate_size);
    defer allocator.free(up);
    try matmulWeight(allocator, store, spec, .mlp_up_proj_weight, up, post_normed);

    const activated = try allocator.alloc(f32, spec.intermediate_size);
    defer allocator.free(activated);
    try cpu.swiglu(activated, gate, up);

    const mlp_out = try allocator.alloc(f32, spec.hidden_size);
    defer allocator.free(mlp_out);
    try matmulWeight(allocator, store, spec, .mlp_down_proj_weight, mlp_out, activated);

    for (hidden_out, post_attn, mlp_out) |*out, residual, mlp_value| {
        out.* = residual + mlp_value;
    }
}

fn matmulWeight(
    allocator: std.mem.Allocator,
    store: *const tensor_store.TensorStore,
    spec: BlockSpec,
    kind: @import("../../weights_layout.zig").LayerTensorKind,
    output: []f32,
    input: []const f32,
) !void {
    const name = try adapter_weights.layerTensorNameAlloc(allocator, spec.layer_index, kind);
    defer allocator.free(name);
    try store.matmulVecByName(output, name, input);
}

fn loadVectorWeight(
    allocator: std.mem.Allocator,
    store: *const tensor_store.TensorStore,
    spec: BlockSpec,
    kind: @import("../../weights_layout.zig").LayerTensorKind,
    expected_len: usize,
) ![]f32 {
    const name = try adapter_weights.layerTensorNameAlloc(allocator, spec.layer_index, kind);
    defer allocator.free(name);
    const values = try store.readElementsAsF32Alloc(name, 0, expected_len);
    errdefer allocator.free(values);
    if (values.len != expected_len) return error.SizeMismatch;
    return values;
}
