const std = @import("std");
const cpu = @import("../../kernel/core/cpu.zig");
const tensor_store = @import("../../tensor/storage/store.zig");
const kv_cache_mod = @import("../runtime/kv_cache.zig");
const gqa_attention = @import("gqa_attention.zig");
const weights_layout = @import("weights_layout.zig");

pub const LayerTensorNameFn = *const fn (std.mem.Allocator, usize, weights_layout.LayerTensorKind) anyerror![]u8;

pub const LayerLayout = struct {
    input_layernorm_kind: weights_layout.LayerTensorKind = .input_layernorm_weight,
    q_norm_kind: ?weights_layout.LayerTensorKind = null,
    k_norm_kind: ?weights_layout.LayerTensorKind = null,
    q_proj_kind: weights_layout.LayerTensorKind = .self_attn_q_proj_weight,
    k_proj_kind: weights_layout.LayerTensorKind = .self_attn_k_proj_weight,
    v_proj_kind: weights_layout.LayerTensorKind = .self_attn_v_proj_weight,
    o_proj_kind: weights_layout.LayerTensorKind = .self_attn_o_proj_weight,
    post_attention_layernorm_kind: weights_layout.LayerTensorKind = .post_attention_layernorm_weight,
    gate_proj_kind: weights_layout.LayerTensorKind = .mlp_gate_proj_weight,
    up_proj_kind: weights_layout.LayerTensorKind = .mlp_up_proj_weight,
    down_proj_kind: weights_layout.LayerTensorKind = .mlp_down_proj_weight,
};

pub const Spec = struct {
    layer_index: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    rope_theta: f32,
    rms_norm_eps: f32,

    pub fn validate(self: Spec) !void {
        if (self.hidden_size == 0) return error.InvalidHiddenSize;
        if (self.intermediate_size == 0) return error.InvalidIntermediateSize;
        if (self.num_attention_heads == 0) return error.InvalidAttentionHeads;
        if (self.num_key_value_heads == 0) return error.InvalidKeyValueHeads;
        if (self.num_attention_heads % self.num_key_value_heads != 0) return error.InvalidGrouping;
    }

    pub fn attentionSpec(self: Spec) gqa_attention.AttentionSpec {
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
    spec: Spec,
    layout: LayerLayout,
    layer_tensor_name_alloc: LayerTensorNameFn,
    cache: *kv_cache_mod.LayerKVCache,
    hidden_in: []const f32,
    hidden_out: []f32,
) !void {
    try spec.validate();
    if (hidden_in.len != spec.hidden_size or hidden_out.len != spec.hidden_size) return error.SizeMismatch;
    if (cache.num_key_value_heads != spec.num_key_value_heads or cache.head_dim != spec.head_dim) {
        return error.CacheSpecMismatch;
    }

    const input_ln_weight = try loadVectorWeight(allocator, store, spec, layer_tensor_name_alloc, layout.input_layernorm_kind, spec.hidden_size);
    defer allocator.free(input_ln_weight);
    const post_ln_weight = try loadVectorWeight(allocator, store, spec, layer_tensor_name_alloc, layout.post_attention_layernorm_kind, spec.hidden_size);
    defer allocator.free(post_ln_weight);

    const q_norm_weight = if (layout.q_norm_kind) |kind|
        try loadVectorWeight(allocator, store, spec, layer_tensor_name_alloc, kind, spec.head_dim)
    else
        null;
    defer if (q_norm_weight) |buffer| allocator.free(buffer);

    const k_norm_weight = if (layout.k_norm_kind) |kind|
        try loadVectorWeight(allocator, store, spec, layer_tensor_name_alloc, kind, spec.head_dim)
    else
        null;
    defer if (k_norm_weight) |buffer| allocator.free(buffer);

    const normed = try allocator.alloc(f32, spec.hidden_size);
    defer allocator.free(normed);
    try cpu.rmsNorm(normed, hidden_in, input_ln_weight, spec.rms_norm_eps);

    const q_proj = try allocator.alloc(f32, spec.num_attention_heads * spec.head_dim);
    defer allocator.free(q_proj);
    try matmulWeight(allocator, store, spec, layout, layer_tensor_name_alloc, layout.q_proj_kind, q_proj, normed);

    const kv_width = spec.num_key_value_heads * spec.head_dim;
    const k_proj = try allocator.alloc(f32, kv_width);
    defer allocator.free(k_proj);
    try matmulWeight(allocator, store, spec, layout, layer_tensor_name_alloc, layout.k_proj_kind, k_proj, normed);

    const v_proj = try allocator.alloc(f32, kv_width);
    defer allocator.free(v_proj);
    try matmulWeight(allocator, store, spec, layout, layer_tensor_name_alloc, layout.v_proj_kind, v_proj, normed);

    const q_normed = try allocator.alloc(f32, q_proj.len);
    defer allocator.free(q_normed);
    if (q_norm_weight) |weight| {
        try cpu.rmsNormRepeated(
            q_normed,
            q_proj,
            spec.num_attention_heads,
            spec.head_dim,
            weight,
            spec.rms_norm_eps,
        );
    } else {
        @memcpy(q_normed, q_proj);
    }

    const k_normed = try allocator.alloc(f32, k_proj.len);
    defer allocator.free(k_normed);
    if (k_norm_weight) |weight| {
        try cpu.rmsNormRepeated(
            k_normed,
            k_proj,
            spec.num_key_value_heads,
            spec.head_dim,
            weight,
            spec.rms_norm_eps,
        );
    } else {
        @memcpy(k_normed, k_proj);
    }

    const position = cache.len;
    try gqa_attention.applyRoPEToProjectedHeadsInPlace(spec.attentionSpec(), q_normed, k_normed, position);
    try cache.append(k_normed, v_proj);

    const attn_flat = try allocator.alloc(f32, spec.num_attention_heads * spec.head_dim);
    defer allocator.free(attn_flat);
    const scores = try allocator.alloc(f32, cache.len);
    defer allocator.free(scores);
    try gqa_attention.forwardProjectedSingleToken(
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
    try matmulWeight(allocator, store, spec, layout, layer_tensor_name_alloc, layout.o_proj_kind, attn_out, attn_flat);

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
    try matmulWeight(allocator, store, spec, layout, layer_tensor_name_alloc, layout.gate_proj_kind, gate, post_normed);

    const up = try allocator.alloc(f32, spec.intermediate_size);
    defer allocator.free(up);
    try matmulWeight(allocator, store, spec, layout, layer_tensor_name_alloc, layout.up_proj_kind, up, post_normed);

    const activated = try allocator.alloc(f32, spec.intermediate_size);
    defer allocator.free(activated);
    try cpu.swiglu(activated, gate, up);

    const mlp_out = try allocator.alloc(f32, spec.hidden_size);
    defer allocator.free(mlp_out);
    try matmulWeight(allocator, store, spec, layout, layer_tensor_name_alloc, layout.down_proj_kind, mlp_out, activated);

    for (hidden_out, post_attn, mlp_out) |*out, residual, mlp_value| {
        out.* = residual + mlp_value;
    }
}

fn matmulWeight(
    allocator: std.mem.Allocator,
    store: *const tensor_store.TensorStore,
    spec: Spec,
    _: LayerLayout,
    layer_tensor_name_alloc: LayerTensorNameFn,
    kind: weights_layout.LayerTensorKind,
    output: []f32,
    input: []const f32,
) !void {
    const name = try layer_tensor_name_alloc(allocator, spec.layer_index, kind);
    defer allocator.free(name);
    try store.matmulVecByName(output, name, input);
}

fn loadVectorWeight(
    allocator: std.mem.Allocator,
    store: *const tensor_store.TensorStore,
    spec: Spec,
    layer_tensor_name_alloc: LayerTensorNameFn,
    kind: weights_layout.LayerTensorKind,
    expected_len: usize,
) ![]f32 {
    const name = try layer_tensor_name_alloc(allocator, spec.layer_index, kind);
    defer allocator.free(name);
    const values = try store.readElementsAsF32Alloc(name, 0, expected_len);
    errdefer allocator.free(values);
    if (values.len != expected_len) return error.SizeMismatch;
    return values;
}

test "decoder block spec validates dimensions" {
    const testing = std.testing;

    const spec = Spec{
        .layer_index = 0,
        .hidden_size = 1024,
        .intermediate_size = 3072,
        .num_attention_heads = 16,
        .num_key_value_heads = 8,
        .head_dim = 64,
        .rope_theta = 1000000.0,
        .rms_norm_eps = 1e-6,
    };
    try spec.validate();
    try testing.expectEqual(@as(usize, 1024), spec.attentionSpec().hidden_size);
}
