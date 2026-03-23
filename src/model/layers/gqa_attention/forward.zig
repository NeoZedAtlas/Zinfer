const attention = @import("../../../kernel/attention/attention.zig");
const spec_mod = @import("spec.zig");

pub fn applyRoPEToProjectedHeadsInPlace(
    spec: spec_mod.AttentionSpec,
    projected_query: []f32,
    projected_key: []f32,
    position: usize,
) !void {
    try spec.validate();
    if (projected_query.len != spec.num_attention_heads * spec.head_dim) return error.SizeMismatch;
    if (projected_key.len != spec.num_key_value_heads * spec.head_dim) return error.SizeMismatch;

    try attention.applyRoPEToHeadsInPlace(
        projected_query,
        spec.num_attention_heads,
        spec.head_dim,
        position,
        spec.rope_theta,
    );
    try attention.applyRoPEToHeadsInPlace(
        projected_key,
        spec.num_key_value_heads,
        spec.head_dim,
        position,
        spec.rope_theta,
    );
}

pub fn applyRoPEToProjectedHeadsWithTableInPlace(
    spec: spec_mod.AttentionSpec,
    projected_query: []f32,
    projected_key: []f32,
    table: *const attention.RoPETable,
    position: usize,
) !void {
    try spec.validate();
    if (projected_query.len != spec.num_attention_heads * spec.head_dim) return error.SizeMismatch;
    if (projected_key.len != spec.num_key_value_heads * spec.head_dim) return error.SizeMismatch;

    try attention.applyRoPEToHeadsWithTableInPlace(
        projected_query,
        spec.num_attention_heads,
        spec.head_dim,
        table,
        position,
    );
    try attention.applyRoPEToHeadsWithTableInPlace(
        projected_key,
        spec.num_key_value_heads,
        spec.head_dim,
        table,
        position,
    );
}

pub fn forwardProjectedSingleToken(
    spec: spec_mod.AttentionSpec,
    output: []f32,
    projected_query: []const f32,
    key_cache: []const f32,
    value_cache: []const f32,
    seq_len: usize,
    scores_scratch: []f32,
) !void {
    try spec.validate();
    try attention.scaledDotProductAttentionSingleQuery(
        output,
        projected_query,
        key_cache,
        value_cache,
        seq_len,
        spec.num_attention_heads,
        spec.num_key_value_heads,
        spec.head_dim,
        scores_scratch,
    );
}

pub fn forwardProjectedSingleTokenBf16Cache(
    spec: spec_mod.AttentionSpec,
    output: []f32,
    projected_query: []const f32,
    key_cache: []const u16,
    value_cache: []const u16,
    seq_len: usize,
    scores_scratch: []f32,
) !void {
    try spec.validate();
    try attention.scaledDotProductAttentionSingleQueryBf16Cache(
        output,
        projected_query,
        key_cache,
        value_cache,
        seq_len,
        spec.num_attention_heads,
        spec.num_key_value_heads,
        spec.head_dim,
        scores_scratch,
    );
}

pub fn forwardProjectedSingleTokenQ8Cache(
    spec: spec_mod.AttentionSpec,
    output: []f32,
    projected_query: []const f32,
    key_cache: []const i8,
    key_scales: []const u16,
    value_cache: []const i8,
    value_scales: []const u16,
    seq_len: usize,
    scores_scratch: []f32,
) !void {
    try spec.validate();
    try attention.scaledDotProductAttentionSingleQueryQ8Cache(
        output,
        projected_query,
        key_cache,
        key_scales,
        value_cache,
        value_scales,
        seq_len,
        spec.num_attention_heads,
        spec.num_key_value_heads,
        spec.head_dim,
        scores_scratch,
    );
}

pub fn forwardProjectedSingleTokenQ8CacheHeadMajor(
    spec: spec_mod.AttentionSpec,
    output: []f32,
    projected_query: []const f32,
    key_cache: []const i8,
    key_scales: []const u16,
    value_cache: []const i8,
    value_scales: []const u16,
    data_head_stride: usize,
    scale_head_stride: usize,
    seq_len: usize,
    scores_scratch: []f32,
) !void {
    try spec.validate();
    try attention.scaledDotProductAttentionSingleQueryQ8CacheHeadMajor(
        output,
        projected_query,
        key_cache,
        key_scales,
        value_cache,
        value_scales,
        data_head_stride,
        scale_head_stride,
        seq_len,
        spec.num_attention_heads,
        spec.num_key_value_heads,
        spec.head_dim,
        scores_scratch,
    );
}
