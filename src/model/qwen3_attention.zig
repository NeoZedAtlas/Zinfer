const std = @import("std");
const attention = @import("../kernel/attention.zig");

pub const Qwen3AttentionSpec = struct {
    hidden_size: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    rope_theta: f32,

    pub fn validate(self: Qwen3AttentionSpec) !void {
        if (self.hidden_size == 0) return error.InvalidHiddenSize;
        if (self.num_attention_heads == 0) return error.InvalidAttentionHeads;
        if (self.num_key_value_heads == 0) return error.InvalidKeyValueHeads;
        if (self.head_dim == 0) return error.InvalidHeadDim;
        if (self.num_attention_heads % self.num_key_value_heads != 0) {
            return error.InvalidGrouping;
        }
        if (self.hidden_size != self.num_attention_heads * self.head_dim) {
            return error.HiddenSizeMismatch;
        }
    }

    pub fn queryGroupSize(self: Qwen3AttentionSpec) usize {
        return self.num_attention_heads / self.num_key_value_heads;
    }

    pub fn kvHeadForQueryHead(self: Qwen3AttentionSpec, q_head_idx: usize) usize {
        return q_head_idx / self.queryGroupSize();
    }
};

pub fn applyRoPEToProjectedHeadsInPlace(
    spec: Qwen3AttentionSpec,
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

pub fn forwardProjectedSingleToken(
    spec: Qwen3AttentionSpec,
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

test "qwen3 attention spec validates grouping and dimensions" {
    const testing = std.testing;

    const spec = Qwen3AttentionSpec{
        .hidden_size = 8,
        .num_attention_heads = 4,
        .num_key_value_heads = 2,
        .head_dim = 2,
        .rope_theta = 1000000.0,
    };

    try spec.validate();
    try testing.expectEqual(@as(usize, 2), spec.queryGroupSize());
    try testing.expectEqual(@as(usize, 0), spec.kvHeadForQueryHead(0));
    try testing.expectEqual(@as(usize, 0), spec.kvHeadForQueryHead(1));
    try testing.expectEqual(@as(usize, 1), spec.kvHeadForQueryHead(2));
    try testing.expectEqual(@as(usize, 1), spec.kvHeadForQueryHead(3));
}
