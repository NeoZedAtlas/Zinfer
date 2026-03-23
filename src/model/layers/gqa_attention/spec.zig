const std = @import("std");

pub const AttentionSpec = struct {
    hidden_size: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    rope_theta: f32,

    pub fn validate(self: AttentionSpec) !void {
        if (self.hidden_size == 0) return error.InvalidHiddenSize;
        if (self.num_attention_heads == 0) return error.InvalidAttentionHeads;
        if (self.num_key_value_heads == 0) return error.InvalidKeyValueHeads;
        if (self.head_dim == 0) return error.InvalidHeadDim;
        if (self.num_attention_heads % self.num_key_value_heads != 0) return error.InvalidGrouping;
        if (self.hidden_size != self.num_attention_heads * self.head_dim) return error.HiddenSizeMismatch;
    }

    pub fn queryGroupSize(self: AttentionSpec) usize {
        return self.num_attention_heads / self.num_key_value_heads;
    }

    pub fn kvHeadForQueryHead(self: AttentionSpec, q_head_idx: usize) usize {
        return q_head_idx / self.queryGroupSize();
    }
};

test "gqa attention spec validates grouping and dimensions" {
    const testing = std.testing;

    const spec = AttentionSpec{
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
