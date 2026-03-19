const std = @import("std");

pub const Architecture = enum {
    qwen3,

    pub fn name(self: Architecture) []const u8 {
        return switch (self) {
            .qwen3 => "qwen3",
        };
    }
};

pub const DecoderConfig = struct {
    architecture: Architecture,
    model_type: []const u8,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    vocab_size: usize,
    max_position_embeddings: usize,
    rope_theta: f64,
    rms_norm_eps: f64,
    torch_dtype: []const u8,
    tie_word_embeddings: bool,
};

pub const ParsedConfig = struct {
    arena: std.heap.ArenaAllocator,
    value: DecoderConfig,

    pub fn deinit(self: *ParsedConfig) void {
        self.arena.deinit();
    }
};
