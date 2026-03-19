const std = @import("std");
const decoder_family = @import("decoder_family.zig");
const qwen3_config = @import("qwen3_config.zig");
const qwen3_model = @import("qwen3_model.zig");
const tensor_store = @import("../tensor/store.zig");

pub const Architecture = decoder_family.Architecture;

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

    pub fn fromQwen3(cfg: qwen3_config.Qwen3Config) DecoderConfig {
        return .{
            .architecture = .qwen3,
            .model_type = cfg.model_type,
            .hidden_size = cfg.hidden_size,
            .intermediate_size = cfg.intermediate_size,
            .num_hidden_layers = cfg.num_hidden_layers,
            .num_attention_heads = cfg.num_attention_heads,
            .num_key_value_heads = cfg.num_key_value_heads,
            .head_dim = cfg.head_dim,
            .vocab_size = cfg.vocab_size,
            .max_position_embeddings = cfg.max_position_embeddings,
            .rope_theta = cfg.rope_theta,
            .rms_norm_eps = cfg.rms_norm_eps,
            .torch_dtype = cfg.torch_dtype,
            .tie_word_embeddings = cfg.tie_word_embeddings,
        };
    }

    pub fn toQwen3(self: DecoderConfig) qwen3_config.Qwen3Config {
        return .{
            .architectures = &.{},
            .head_dim = self.head_dim,
            .hidden_size = self.hidden_size,
            .intermediate_size = self.intermediate_size,
            .max_position_embeddings = self.max_position_embeddings,
            .model_type = self.model_type,
            .num_attention_heads = self.num_attention_heads,
            .num_hidden_layers = self.num_hidden_layers,
            .num_key_value_heads = self.num_key_value_heads,
            .rms_norm_eps = self.rms_norm_eps,
            .rope_theta = self.rope_theta,
            .tie_word_embeddings = self.tie_word_embeddings,
            .torch_dtype = self.torch_dtype,
            .vocab_size = self.vocab_size,
        };
    }
};

pub const ParsedConfig = struct {
    arena: std.heap.ArenaAllocator,
    value: DecoderConfig,

    pub fn deinit(self: *ParsedConfig) void {
        self.arena.deinit();
    }
};

pub const ModelCache = struct {
    architecture: Architecture,
    qwen3: qwen3_model.ModelCache,

    pub fn init(
        allocator: std.mem.Allocator,
        cfg: DecoderConfig,
        max_seq_len: usize,
    ) !ModelCache {
        return switch (cfg.architecture) {
            .qwen3 => .{
                .architecture = .qwen3,
                .qwen3 = try qwen3_model.ModelCache.init(
                    allocator,
                    cfg.num_hidden_layers,
                    max_seq_len,
                    cfg.num_key_value_heads,
                    cfg.head_dim,
                ),
            },
        };
    }

    pub fn deinit(self: *ModelCache) void {
        switch (self.architecture) {
            .qwen3 => self.qwen3.deinit(),
        }
    }
};

pub const TopLogit = qwen3_model.TopLogit;

pub fn loadConfigFromFile(backing_allocator: std.mem.Allocator, path: []const u8) !ParsedConfig {
    var qwen3 = try qwen3_config.loadFromFile(backing_allocator, path);
    errdefer qwen3.deinit();

    const architecture = decoder_family.detectArchitecture(qwen3.value.model_type) orelse return error.UnsupportedModelType;
    if (architecture != .qwen3) return error.UnsupportedModelType;

    return .{
        .arena = qwen3.arena,
        .value = DecoderConfig.fromQwen3(qwen3.value),
    };
}

pub fn forwardTokenId(
    allocator: std.mem.Allocator,
    store: *const tensor_store.TensorStore,
    cfg: DecoderConfig,
    cache: *ModelCache,
    token_id: usize,
) ![]f32 {
    return switch (cfg.architecture) {
        .qwen3 => qwen3_model.forwardTokenId(allocator, store, cfg.toQwen3(), &cache.qwen3, token_id),
    };
}

pub fn topKLogitsAlloc(
    allocator: std.mem.Allocator,
    logits: []const f32,
    k: usize,
) ![]TopLogit {
    return qwen3_model.topKLogitsAlloc(allocator, logits, k);
}

pub fn argMaxLogit(logits: []const f32) !usize {
    return qwen3_model.argMaxLogit(logits);
}

test "decoder config loads qwen3 architecture" {
    const testing = std.testing;

    var parsed = try loadConfigFromFile(testing.allocator, "models/Qwen3-0.6B/config.json");
    defer parsed.deinit();

    try testing.expectEqual(Architecture.qwen3, parsed.value.architecture);
    try testing.expectEqualStrings("qwen3", parsed.value.model_type);
}
