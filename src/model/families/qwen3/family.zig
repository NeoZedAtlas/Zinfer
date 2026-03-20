const std = @import("std");
const decoder_types = @import("../../runtime/decoder_types.zig");
const chat_types = @import("../../runtime/chat_types.zig");
const bpe_tokenizer = @import("../../../tokenizer/bpe.zig");
const config = @import("config.zig");
const generation_policy = @import("generation_policy.zig");
const layout = @import("layout.zig");
const chat_template = @import("chat_template.zig");
const weights = @import("weights.zig");

pub const architecture = decoder_types.Architecture.qwen3;
pub const model_type = "qwen3";

pub const TokenizerImpl = bpe_tokenizer.Tokenizer;
pub const ThinkingMode = chat_types.ThinkingMode;
pub const Role = chat_types.Role;
pub const ToolCall = chat_types.ToolCall;
pub const Message = chat_types.Message;

pub const eos_token_ids = generation_policy.eos_token_ids;
pub const default_stop_sequences = generation_policy.default_stop_sequences;
pub const common_weights = weights.common_weights;
pub const layer_layout = layout.layer_layout;
pub const layerTensorNameAlloc = weights.layerTensorNameAlloc;
pub const renderMessagesPromptAlloc = chat_template.renderMessagesPromptAlloc;
pub const renderSingleUserPromptAlloc = chat_template.renderSingleUserPromptAlloc;
pub const assistantHistoryContent = chat_template.assistantHistoryContent;

pub fn loadParsedConfig(backing_allocator: std.mem.Allocator, path: []const u8) !decoder_types.ParsedConfig {
    var parsed = try config.loadFromFile(backing_allocator, path);
    errdefer parsed.deinit();

    return .{
        .arena = parsed.arena,
        .value = .{
            .architecture = architecture,
            .model_type = parsed.value.model_type,
            .hidden_size = parsed.value.hidden_size,
            .intermediate_size = parsed.value.intermediate_size,
            .num_hidden_layers = parsed.value.num_hidden_layers,
            .num_attention_heads = parsed.value.num_attention_heads,
            .num_key_value_heads = parsed.value.num_key_value_heads,
            .head_dim = parsed.value.head_dim,
            .vocab_size = parsed.value.vocab_size,
            .max_position_embeddings = parsed.value.max_position_embeddings,
            .rope_theta = parsed.value.rope_theta,
            .rms_norm_eps = parsed.value.rms_norm_eps,
            .torch_dtype = parsed.value.torch_dtype,
            .tie_word_embeddings = parsed.value.tie_word_embeddings,
        },
    };
}

pub fn loadTokenizerFromModelDir(backing_allocator: std.mem.Allocator, model_dir: []const u8) !TokenizerImpl {
    return try bpe_tokenizer.Tokenizer.loadFromModelDir(backing_allocator, model_dir);
}

test "adapter family loads parsed config into shared decoder config" {
    const testing = std.testing;

    var parsed = try loadParsedConfig(testing.allocator, "models/Qwen3-0.6B/config.json");
    defer parsed.deinit();

    try testing.expectEqual(decoder_types.Architecture.qwen3, parsed.value.architecture);
    try testing.expectEqualStrings("qwen3", parsed.value.model_type);
}
