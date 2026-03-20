const std = @import("std");
const tensor_store = @import("../../tensor/storage/store.zig");
const kv_cache = @import("kv_cache.zig");
const decoder_cache = @import("decoder_cache.zig");
const decoder_only_stack = @import("../layers/decoder_only_stack.zig");
const chat_types = @import("chat_types.zig");
const decoder_registry = @import("decoder_registry.zig");
const decoder_types = @import("decoder_types.zig");
const generic_block = @import("../layers/rmsnorm_gqa_swiglu_block.zig");
const logits_util = @import("../layers/logits.zig");
const weights_layout = @import("../layers/weights_layout.zig");
const qwen3_family = @import("../families/qwen3/family.zig");

pub const Architecture = decoder_types.Architecture;
pub const ThinkingMode = chat_types.ThinkingMode;
pub const Role = chat_types.Role;
pub const ToolCall = chat_types.ToolCall;
pub const Message = chat_types.Message;
pub const TopLogit = logits_util.TopLogit;
pub const CommonWeights = weights_layout.CommonWeights;
pub const LayerTensorKind = weights_layout.LayerTensorKind;
pub const ModelCache = decoder_cache.ModelCache;
pub const DecoderConfig = decoder_types.DecoderConfig;
pub const ParsedConfig = decoder_types.ParsedConfig;

pub const Tokenizer = union(Architecture) {
    qwen3: qwen3_family.TokenizerImpl,

    pub fn loadFromModelDir(
        backing_allocator: std.mem.Allocator,
        architecture: Architecture,
        model_dir: []const u8,
    ) !Tokenizer {
        return try entryForArchitecture(architecture).load_tokenizer(backing_allocator, model_dir);
    }

    pub fn deinit(self: *Tokenizer) void {
        switch (self.*) {
            inline else => |*tokenizer| tokenizer.deinit(),
        }
    }

    pub fn encodeAlloc(self: *const Tokenizer, allocator: std.mem.Allocator, text: []const u8) ![]u32 {
        return switch (self.*) {
            inline else => |*tokenizer| tokenizer.encodeAlloc(allocator, text),
        };
    }

    pub fn decodeAlloc(self: *const Tokenizer, allocator: std.mem.Allocator, ids: []const u32) ![]u8 {
        return switch (self.*) {
            inline else => |*tokenizer| tokenizer.decodeAlloc(allocator, ids),
        };
    }
};

const Entry = decoder_registry.Entry(Tokenizer);

pub fn detectArchitecture(model_type: []const u8) ?Architecture {
    inline for (std.meta.tags(Architecture)) |tag| {
        if (std.mem.eql(u8, model_type, entryForArchitecture(tag).model_type)) return tag;
    }
    return null;
}

pub fn loadConfigFromFile(backing_allocator: std.mem.Allocator, path: []const u8) !ParsedConfig {
    const architecture = try detectArchitectureFromConfigFile(backing_allocator, path);
    return try entryForArchitecture(architecture).load_config_from_file(backing_allocator, path);
}

pub fn loadTokenizerFromModelDir(
    backing_allocator: std.mem.Allocator,
    architecture: Architecture,
    model_dir: []const u8,
) !Tokenizer {
    return Tokenizer.loadFromModelDir(backing_allocator, architecture, model_dir);
}

pub fn forwardTokenId(
    allocator: std.mem.Allocator,
    store: *const tensor_store.TensorStore,
    cfg: DecoderConfig,
    cache: *ModelCache,
    token_id: usize,
) ![]f32 {
    return try decoder_only_stack.forwardTokenId(
        allocator,
        store,
        decoder_only_stack.Config.fromDecoderConfig(cfg),
        commonWeights(cfg.architecture),
        entryForArchitecture(cfg.architecture).layer_layout,
        entryForArchitecture(cfg.architecture).layer_tensor_name_alloc,
        cache,
        token_id,
    );
}

pub fn prefillTokenIds(
    allocator: std.mem.Allocator,
    store: *const tensor_store.TensorStore,
    cfg: DecoderConfig,
    cache: *ModelCache,
    token_ids: []const usize,
) ![]f32 {
    if (token_ids.len == 0) return error.EmptyPrompt;

    var last_logits: ?[]f32 = null;
    errdefer if (last_logits) |buffer| allocator.free(buffer);

    for (token_ids) |token_id| {
        const token_logits = try forwardTokenId(allocator, store, cfg, cache, token_id);
        if (last_logits) |buffer| allocator.free(buffer);
        last_logits = token_logits;
    }

    return last_logits orelse return error.MissingPromptLogits;
}

pub fn forwardSingleBlock(
    allocator: std.mem.Allocator,
    store: *const tensor_store.TensorStore,
    cfg: DecoderConfig,
    layer_index: usize,
    cache: *kv_cache.LayerKVCache,
    hidden_in: []const f32,
    hidden_out: []f32,
) !void {
    return try generic_block.forwardSingleToken(
        allocator,
        store,
        decoder_only_stack.Config.fromDecoderConfig(cfg).blockSpec(layer_index),
        entryForArchitecture(cfg.architecture).layer_layout,
        entryForArchitecture(cfg.architecture).layer_tensor_name_alloc,
        cache,
        hidden_in,
        hidden_out,
    );
}

pub fn topKLogitsAlloc(
    allocator: std.mem.Allocator,
    values: []const f32,
    k: usize,
) ![]TopLogit {
    return try logits_util.topKLogitsAlloc(allocator, values, k);
}

pub fn argMaxLogit(
    values: []const f32,
) !usize {
    return try logits_util.argMaxLogit(values);
}

pub fn eosTokenIds(architecture: Architecture) []const u32 {
    return entryForArchitecture(architecture).eos_token_ids;
}

pub fn isEosToken(architecture: Architecture, token_id: usize) bool {
    for (eosTokenIds(architecture)) |eos_id| {
        if (token_id == eos_id) return true;
    }
    return false;
}

pub fn defaultStopSequences(architecture: Architecture) []const []const u8 {
    return entryForArchitecture(architecture).default_stop_sequences;
}

pub fn effectiveStopSequencesAlloc(
    allocator: std.mem.Allocator,
    architecture: Architecture,
    extra_stop_sequences: [][]const u8,
) ![][]const u8 {
    const defaults = defaultStopSequences(architecture);
    var unique_extra_count: usize = 0;

    for (extra_stop_sequences, 0..) |stop_sequence, idx| {
        if (containsStopSequence(defaults, stop_sequence)) continue;
        if (containsStopSequence(extra_stop_sequences[0..idx], stop_sequence)) continue;
        unique_extra_count += 1;
    }

    const combined = try allocator.alloc([]const u8, defaults.len + unique_extra_count);
    var count: usize = 0;

    for (defaults) |stop_sequence| {
        combined[count] = stop_sequence;
        count += 1;
    }

    for (extra_stop_sequences, 0..) |stop_sequence, idx| {
        if (containsStopSequence(defaults, stop_sequence)) continue;
        if (containsStopSequence(extra_stop_sequences[0..idx], stop_sequence)) continue;
        combined[count] = stop_sequence;
        count += 1;
    }

    return combined;
}

fn containsStopSequence(haystack: []const []const u8, needle: []const u8) bool {
    for (haystack) |existing| {
        if (std.mem.eql(u8, existing, needle)) return true;
    }
    return false;
}

pub fn commonWeights(architecture: Architecture) CommonWeights {
    return entryForArchitecture(architecture).common_weights;
}

pub fn layerLayout(architecture: Architecture) generic_block.LayerLayout {
    return entryForArchitecture(architecture).layer_layout;
}

pub fn layerTensorNameAlloc(
    allocator: std.mem.Allocator,
    architecture: Architecture,
    layer_index: usize,
    kind: LayerTensorKind,
) ![]u8 {
    return try entryForArchitecture(architecture).layer_tensor_name_alloc(allocator, layer_index, kind);
}

pub fn renderMessagesPromptAlloc(
    allocator: std.mem.Allocator,
    architecture: Architecture,
    messages: []const Message,
    mode: ThinkingMode,
) ![]u8 {
    return try entryForArchitecture(architecture).render_messages_prompt_alloc(allocator, messages, mode);
}

pub fn renderSingleUserPromptAlloc(
    allocator: std.mem.Allocator,
    architecture: Architecture,
    user_text: []const u8,
    mode: ThinkingMode,
) ![]u8 {
    return try entryForArchitecture(architecture).render_single_user_prompt_alloc(allocator, user_text, mode);
}

pub fn assistantHistoryContent(
    architecture: Architecture,
    content: []const u8,
) []const u8 {
    return entryForArchitecture(architecture).assistant_history_content(content);
}

fn entryForArchitecture(architecture: Architecture) Entry {
    return switch (architecture) {
        .qwen3 => .{
            .model_type = qwen3_family.model_type,
            .load_config_from_file = qwen3_family.loadParsedConfig,
            .layer_layout = qwen3_family.layer_layout,
            .eos_token_ids = qwen3_family.eos_token_ids,
            .default_stop_sequences = qwen3_family.default_stop_sequences,
            .common_weights = qwen3_family.common_weights,
            .layer_tensor_name_alloc = qwen3_family.layerTensorNameAlloc,
            .load_tokenizer = loadQwen3TokenizerFromModelDir,
            .render_messages_prompt_alloc = qwen3_family.renderMessagesPromptAlloc,
            .render_single_user_prompt_alloc = qwen3_family.renderSingleUserPromptAlloc,
            .assistant_history_content = qwen3_family.assistantHistoryContent,
        },
    };
}

fn detectArchitectureFromConfigFile(
    backing_allocator: std.mem.Allocator,
    path: []const u8,
) !Architecture {
    var arena = std.heap.ArenaAllocator.init(backing_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const bytes = try readFileAllocAtPath(allocator, path, 1024 * 1024);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, bytes, .{});
    const model_type_value = parsed.value.object.get("model_type") orelse return error.MissingModelType;
    if (model_type_value != .string) return error.InvalidModelType;

    return detectArchitecture(model_type_value.string) orelse error.UnsupportedModelType;
}

fn readFileAllocAtPath(
    allocator: std.mem.Allocator,
    path: []const u8,
    max_bytes: usize,
) ![]u8 {
    if (std.fs.path.isAbsolute(path)) {
        const file = try std.fs.openFileAbsolute(path, .{});
        defer file.close();
        return file.readToEndAlloc(allocator, max_bytes);
    }
    return std.fs.cwd().readFileAlloc(allocator, path, max_bytes);
}

fn loadQwen3TokenizerFromModelDir(
    backing_allocator: std.mem.Allocator,
    model_dir: []const u8,
) !Tokenizer {
    return .{
        .qwen3 = try qwen3_family.loadTokenizerFromModelDir(backing_allocator, model_dir),
    };
}

test "family detects qwen3 model type" {
    const testing = std.testing;
    try testing.expectEqual(Architecture.qwen3, detectArchitecture("qwen3").?);
    try testing.expect(detectArchitecture("unknown-model") == null);
}

test "family loads qwen3 config through registry" {
    const testing = std.testing;

    var parsed = try loadConfigFromFile(testing.allocator, "models/Qwen3-0.6B/config.json");
    defer parsed.deinit();

    try testing.expectEqual(Architecture.qwen3, parsed.value.architecture);
    try testing.expectEqualStrings("qwen3", parsed.value.model_type);
}

test "family tokenizer loads qwen3 and roundtrips prompt text" {
    const testing = std.testing;

    var tokenizer = try loadTokenizerFromModelDir(testing.allocator, .qwen3, "models/Qwen3-0.6B");
    defer tokenizer.deinit();

    const ids = try tokenizer.encodeAlloc(testing.allocator, "<|im_start|>user\nHello<|im_end|>\n");
    defer testing.allocator.free(ids);
    try testing.expectEqualSlices(u32, &[_]u32{ 151644, 872, 198, 9707, 151645, 198 }, ids);

    const text = try tokenizer.decodeAlloc(testing.allocator, ids);
    defer testing.allocator.free(text);
    try testing.expectEqualStrings("<|im_start|>user\nHello<|im_end|>\n", text);
}

test "family exposes qwen3 weight naming policy" {
    const testing = std.testing;

    const common = commonWeights(.qwen3);
    try testing.expectEqualStrings("model.embed_tokens.weight", common.embed_tokens_weight);
    try testing.expectEqualStrings("model.norm.weight", common.final_norm_weight);
    try testing.expectEqualStrings("lm_head.weight", common.lm_head_weight);

    const layer_name = try layerTensorNameAlloc(testing.allocator, .qwen3, 2, .mlp_down_proj_weight);
    defer testing.allocator.free(layer_name);
    try testing.expectEqualStrings("model.layers.2.mlp.down_proj.weight", layer_name);
}

test "family exposes qwen3 generation policy" {
    const testing = std.testing;

    try testing.expect(isEosToken(.qwen3, 151645));
    try testing.expect(isEosToken(.qwen3, 151643));
    try testing.expect(!isEosToken(.qwen3, 1));

    const stops = defaultStopSequences(.qwen3);
    try testing.expectEqual(@as(usize, 1), stops.len);
    try testing.expectEqualStrings("<|im_end|>", stops[0]);

    const merged = try effectiveStopSequencesAlloc(testing.allocator, .qwen3, @constCast(&[_][]const u8{ "</tool_response>", "<|im_end|>" }));
    defer testing.allocator.free(merged);
    try testing.expectEqual(@as(usize, 2), merged.len);
    try testing.expectEqualStrings("<|im_end|>", merged[0]);
    try testing.expectEqualStrings("</tool_response>", merged[1]);
}
