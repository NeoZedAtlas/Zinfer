const std = @import("std");
const tensor_store = @import("../tensor/store.zig");
const weights_layout = @import("weights_layout.zig");
const adapter_config = @import("adapters/qwen3/config.zig");
const adapter_generation_policy = @import("adapters/qwen3/generation_policy.zig");
const adapter_runtime = @import("adapters/qwen3/runtime.zig");
const adapter_spec = @import("adapters/qwen3/spec.zig");
const adapter_tokenizer = @import("adapters/qwen3/tokenizer.zig");
const adapter_chat_template = @import("adapters/qwen3/chat_template.zig");
const adapter_weights = @import("adapters/qwen3/weights.zig");

pub const Architecture = enum {
    qwen3,

    pub fn name(self: Architecture) []const u8 {
        return entryForArchitecture(self).model_type;
    }
};

pub const ThinkingMode = adapter_chat_template.ThinkingMode;
pub const Role = adapter_chat_template.Role;
pub const ToolCall = adapter_chat_template.ToolCall;
pub const Message = adapter_chat_template.Message;
pub const TopLogit = adapter_runtime.TopLogit;
pub const CommonWeights = weights_layout.CommonWeights;
pub const LayerTensorKind = weights_layout.LayerTensorKind;

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

pub const ModelCache = struct {
    architecture: Architecture,
    qwen3: adapter_runtime.ModelCache,

    pub fn init(
        allocator: std.mem.Allocator,
        cfg: DecoderConfig,
        max_seq_len: usize,
    ) !ModelCache {
        return try entryForArchitecture(cfg.architecture).init_model_cache(allocator, cfg, max_seq_len);
    }

    pub fn deinit(self: *ModelCache) void {
        switch (self.architecture) {
            .qwen3 => self.qwen3.deinit(),
        }
    }
};

pub const Tokenizer = union(Architecture) {
    qwen3: adapter_tokenizer.Tokenizer,

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

const Entry = struct {
    model_type: []const u8,
    load_config_from_file: *const fn (std.mem.Allocator, []const u8) anyerror!ParsedConfig,
    init_model_cache: *const fn (std.mem.Allocator, DecoderConfig, usize) anyerror!ModelCache,
    forward_token_id: *const fn (std.mem.Allocator, *const tensor_store.TensorStore, DecoderConfig, *ModelCache, usize) anyerror![]f32,
    prefill_token_ids: *const fn (std.mem.Allocator, *const tensor_store.TensorStore, DecoderConfig, *ModelCache, []const usize) anyerror![]f32,
    top_k_logits_alloc: *const fn (std.mem.Allocator, []const f32, usize) anyerror![]TopLogit,
    arg_max_logit: *const fn ([]const f32) anyerror!usize,
    eos_token_ids: []const u32,
    default_stop_sequences: []const []const u8,
    common_weights: CommonWeights,
    layer_tensor_name_alloc: *const fn (std.mem.Allocator, usize, LayerTensorKind) anyerror![]u8,
    load_tokenizer: *const fn (std.mem.Allocator, []const u8) anyerror!Tokenizer,
    render_messages_prompt_alloc: *const fn (std.mem.Allocator, []const Message, ThinkingMode) anyerror![]u8,
    render_single_user_prompt_alloc: *const fn (std.mem.Allocator, []const u8, ThinkingMode) anyerror![]u8,
    assistant_history_content: *const fn ([]const u8) []const u8,
};

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
    return try entryForArchitecture(cfg.architecture).forward_token_id(allocator, store, cfg, cache, token_id);
}

pub fn prefillTokenIds(
    allocator: std.mem.Allocator,
    store: *const tensor_store.TensorStore,
    cfg: DecoderConfig,
    cache: *ModelCache,
    token_ids: []const usize,
) ![]f32 {
    return try entryForArchitecture(cfg.architecture).prefill_token_ids(allocator, store, cfg, cache, token_ids);
}

pub fn topKLogitsAlloc(
    allocator: std.mem.Allocator,
    architecture: Architecture,
    logits: []const f32,
    k: usize,
) ![]TopLogit {
    return try entryForArchitecture(architecture).top_k_logits_alloc(allocator, logits, k);
}

pub fn argMaxLogit(
    architecture: Architecture,
    logits: []const f32,
) !usize {
    return try entryForArchitecture(architecture).arg_max_logit(logits);
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
            .model_type = "qwen3",
            .load_config_from_file = loadQwen3ConfigFromFile,
            .init_model_cache = initQwen3ModelCache,
            .forward_token_id = forwardQwen3TokenId,
            .prefill_token_ids = prefillQwen3TokenIds,
            .top_k_logits_alloc = adapter_runtime.topKLogitsAlloc,
            .arg_max_logit = adapter_runtime.argMaxLogit,
            .eos_token_ids = adapter_generation_policy.eos_token_ids,
            .default_stop_sequences = adapter_generation_policy.default_stop_sequences,
            .common_weights = adapter_weights.common_weights,
            .layer_tensor_name_alloc = adapter_weights.layerTensorNameAlloc,
            .load_tokenizer = loadQwen3TokenizerFromModelDir,
            .render_messages_prompt_alloc = adapter_chat_template.renderMessagesPromptAlloc,
            .render_single_user_prompt_alloc = adapter_chat_template.renderSingleUserPromptAlloc,
            .assistant_history_content = adapter_chat_template.assistantHistoryContent,
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

fn loadQwen3ConfigFromFile(backing_allocator: std.mem.Allocator, path: []const u8) !ParsedConfig {
    var parsed = try adapter_config.loadFromFile(backing_allocator, path);
    errdefer parsed.deinit();

    return .{
        .arena = parsed.arena,
        .value = .{
            .architecture = .qwen3,
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

fn qwen3ConfigFromDecoder(cfg: DecoderConfig) adapter_config.Config {
    return .{
        .architectures = &.{},
        .head_dim = cfg.head_dim,
        .hidden_size = cfg.hidden_size,
        .intermediate_size = cfg.intermediate_size,
        .max_position_embeddings = cfg.max_position_embeddings,
        .model_type = cfg.model_type,
        .num_attention_heads = cfg.num_attention_heads,
        .num_hidden_layers = cfg.num_hidden_layers,
        .num_key_value_heads = cfg.num_key_value_heads,
        .rms_norm_eps = cfg.rms_norm_eps,
        .rope_theta = cfg.rope_theta,
        .tie_word_embeddings = cfg.tie_word_embeddings,
        .torch_dtype = cfg.torch_dtype,
        .vocab_size = cfg.vocab_size,
    };
}

fn initQwen3ModelCache(
    allocator: std.mem.Allocator,
    cfg: DecoderConfig,
    max_seq_len: usize,
) !ModelCache {
    return .{
        .architecture = .qwen3,
        .qwen3 = try adapter_runtime.ModelCache.init(
            allocator,
            cfg.num_hidden_layers,
            max_seq_len,
            cfg.num_key_value_heads,
            cfg.head_dim,
        ),
    };
}

fn forwardQwen3TokenId(
    allocator: std.mem.Allocator,
    store: *const tensor_store.TensorStore,
    cfg: DecoderConfig,
    cache: *ModelCache,
    token_id: usize,
) ![]f32 {
    if (cache.architecture != .qwen3) return error.ModelCacheArchitectureMismatch;
    return try adapter_runtime.forwardTokenId(
        allocator,
        store,
        qwen3ConfigFromDecoder(cfg),
        &cache.qwen3,
        token_id,
    );
}

fn prefillQwen3TokenIds(
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
        const logits = try forwardQwen3TokenId(allocator, store, cfg, cache, token_id);
        if (last_logits) |buffer| allocator.free(buffer);
        last_logits = logits;
    }

    return last_logits orelse return error.MissingPromptLogits;
}

fn loadQwen3TokenizerFromModelDir(
    backing_allocator: std.mem.Allocator,
    model_dir: []const u8,
) !Tokenizer {
    return .{
        .qwen3 = try adapter_tokenizer.Tokenizer.loadFromModelDir(backing_allocator, model_dir),
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
