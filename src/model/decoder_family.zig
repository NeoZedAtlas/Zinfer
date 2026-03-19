const std = @import("std");
const tensor_store = @import("../tensor/store.zig");
const adapter_config = @import("adapters/qwen3/config.zig");
const adapter_runtime = @import("adapters/qwen3/runtime.zig");
const adapter_tokenizer = @import("adapters/qwen3/tokenizer.zig");
const adapter_chat_template = @import("adapters/qwen3/chat_template.zig");

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
    top_k_logits_alloc: *const fn (std.mem.Allocator, []const f32, usize) anyerror![]TopLogit,
    arg_max_logit: *const fn ([]const f32) anyerror!usize,
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
            .top_k_logits_alloc = adapter_runtime.topKLogitsAlloc,
            .arg_max_logit = adapter_runtime.argMaxLogit,
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
