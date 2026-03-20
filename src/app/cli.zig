const std = @import("std");
const cli_args = @import("cli/args.zig");
const cli_inspect = @import("cli/inspect.zig");
const cli_prompts = @import("cli/prompts.zig");
const cli_tools = @import("cli/tools.zig");
const cli_usage = @import("cli/usage.zig");
const optimized_kv_cache = @import("../model/optimized_kv_cache.zig");
const decoder_family = @import("../model/decoder_family.zig");
const optimized_decoder = @import("../model/optimized_decoder.zig");
const tensor_backend = @import("../tensor/backend.zig");
const sampler = @import("../sampling/sampler.zig");

const default_model_dir = cli_args.default_model_dir;
const GenerateOptions = cli_args.GenerateOptions;
const ParsedGenerateInvocation = cli_args.ParsedGenerateInvocation;
const ParsedGenerateChatInvocation = cli_args.ParsedGenerateChatInvocation;
const ParsedChatInvocation = cli_args.ParsedChatInvocation;
const ParsedBenchInvocation = cli_args.ParsedBenchInvocation;

pub fn run(allocator: std.mem.Allocator) !void {
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len == 1) {
        try cli_inspect.inspectConfig(allocator, default_model_dir);
        return;
    }

    const command = args[1];
    if (std.mem.eql(u8, command, "inspect-config")) {
        const model_dir = if (args.len >= 3) args[2] else default_model_dir;
        try cli_inspect.inspectConfig(allocator, model_dir);
        return;
    }

    if (std.mem.eql(u8, command, "inspect-weights")) {
        const model_dir = if (args.len >= 3) args[2] else default_model_dir;
        try cli_inspect.inspectWeights(allocator, model_dir);
        return;
    }

    if (std.mem.eql(u8, command, "inspect-tensor")) {
        if (args.len == 3) {
            try cli_inspect.inspectTensor(allocator, default_model_dir, args[2], 8);
            return;
        }
        if (args.len >= 4) {
            const count = if (args.len >= 5) try std.fmt.parseInt(usize, args[4], 10) else 8;
            try cli_inspect.inspectTensor(allocator, args[2], args[3], count);
            return;
        }
        try printUsage();
        return error.InvalidCommand;
    }

    if (std.mem.eql(u8, command, "probe-linear")) {
        if (args.len == 3) {
            try cli_inspect.probeLinear(allocator, default_model_dir, args[2], 0, 8);
            return;
        }
        if (args.len == 4) {
            const input_index = try std.fmt.parseInt(usize, args[3], 10);
            try cli_inspect.probeLinear(allocator, default_model_dir, args[2], input_index, 8);
            return;
        }
        if (args.len == 5) {
            const input_index = try std.fmt.parseInt(usize, args[3], 10);
            const rows_to_print = try std.fmt.parseInt(usize, args[4], 10);
            try cli_inspect.probeLinear(allocator, default_model_dir, args[2], input_index, rows_to_print);
            return;
        }
        if (args.len >= 6) {
            const input_index = try std.fmt.parseInt(usize, args[4], 10);
            const rows_to_print = try std.fmt.parseInt(usize, args[5], 10);
            try cli_inspect.probeLinear(allocator, args[2], args[3], input_index, rows_to_print);
            return;
        }
        try printUsage();
        return error.InvalidCommand;
    }

    if (std.mem.eql(u8, command, "probe-block")) {
        if (args.len == 2) {
            try cli_inspect.probeBlock(allocator, default_model_dir, 0, 0, 8);
            return;
        }
        if (args.len == 3) {
            const layer_index = try std.fmt.parseInt(usize, args[2], 10);
            try cli_inspect.probeBlock(allocator, default_model_dir, layer_index, 0, 8);
            return;
        }
        if (args.len == 4) {
            const layer_index = try std.fmt.parseInt(usize, args[2], 10);
            const input_index = try std.fmt.parseInt(usize, args[3], 10);
            try cli_inspect.probeBlock(allocator, default_model_dir, layer_index, input_index, 8);
            return;
        }
        if (args.len == 5) {
            const layer_index = try std.fmt.parseInt(usize, args[2], 10);
            const input_index = try std.fmt.parseInt(usize, args[3], 10);
            const count = try std.fmt.parseInt(usize, args[4], 10);
            try cli_inspect.probeBlock(allocator, default_model_dir, layer_index, input_index, count);
            return;
        }
        if (args.len >= 6) {
            const layer_index = try std.fmt.parseInt(usize, args[3], 10);
            const input_index = try std.fmt.parseInt(usize, args[4], 10);
            const count = try std.fmt.parseInt(usize, args[5], 10);
            try cli_inspect.probeBlock(allocator, args[2], layer_index, input_index, count);
            return;
        }
        try printUsage();
        return error.InvalidCommand;
    }

    if (std.mem.eql(u8, command, "probe-model")) {
        if (args.len == 2) {
            try cli_inspect.probeModel(allocator, default_model_dir, 0, 8);
            return;
        }
        if (args.len == 3) {
            const token_id = try std.fmt.parseInt(usize, args[2], 10);
            try cli_inspect.probeModel(allocator, default_model_dir, token_id, 8);
            return;
        }
        if (args.len == 4) {
            const token_id = try std.fmt.parseInt(usize, args[2], 10);
            const top_k = try std.fmt.parseInt(usize, args[3], 10);
            try cli_inspect.probeModel(allocator, default_model_dir, token_id, top_k);
            return;
        }
        if (args.len >= 5) {
            const token_id = try std.fmt.parseInt(usize, args[3], 10);
            const top_k = try std.fmt.parseInt(usize, args[4], 10);
            try cli_inspect.probeModel(allocator, args[2], token_id, top_k);
            return;
        }
        try printUsage();
        return error.InvalidCommand;
    }

    if (std.mem.eql(u8, command, "generate-token-ids")) {
        if (args.len == 2) {
            try cli_inspect.generateTokenIds(allocator, default_model_dir, "0", 5);
            return;
        }
        if (args.len == 3) {
            try cli_inspect.generateTokenIds(allocator, default_model_dir, args[2], 5);
            return;
        }
        if (args.len == 4) {
            const steps = try std.fmt.parseInt(usize, args[3], 10);
            try cli_inspect.generateTokenIds(allocator, default_model_dir, args[2], steps);
            return;
        }
        if (args.len >= 5) {
            const steps = try std.fmt.parseInt(usize, args[4], 10);
            try cli_inspect.generateTokenIds(allocator, args[2], args[3], steps);
            return;
        }
        try printUsage();
        return error.InvalidCommand;
    }

    if (std.mem.eql(u8, command, "bench")) {
        var invocation = try cli_args.parseBenchInvocation(allocator, args);
        defer invocation.deinit(allocator);
        try cli_tools.benchPrompt(allocator, invocation.model_dir, invocation.user_text, invocation.options);
        return;
    }

    if (std.mem.eql(u8, command, "quantize")) {
        if (args.len == 3) {
            try cli_tools.quantizeModelDir(allocator, default_model_dir, args[2]);
            return;
        }
        if (args.len >= 4) {
            try cli_tools.quantizeModelDir(allocator, args[3], args[2]);
            return;
        }
        try printUsage();
        return error.InvalidCommand;
    }

    if (std.mem.eql(u8, command, "tokenize")) {
        if (args.len == 3) {
            try cli_tools.tokenizeText(allocator, default_model_dir, args[2]);
            return;
        }
        if (args.len >= 4) {
            try cli_tools.tokenizeText(allocator, args[2], args[3]);
            return;
        }
        try printUsage();
        return error.InvalidCommand;
    }

    if (std.mem.eql(u8, command, "decode-ids")) {
        if (args.len == 3) {
            try cli_tools.decodeIds(allocator, default_model_dir, args[2]);
            return;
        }
        if (args.len >= 4) {
            try cli_tools.decodeIds(allocator, args[2], args[3]);
            return;
        }
        try printUsage();
        return error.InvalidCommand;
    }

    if (std.mem.eql(u8, command, "generate")) {
        var invocation = try cli_args.parseGenerateInvocation(allocator, args);
        defer invocation.deinit(allocator);
        try generateText(allocator, invocation.model_dir, invocation.user_text, invocation.options);
        return;
    }

    if (std.mem.eql(u8, command, "generate-chat")) {
        var invocation = try cli_args.parseGenerateChatInvocation(allocator, args);
        defer invocation.deinit(allocator);
        try generateChatFromFile(allocator, invocation.model_dir, invocation.messages_json_path, invocation.options);
        return;
    }

    if (std.mem.eql(u8, command, "chat")) {
        var invocation = try cli_args.parseChatInvocation(allocator, args);
        defer invocation.deinit(allocator);
        try chatLoop(
            allocator,
            invocation.model_dir,
            invocation.options,
            invocation.load_path,
            invocation.save_path,
        );
        return;
    }

    if (std.mem.eql(u8, command, "--help") or std.mem.eql(u8, command, "-h")) {
        try printUsage();
        return;
    }

    std.log.err("unknown command: {s}", .{command});
    try printUsage();
    return error.InvalidCommand;
}

fn printUsage() !void {
    try cli_usage.printUsage();
}


const StopAnalysis = struct {
    printable_len: usize,
    stop_hit: bool,
    response_len: usize,
};

fn analyzeGeneratedText(text: []const u8, stop_sequences: [][]const u8) StopAnalysis {
    var max_overlap: usize = 0;

    for (stop_sequences) |stop_sequence| {
        if (stop_sequence.len == 0) continue;
        if (std.mem.endsWith(u8, text, stop_sequence)) {
            return .{
                .printable_len = text.len - stop_sequence.len,
                .stop_hit = true,
                .response_len = text.len - stop_sequence.len,
            };
        }

        const max_candidate = @min(text.len, stop_sequence.len - 1);
        var overlap = max_candidate;
        while (overlap > 0) : (overlap -= 1) {
            if (std.mem.eql(u8, text[text.len - overlap ..], stop_sequence[0..overlap])) {
                max_overlap = @max(max_overlap, overlap);
                break;
            }
        }
    }

    return .{
        .printable_len = text.len - max_overlap,
        .stop_hit = false,
        .response_len = text.len,
    };
}

fn analyzeAndMaybeStream(
    allocator: std.mem.Allocator,
    tokenizer: *decoder_family.Tokenizer,
    generated_ids: []const u32,
    options: GenerateOptions,
    stdout: anytype,
    streamed_len: *usize,
) !?[]u8 {
    if (!options.stream_output and options.stop_sequences.len == 0) return null;

    const decoded = tokenizer.decodeAlloc(allocator, generated_ids) catch |err| switch (err) {
        error.InvalidWtf8 => return null,
        else => return err,
    };
    defer allocator.free(decoded);

    const analysis = analyzeGeneratedText(decoded, options.stop_sequences);
    if (options.stream_output and analysis.printable_len > streamed_len.*) {
        try stdout.writeAll(decoded[streamed_len.*..analysis.printable_len]);
        streamed_len.* = analysis.printable_len;
    }
    if (!analysis.stop_hit) return null;

    return try allocator.dupe(u8, decoded[0..analysis.response_len]);
}

fn generateText(
    allocator: std.mem.Allocator,
    model_dir: []const u8,
    user_text: []const u8,
    options: GenerateOptions,
) !void {
    var runtime = try GeneratorRuntime.init(allocator, model_dir, options.backend_scheme, options.thread_count);
    defer runtime.deinit();
    const stdout = std.fs.File.stdout().deprecatedWriter();

    try stdout.print("Zinfer generate\n", .{});
    try stdout.print("mode: {s}\n", .{cli_prompts.thinkingModeName(options.thinking_mode)});
    try stdout.print("prompt: {s}\n", .{user_text});
    try stdout.writeAll("response: ");

    const prompt = try cli_prompts.buildSingleUserPromptAlloc(
        allocator,
        runtime.model.cfg.architecture,
        user_text,
        options.system_prompt,
        options.thinking_mode,
    );
    defer allocator.free(prompt);

    const response = try runtime.generateFromPrompt(prompt, options);
    defer allocator.free(response);
    if (!options.stream_output) {
        try stdout.print("{s}", .{response});
    }
    try stdout.writeAll("\n");
}

fn generateChatFromFile(
    allocator: std.mem.Allocator,
    model_dir: []const u8,
    messages_json_path: []const u8,
    options: GenerateOptions,
) !void {
    var runtime = try GeneratorRuntime.init(allocator, model_dir, options.backend_scheme, options.thread_count);
    defer runtime.deinit();
    const stdout = std.fs.File.stdout().deprecatedWriter();

    try stdout.print("Zinfer generate-chat\n", .{});
    try stdout.print("mode: {s}\n", .{cli_prompts.thinkingModeName(options.thinking_mode)});
    try stdout.print("messages_path: {s}\n", .{messages_json_path});
    try stdout.writeAll("response: ");

    var messages = try loadChatMessages(allocator, messages_json_path);
    defer messages.deinit();

    const prompt = try cli_prompts.buildMessagesPromptAlloc(
        allocator,
        runtime.model.cfg.architecture,
        messages.items,
        options.system_prompt,
        options.thinking_mode,
    );
    defer allocator.free(prompt);

    const response = try runtime.generateFromPrompt(prompt, options);
    defer allocator.free(response);
    if (!options.stream_output) {
        try stdout.print("{s}", .{response});
    }
    try stdout.writeAll("\n");
}

const LoadedChatMessages = struct {
    arena: std.heap.ArenaAllocator,
    items: []decoder_family.Message,

    fn deinit(self: *LoadedChatMessages) void {
        self.arena.deinit();
    }
};

const SessionMetadata = struct {
    model_dir: []const u8,
    options: GenerateOptions,
};

fn loadChatMessages(
    backing_allocator: std.mem.Allocator,
    path: []const u8,
) !LoadedChatMessages {
    var arena = std.heap.ArenaAllocator.init(backing_allocator);
    errdefer arena.deinit();
    const allocator = arena.allocator();

    const bytes = try readFileAllocAtPath(allocator, path, 1024 * 1024);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, bytes, .{});
    const messages_value = switch (parsed.value) {
        .array => parsed.value,
        .object => parsed.value.object.get("messages") orelse return error.InvalidMessagesJson,
        else => return error.InvalidMessagesJson,
    };
    if (messages_value != .array) return error.InvalidMessagesJson;

    const messages = try allocator.alloc(decoder_family.Message, messages_value.array.items.len);

    for (messages_value.array.items, 0..) |item, idx| {
        if (item != .object) return error.InvalidMessagesJson;

        const role_value = item.object.get("role") orelse return error.InvalidMessagesJson;
        const content_value = item.object.get("content") orelse return error.InvalidMessagesJson;
        if (role_value != .string or content_value != .string) return error.InvalidMessagesJson;

        var tool_calls: []const decoder_family.ToolCall = &.{};
        if (item.object.get("tool_calls")) |tool_calls_value| {
            if (tool_calls_value != .array) return error.InvalidMessagesJson;
            const parsed_tool_calls = try allocator.alloc(decoder_family.ToolCall, tool_calls_value.array.items.len);
            for (tool_calls_value.array.items, 0..) |tool_call_item, tool_idx| {
                if (tool_call_item != .object) return error.InvalidMessagesJson;
                const name_value = tool_call_item.object.get("name") orelse return error.InvalidMessagesJson;
                const args_value = tool_call_item.object.get("arguments") orelse return error.InvalidMessagesJson;
                if (name_value != .string or args_value != .string) return error.InvalidMessagesJson;
                parsed_tool_calls[tool_idx] = .{
                    .name = try allocator.dupe(u8, name_value.string),
                    .arguments_json = try allocator.dupe(u8, args_value.string),
                };
            }
            tool_calls = parsed_tool_calls;
        }

        messages[idx] = .{
            .role = try parseChatRole(role_value.string),
            .content = try allocator.dupe(u8, content_value.string),
            .tool_calls = tool_calls,
        };
    }

    return .{
        .arena = arena,
        .items = messages,
    };
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

fn parseChatRole(text: []const u8) !decoder_family.Role {
    if (std.mem.eql(u8, text, "system")) return .system;
    if (std.mem.eql(u8, text, "user")) return .user;
    if (std.mem.eql(u8, text, "assistant")) return .assistant;
    if (std.mem.eql(u8, text, "tool")) return .tool;
    return error.InvalidChatRole;
}

const GeneratorRuntime = struct {
    allocator: std.mem.Allocator,
    tokenizer: decoder_family.Tokenizer,
    model: optimized_decoder.Runtime,

    fn init(
        allocator: std.mem.Allocator,
        model_dir: []const u8,
        backend_scheme: tensor_backend.Scheme,
        thread_count: usize,
    ) !GeneratorRuntime {
        const config_path = try std.fs.path.join(allocator, &.{ model_dir, "config.json" });
        defer allocator.free(config_path);
        var parsed_config = try decoder_family.loadConfigFromFile(allocator, config_path);
        defer parsed_config.deinit();

        var tokenizer = try decoder_family.loadTokenizerFromModelDir(
            allocator,
            parsed_config.value.architecture,
            model_dir,
        );
        errdefer tokenizer.deinit();

        var model = try optimized_decoder.Runtime.init(
            allocator,
            model_dir,
            backend_scheme,
            if (thread_count == 0) null else thread_count,
        );
        errdefer model.deinit();

        return .{
            .allocator = allocator,
            .tokenizer = tokenizer,
            .model = model,
        };
    }

    fn deinit(self: *GeneratorRuntime) void {
        self.model.deinit();
        self.tokenizer.deinit();
    }

    fn generateFromPrompt(
        self: *GeneratorRuntime,
        prompt: []const u8,
        options: GenerateOptions,
    ) ![]u8 {
        const prompt_ids_u32 = try self.tokenizer.encodeAlloc(self.allocator, prompt);
        defer self.allocator.free(prompt_ids_u32);
        if (prompt_ids_u32.len == 0) return error.EmptyPrompt;

        const prompt_ids = try self.allocator.alloc(usize, prompt_ids_u32.len);
        defer self.allocator.free(prompt_ids);
        for (prompt_ids_u32, 0..) |token_id, idx| {
            prompt_ids[idx] = token_id;
        }

        const cfg = self.model.cfg;
        const resolved_kv_cache_scheme = optimized_kv_cache.resolveScheme(options.kv_cache_scheme, self.model.backendName());
        var cache = try optimized_kv_cache.ModelCache.init(
            self.allocator,
            cfg.num_hidden_layers,
            prompt_ids.len + options.max_new_tokens,
            cfg.num_key_value_heads,
            cfg.head_dim,
            resolved_kv_cache_scheme,
        );
        defer cache.deinit();
        var workspace = try self.model.initWorkspace(prompt_ids.len + options.max_new_tokens);
        defer workspace.deinit();

        const effective_stop_sequences = try decoder_family.effectiveStopSequencesAlloc(
            self.allocator,
            cfg.architecture,
            options.stop_sequences,
        );
        defer self.allocator.free(effective_stop_sequences);

        var current_logits = try self.model.prefillTokenIds(&workspace, &cache, prompt_ids);

        var generated = std.ArrayListUnmanaged(u32).empty;
        defer generated.deinit(self.allocator);

        var history_ids = std.ArrayListUnmanaged(usize).empty;
        defer history_ids.deinit(self.allocator);
        try history_ids.appendSlice(self.allocator, prompt_ids);
        const stdout = std.fs.File.stdout().deprecatedWriter();
        var streamed_len: usize = 0;
        var prng = std.Random.DefaultPrng.init(options.seed);
        for (0..options.max_new_tokens) |_| {
            const next_token = try sampler.sampleToken(self.allocator, prng.random(), current_logits, history_ids.items, options.sampling);
            if (decoder_family.isEosToken(cfg.architecture, next_token)) {
                break;
            }

            try generated.append(self.allocator, std.math.cast(u32, next_token) orelse return error.TokenIdOutOfRange);
            try history_ids.append(self.allocator, next_token);
            var effective_options = options;
            effective_options.stop_sequences = effective_stop_sequences;
            if (try analyzeAndMaybeStream(self.allocator, &self.tokenizer, generated.items, effective_options, stdout, &streamed_len)) |trimmed| {
                return trimmed;
            }

            current_logits = try self.model.forwardTokenId(&workspace, &cache, next_token);
        }

        const response = try self.tokenizer.decodeAlloc(self.allocator, generated.items);
        if (options.stream_output and response.len > streamed_len) {
            try stdout.writeAll(response[streamed_len..]);
        }
        return response;
    }
};

fn chatLoop(
    allocator: std.mem.Allocator,
    model_dir: []const u8,
    options: GenerateOptions,
    load_path: ?[]const u8,
    save_path: ?[]const u8,
) !void {
    var runtime = try GeneratorRuntime.init(allocator, model_dir, options.backend_scheme, options.thread_count);
    defer runtime.deinit();

    var history = ChatHistory.init(allocator);
    defer history.deinit();

    const stdout = std.fs.File.stdout().deprecatedWriter();
    const stdin = std.fs.File.stdin();
    const reader = stdin.deprecatedReader();

    try stdout.print("Zinfer chat\n", .{});
    try stdout.print("model_dir: {s}\n", .{model_dir});
    try stdout.print("mode: {s}\n", .{cli_prompts.thinkingModeName(options.thinking_mode)});
    try stdout.writeAll("commands: /exit /quit /clear /save <path> /load <path>\n");

    if (load_path) |path| {
        try history.loadFromFile(path);
    }
    if (options.system_prompt) |system_prompt| {
        try history.prependSystemIfMissing(system_prompt);
    }

    while (true) {
        try stdout.writeAll("user> ");
        const line_opt = try reader.readUntilDelimiterOrEofAlloc(allocator, '\n', 64 * 1024);
        if (line_opt == null) break;
        defer allocator.free(line_opt.?);

        const line = std.mem.trimRight(u8, line_opt.?, "\r\n");
        if (line.len == 0) continue;

        if (std.mem.eql(u8, line, "/exit") or std.mem.eql(u8, line, "/quit")) break;
        if (std.mem.eql(u8, line, "/clear")) {
            history.clear();
            if (options.system_prompt) |system_prompt| {
                try history.prependSystemIfMissing(system_prompt);
            }
            try stdout.writeAll("history cleared\n");
            continue;
        }
        if (std.mem.startsWith(u8, line, "/save ")) {
            const path = std.mem.trim(u8, line["/save ".len..], " ");
            if (path.len == 0) return error.MissingFlagValue;
            try history.saveToFile(path, .{ .model_dir = model_dir, .options = options });
            try stdout.print("saved: {s}\n", .{path});
            continue;
        }
        if (std.mem.startsWith(u8, line, "/load ")) {
            const path = std.mem.trim(u8, line["/load ".len..], " ");
            if (path.len == 0) return error.MissingFlagValue;
            try history.loadFromFile(path);
            if (options.system_prompt) |system_prompt| {
                try history.prependSystemIfMissing(system_prompt);
            }
            try stdout.print("loaded: {s}\n", .{path});
            continue;
        }

        try history.append(.user, line);

        const prompt = try decoder_family.renderMessagesPromptAlloc(allocator, runtime.model.cfg.architecture, history.items(), options.thinking_mode);
        defer allocator.free(prompt);

        try stdout.writeAll("assistant> ");
        const response = try runtime.generateFromPrompt(prompt, options);
        defer allocator.free(response);

        if (!options.stream_output) {
            try stdout.print("{s}", .{response});
        }
        try stdout.writeAll("\n");
        try history.append(.assistant, decoder_family.assistantHistoryContent(runtime.model.cfg.architecture, response));
    }

    if (save_path) |path| {
        try history.saveToFile(path, .{ .model_dir = model_dir, .options = options });
    }
}

const ChatHistory = struct {
    allocator: std.mem.Allocator,
    messages: std.ArrayListUnmanaged(decoder_family.Message),

    fn init(allocator: std.mem.Allocator) ChatHistory {
        return .{
            .allocator = allocator,
            .messages = .empty,
        };
    }

    fn deinit(self: *ChatHistory) void {
        self.clear();
        self.messages.deinit(self.allocator);
    }

    fn clear(self: *ChatHistory) void {
        for (self.messages.items) |message| {
            self.allocator.free(message.content);
            if (message.tool_calls.len != 0) {
                for (message.tool_calls) |tool_call| {
                    self.allocator.free(tool_call.name);
                    self.allocator.free(tool_call.arguments_json);
                }
                self.allocator.free(message.tool_calls);
            }
        }
        self.messages.clearRetainingCapacity();
    }

    fn append(self: *ChatHistory, role: decoder_family.Role, content: []const u8) !void {
        try self.appendMessage(.{
            .role = role,
            .content = content,
        });
    }

    fn appendMessage(self: *ChatHistory, message: decoder_family.Message) !void {
        const owned_content = try self.allocator.dupe(u8, message.content);
        errdefer self.allocator.free(owned_content);

        var owned_tool_calls: []const decoder_family.ToolCall = &.{};
        if (message.tool_calls.len != 0) {
            const copied = try self.allocator.alloc(decoder_family.ToolCall, message.tool_calls.len);
            errdefer self.allocator.free(copied);

            var copied_len: usize = 0;
            errdefer {
                for (copied[0..copied_len]) |tool_call| {
                    self.allocator.free(tool_call.name);
                    self.allocator.free(tool_call.arguments_json);
                }
            }

            for (message.tool_calls, 0..) |tool_call, idx| {
                copied[idx] = .{
                    .name = try self.allocator.dupe(u8, tool_call.name),
                    .arguments_json = try self.allocator.dupe(u8, tool_call.arguments_json),
                };
                copied_len += 1;
            }
            owned_tool_calls = copied;
        }
        errdefer if (owned_tool_calls.len != 0) {
            for (owned_tool_calls) |tool_call| {
                self.allocator.free(tool_call.name);
                self.allocator.free(tool_call.arguments_json);
            }
            self.allocator.free(owned_tool_calls);
        };

        try self.messages.append(self.allocator, .{
            .role = message.role,
            .content = owned_content,
            .tool_calls = owned_tool_calls,
        });
    }

    fn prependSystemIfMissing(self: *ChatHistory, content: []const u8) !void {
        if (self.messages.items.len != 0 and self.messages.items[0].role == .system) return;

        const owned = try self.allocator.dupe(u8, content);
        errdefer self.allocator.free(owned);
        try self.messages.insert(self.allocator, 0, .{
            .role = .system,
            .content = owned,
        });
    }

    fn saveToFile(self: *const ChatHistory, path: []const u8, metadata: SessionMetadata) !void {
        const file = if (std.fs.path.isAbsolute(path))
            try std.fs.createFileAbsolute(path, .{ .truncate = true })
        else
            try std.fs.cwd().createFile(path, .{ .truncate = true });
        defer file.close();
        const writer = file.deprecatedWriter();

        try writer.writeAll("{\n");
        try writer.writeAll("  \"version\": 1,\n");
        try writer.writeAll("  \"kind\": \"zinfer_chat_session\",\n");
        try writer.writeAll("  \"saved_unix\": ");
        try writer.print("{d}", .{std.time.timestamp()});
        try writer.writeAll(",\n");
        try writer.writeAll("  \"model_dir\": ");
        try writer.print("{f}", .{std.json.fmt(metadata.model_dir, .{})});
        try writer.writeAll(",\n");
        try writer.writeAll("  \"options\": {\n");
        try writer.writeAll("    \"thinking_mode\": ");
        try writer.print("{f}", .{std.json.fmt(cli_prompts.thinkingModeName(metadata.options.thinking_mode), .{})});
        try writer.writeAll(",\n");
        try writer.writeAll("    \"max_new_tokens\": ");
        try writer.print("{d}", .{metadata.options.max_new_tokens});
        try writer.writeAll(",\n");
        try writer.writeAll("    \"seed\": ");
        try writer.print("{d}", .{metadata.options.seed});
        try writer.writeAll(",\n");
        try writer.writeAll("    \"backend\": ");
        try writer.print("{f}", .{std.json.fmt(metadata.options.backend_scheme.name(), .{})});
        try writer.writeAll(",\n");
        try writer.writeAll("    \"kv_cache\": ");
        try writer.print("{f}", .{std.json.fmt(metadata.options.kv_cache_scheme.name(), .{})});
        try writer.writeAll(",\n");
        try writer.writeAll("    \"threads\": ");
        try writer.print("{d}", .{metadata.options.thread_count});
        try writer.writeAll(",\n");
        try writer.writeAll("    \"stream_output\": ");
        try writer.writeAll(if (metadata.options.stream_output) "true" else "false");
        try writer.writeAll(",\n");
        try writer.writeAll("    \"system_prompt\": ");
        if (metadata.options.system_prompt) |system_prompt| {
            try writer.print("{f}", .{std.json.fmt(system_prompt, .{})});
        } else {
            try writer.writeAll("null");
        }
        try writer.writeAll(",\n");
        try writer.writeAll("    \"sampling\": {\n");
        try writer.writeAll("      \"temperature\": ");
        try writer.print("{d}", .{metadata.options.sampling.temperature});
        try writer.writeAll(",\n");
        try writer.writeAll("      \"top_p\": ");
        try writer.print("{d}", .{metadata.options.sampling.top_p});
        try writer.writeAll(",\n");
        try writer.writeAll("      \"top_k\": ");
        try writer.print("{d}", .{metadata.options.sampling.top_k});
        try writer.writeAll(",\n");
        try writer.writeAll("      \"min_p\": ");
        try writer.print("{d}", .{metadata.options.sampling.min_p});
        try writer.writeAll(",\n");
        try writer.writeAll("      \"presence_penalty\": ");
        try writer.print("{d}", .{metadata.options.sampling.presence_penalty});
        try writer.writeAll(",\n");
        try writer.writeAll("      \"frequency_penalty\": ");
        try writer.print("{d}", .{metadata.options.sampling.frequency_penalty});
        try writer.writeAll(",\n");
        try writer.writeAll("      \"repetition_penalty\": ");
        try writer.print("{d}", .{metadata.options.sampling.repetition_penalty});
        try writer.writeAll("\n");
        try writer.writeAll("    },\n");
        try writer.writeAll("    \"stop_sequences\": [");
        for (metadata.options.stop_sequences, 0..) |stop_sequence, idx| {
            if (idx != 0) try writer.writeAll(", ");
            try writer.print("{f}", .{std.json.fmt(stop_sequence, .{})});
        }
        try writer.writeAll("]\n");
        try writer.writeAll("  },\n");
        try writer.writeAll("  \"messages\": [\n");
        for (self.messages.items, 0..) |message, idx| {
            if (idx != 0) try writer.writeAll(",\n");
            try writer.writeAll("  {\"role\": ");
            try writer.print("{f}", .{std.json.fmt(message.role.name(), .{})});
            try writer.writeAll(", \"content\": ");
            try writer.print("{f}", .{std.json.fmt(message.content, .{})});
            if (message.tool_calls.len != 0) {
                try writer.writeAll(", \"tool_calls\": [");
                for (message.tool_calls, 0..) |tool_call, tool_idx| {
                    if (tool_idx != 0) try writer.writeAll(", ");
                    try writer.writeAll("{\"name\": ");
                    try writer.print("{f}", .{std.json.fmt(tool_call.name, .{})});
                    try writer.writeAll(", \"arguments\": ");
                    try writer.print("{f}", .{std.json.fmt(tool_call.arguments_json, .{})});
                    try writer.writeAll("}");
                }
                try writer.writeAll("]");
            }
            try writer.writeAll("}");
        }
        try writer.writeAll("\n  ]\n}\n");
    }

    fn loadFromFile(self: *ChatHistory, path: []const u8) !void {
        self.clear();

        var loaded = try loadChatMessages(self.allocator, path);
        defer loaded.deinit();

        for (loaded.items) |message| {
            try self.appendMessage(message);
        }
    }

    fn items(self: *const ChatHistory) []const decoder_family.Message {
        return self.messages.items;
    }
};

test "analyzeGeneratedText trims full stop sequence" {
    const testing = std.testing;

    const stops: []const []const u8 = &[_][]const u8{"today?"};
    const analysis = analyzeGeneratedText("Hello today?", @constCast(stops));
    try testing.expect(analysis.stop_hit);
    try testing.expectEqual(@as(usize, 6), analysis.printable_len);
    try testing.expectEqual(@as(usize, 6), analysis.response_len);
}

test "analyzeGeneratedText holds back partial stop prefix" {
    const testing = std.testing;

    const stops: []const []const u8 = &[_][]const u8{"today?"};
    const analysis = analyzeGeneratedText("Hello to", @constCast(stops));
    try testing.expect(!analysis.stop_hit);
    try testing.expectEqual(@as(usize, 6), analysis.printable_len);
    try testing.expectEqual(@as(usize, 8), analysis.response_len);
}

test "chat history session save and load preserves tool calls" {
    const testing = std.testing;

    var temp_dir = testing.tmpDir(.{});
    defer temp_dir.cleanup();

    const temp_root = try temp_dir.dir.realpathAlloc(testing.allocator, ".");
    defer testing.allocator.free(temp_root);
    const session_path = try std.fs.path.join(testing.allocator, &.{ temp_root, "session.json" });
    defer testing.allocator.free(session_path);

    var history = ChatHistory.init(testing.allocator);
    defer history.deinit();

    try history.append(.system, "You are terse.");
    try history.append(.user, "Use the weather tool.");
    try history.appendMessage(.{
        .role = .assistant,
        .content = "",
            .tool_calls = &[_]decoder_family.ToolCall{
                .{ .name = "lookup_weather", .arguments_json = "{\"city\":\"Shanghai\"}" },
            },
        });
    try history.append(.tool, "{\"temp\":22}");

    const stop_sequences = [_][]const u8{"</tool_response>"};
    try history.saveToFile(session_path, .{
        .model_dir = "models/Qwen3-0.6B",
        .options = .{
            .max_new_tokens = 64,
            .thinking_mode = .disabled,
            .system_prompt = "You are terse.",
            .sampling = cli_args.defaultSamplingConfig(.disabled),
            .seed = 7,
            .stream_output = true,
            .stop_sequences = @constCast(stop_sequences[0..]),
            .backend_scheme = .q4,
            .kv_cache_scheme = .q8,
            .thread_count = 8,
        },
    });

    const saved = try readFileAllocAtPath(testing.allocator, session_path, 64 * 1024);
    defer testing.allocator.free(saved);
    try testing.expect(std.mem.indexOf(u8, saved, "\"kind\": \"zinfer_chat_session\"") != null);
    try testing.expect(std.mem.indexOf(u8, saved, "\"tool_calls\"") != null);
    try testing.expect(std.mem.indexOf(u8, saved, "\"model_dir\": \"models/Qwen3-0.6B\"") != null);

    var loaded = ChatHistory.init(testing.allocator);
    defer loaded.deinit();
    try loaded.loadFromFile(session_path);

    try testing.expectEqual(@as(usize, 4), loaded.items().len);
    try testing.expectEqual(decoder_family.Role.system, loaded.items()[0].role);
    try testing.expectEqual(decoder_family.Role.assistant, loaded.items()[2].role);
    try testing.expectEqual(decoder_family.Role.tool, loaded.items()[3].role);
    try testing.expectEqual(@as(usize, 1), loaded.items()[2].tool_calls.len);
    try testing.expectEqualStrings("lookup_weather", loaded.items()[2].tool_calls[0].name);
    try testing.expectEqualStrings("{\"city\":\"Shanghai\"}", loaded.items()[2].tool_calls[0].arguments_json);
}
