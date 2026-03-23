const std = @import("std");
const cli_args = @import("args.zig");
const GenerateOptions = cli_args.GenerateOptions;
const cli_messages = @import("messages.zig");
const cli_prompts = @import("prompts.zig");
const cli_runtime = @import("runtime.zig");
const decoder_family = @import("../../model/runtime/decoder_family.zig");

pub const SessionMetadata = struct {
    model_dir: []const u8,
    options: GenerateOptions,
};

pub const ChatHistory = struct {
    allocator: std.mem.Allocator,
    messages: std.ArrayListUnmanaged(decoder_family.Message),

    pub fn init(allocator: std.mem.Allocator) ChatHistory {
        return .{
            .allocator = allocator,
            .messages = .empty,
        };
    }

    pub fn deinit(self: *ChatHistory) void {
        self.clear();
        self.messages.deinit(self.allocator);
    }

    pub fn clear(self: *ChatHistory) void {
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

    pub fn append(self: *ChatHistory, role: decoder_family.Role, content: []const u8) !void {
        try self.appendMessage(.{
            .role = role,
            .content = content,
        });
    }

    pub fn appendMessage(self: *ChatHistory, message: decoder_family.Message) !void {
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

    pub fn prependSystemIfMissing(self: *ChatHistory, content: []const u8) !void {
        if (self.messages.items.len != 0 and self.messages.items[0].role == .system) return;

        const owned = try self.allocator.dupe(u8, content);
        errdefer self.allocator.free(owned);
        try self.messages.insert(self.allocator, 0, .{
            .role = .system,
            .content = owned,
        });
    }

    pub fn saveToFile(self: *const ChatHistory, path: []const u8, metadata: SessionMetadata) !void {
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
        try writer.writeAll("    \"q8_layout\": ");
        try writer.print("{f}", .{std.json.fmt(metadata.options.q8_layout.name(), .{})});
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

    pub fn loadFromFile(self: *ChatHistory, path: []const u8) !void {
        self.clear();

        var loaded = try cli_messages.loadChatMessages(self.allocator, path);
        defer loaded.deinit();

        for (loaded.items) |message| {
            try self.appendMessage(message);
        }
    }

    pub fn items(self: *const ChatHistory) []const decoder_family.Message {
        return self.messages.items;
    }
};

pub fn chatLoop(
    allocator: std.mem.Allocator,
    model_dir: []const u8,
    options: GenerateOptions,
    load_path: ?[]const u8,
    save_path: ?[]const u8,
) !void {
    var runtime = try cli_runtime.GeneratorRuntime.init(allocator, model_dir, options.backend_scheme, options.thread_count);
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
            .q8_layout = .head_major,
            .thread_count = 8,
        },
    });

    const saved = try cli_messages.readFileAllocAtPath(testing.allocator, session_path, 64 * 1024);
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
