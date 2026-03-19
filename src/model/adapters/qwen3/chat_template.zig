const std = @import("std");
const adapter_tokenizer = @import("tokenizer.zig");

pub const ThinkingMode = enum {
    enabled,
    disabled,
};

pub const Role = enum {
    system,
    user,
    assistant,
    tool,

    pub fn name(self: Role) []const u8 {
        return switch (self) {
            .system => "system",
            .user => "user",
            .assistant => "assistant",
            .tool => "tool",
        };
    }
};

pub const ToolCall = struct {
    name: []const u8,
    arguments_json: []const u8,
};

pub const Message = struct {
    role: Role,
    content: []const u8,
    tool_calls: []const ToolCall = &.{},
};

pub fn renderSingleUserPromptAlloc(
    allocator: std.mem.Allocator,
    user_text: []const u8,
    mode: ThinkingMode,
) ![]u8 {
    const messages = [_]Message{
        .{ .role = .user, .content = user_text },
    };
    return renderMessagesPromptAlloc(allocator, &messages, mode);
}

pub fn renderMessagesPromptAlloc(
    allocator: std.mem.Allocator,
    messages: []const Message,
    mode: ThinkingMode,
) ![]u8 {
    var output = std.ArrayListUnmanaged(u8).empty;
    defer output.deinit(allocator);

    for (messages, 0..) |message, idx| {
        if (message.role == .tool) {
            try appendToolResponseMessage(allocator, &output, message, idx, messages);
            continue;
        }
        try appendRenderedMessage(allocator, &output, message);
    }

    try output.appendSlice(allocator, "<|im_start|>assistant\n");
    switch (mode) {
        .enabled => {},
        .disabled => try output.appendSlice(allocator, "<think>\n\n</think>\n\n"),
    }

    return output.toOwnedSlice(allocator);
}

pub fn assistantHistoryContent(content: []const u8) []const u8 {
    return stripHistoricalThinking(content);
}

fn appendRenderedMessage(
    allocator: std.mem.Allocator,
    output: *std.ArrayListUnmanaged(u8),
    message: Message,
) !void {
    try output.appendSlice(allocator, "<|im_start|>");
    try output.appendSlice(allocator, switch (message.role) {
        .tool => unreachable,
        else => message.role.name(),
    });
    try output.appendSlice(allocator, "\n");

    const content = switch (message.role) {
        .assistant => stripHistoricalThinking(message.content),
        else => message.content,
    };
    try output.appendSlice(allocator, content);

    if (message.role == .assistant and message.tool_calls.len != 0) {
        for (message.tool_calls, 0..) |tool_call, idx| {
            if ((idx == 0 and content.len != 0) or idx != 0) {
                try output.appendSlice(allocator, "\n");
            }
            try appendToolCallXml(allocator, output, tool_call);
        }
    }

    try output.appendSlice(allocator, "<|im_end|>\n");
}

fn appendToolCallXml(
    allocator: std.mem.Allocator,
    output: *std.ArrayListUnmanaged(u8),
    tool_call: ToolCall,
) !void {
    try output.appendSlice(allocator, "<tool_call>\n{\"name\": ");
    try appendJsonString(allocator, output, tool_call.name);
    try output.appendSlice(allocator, ", \"arguments\": ");
    try output.appendSlice(allocator, tool_call.arguments_json);
    try output.appendSlice(allocator, "}\n</tool_call>");
}

fn appendToolResponseMessage(
    allocator: std.mem.Allocator,
    output: *std.ArrayListUnmanaged(u8),
    message: Message,
    idx: usize,
    messages: []const Message,
) !void {
    const first_tool = idx == 0 or messages[idx - 1].role != .tool;
    const last_tool = idx + 1 == messages.len or messages[idx + 1].role != .tool;

    if (first_tool) try output.appendSlice(allocator, "<|im_start|>user");
    try output.appendSlice(allocator, "\n<tool_response>\n");
    try output.appendSlice(allocator, message.content);
    try output.appendSlice(allocator, "\n</tool_response>");
    if (last_tool) try output.appendSlice(allocator, "<|im_end|>\n");
}

fn appendJsonString(
    allocator: std.mem.Allocator,
    output: *std.ArrayListUnmanaged(u8),
    value: []const u8,
) !void {
    var temp = std.ArrayListUnmanaged(u8).empty;
    defer temp.deinit(allocator);
    try temp.writer(allocator).print("{f}", .{std.json.fmt(value, .{})});
    try output.appendSlice(allocator, temp.items);
}

fn stripHistoricalThinking(content: []const u8) []const u8 {
    const close_tag = "</think>";
    const close_index = std.mem.indexOf(u8, content, close_tag) orelse return content;
    return std.mem.trimLeft(u8, content[close_index + close_tag.len ..], "\n");
}

test "single user prompt matches adapter thinking template" {
    const testing = std.testing;
    const prompt = try renderSingleUserPromptAlloc(testing.allocator, "Hello", .enabled);
    defer testing.allocator.free(prompt);

    try testing.expectEqualStrings(
        "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n",
        prompt,
    );

    var tokenizer = try adapter_tokenizer.Tokenizer.loadFromModelDir(testing.allocator, "models/Qwen3-0.6B");
    defer tokenizer.deinit();

    const ids = try tokenizer.encodeAlloc(testing.allocator, prompt);
    defer testing.allocator.free(ids);
    try testing.expectEqualSlices(u32, &[_]u32{ 151644, 872, 198, 9707, 151645, 198, 151644, 77091, 198 }, ids);
}

test "single user prompt matches adapter non-thinking template" {
    const testing = std.testing;
    const prompt = try renderSingleUserPromptAlloc(testing.allocator, "Hello", .disabled);
    defer testing.allocator.free(prompt);

    try testing.expectEqualStrings(
        "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        prompt,
    );

    var tokenizer = try adapter_tokenizer.Tokenizer.loadFromModelDir(testing.allocator, "models/Qwen3-0.6B");
    defer tokenizer.deinit();

    const ids = try tokenizer.encodeAlloc(testing.allocator, prompt);
    defer testing.allocator.free(ids);
    try testing.expectEqualSlices(u32, &[_]u32{ 151644, 872, 198, 9707, 151645, 198, 151644, 77091, 198, 151667, 271, 151668, 271 }, ids);
}

test "multi-message prompt matches adapter template and strips historical thinking" {
    const testing = std.testing;
    const messages = [_]Message{
        .{ .role = .system, .content = "You are terse." },
        .{ .role = .user, .content = "Hi" },
        .{ .role = .assistant, .content = "<think>\ninternal\n</think>\nHello there" },
        .{ .role = .user, .content = "Next" },
    };

    const prompt = try renderMessagesPromptAlloc(testing.allocator, &messages, .disabled);
    defer testing.allocator.free(prompt);

    try testing.expectEqualStrings(
        "<|im_start|>system\nYou are terse.<|im_end|>\n" ++
            "<|im_start|>user\nHi<|im_end|>\n" ++
            "<|im_start|>assistant\nHello there<|im_end|>\n" ++
            "<|im_start|>user\nNext<|im_end|>\n" ++
            "<|im_start|>assistant\n<think>\n\n</think>\n\n",
        prompt,
    );
}

test "assistant tool call renders xml payload" {
    const testing = std.testing;
    const messages = [_]Message{
        .{
            .role = .assistant,
            .content = "",
            .tool_calls = &[_]ToolCall{
                .{ .name = "lookup_weather", .arguments_json = "{\"city\":\"Shanghai\"}" },
            },
        },
    };

    const prompt = try renderMessagesPromptAlloc(testing.allocator, &messages, .enabled);
    defer testing.allocator.free(prompt);

    try testing.expectEqualStrings(
        "<|im_start|>assistant\n" ++
            "<tool_call>\n{\"name\": \"lookup_weather\", \"arguments\": {\"city\":\"Shanghai\"}}\n</tool_call><|im_end|>\n" ++
            "<|im_start|>assistant\n",
        prompt,
    );
}

test "tool response renders in user wrapper" {
    const testing = std.testing;
    const messages = [_]Message{
        .{ .role = .tool, .content = "{\"ok\":true}" },
        .{ .role = .tool, .content = "{\"temp\":22}" },
    };

    const prompt = try renderMessagesPromptAlloc(testing.allocator, &messages, .enabled);
    defer testing.allocator.free(prompt);

    try testing.expectEqualStrings(
        "<|im_start|>user\n<tool_response>\n{\"ok\":true}\n</tool_response>\n<tool_response>\n{\"temp\":22}\n</tool_response><|im_end|>\n" ++
            "<|im_start|>assistant\n",
        prompt,
    );
}
