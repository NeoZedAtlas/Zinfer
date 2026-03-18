const std = @import("std");
const qwen_bpe = @import("qwen_bpe.zig");

pub const ThinkingMode = enum {
    enabled,
    disabled,
};

pub const Role = enum {
    system,
    user,
    assistant,

    pub fn name(self: Role) []const u8 {
        return switch (self) {
            .system => "system",
            .user => "user",
            .assistant => "assistant",
        };
    }
};

pub const Message = struct {
    role: Role,
    content: []const u8,
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

    for (messages) |message| {
        try appendRenderedMessage(allocator, &output, message);
    }

    try output.appendSlice(allocator, "<|im_start|>assistant\n");
    switch (mode) {
        .enabled => {},
        .disabled => try output.appendSlice(allocator, "<think>\n\n</think>\n\n"),
    }

    return output.toOwnedSlice(allocator);
}

fn appendRenderedMessage(
    allocator: std.mem.Allocator,
    output: *std.ArrayListUnmanaged(u8),
    message: Message,
) !void {
    try output.appendSlice(allocator, "<|im_start|>");
    try output.appendSlice(allocator, message.role.name());
    try output.appendSlice(allocator, "\n");

    const content = switch (message.role) {
        .assistant => stripHistoricalThinking(message.content),
        else => message.content,
    };
    try output.appendSlice(allocator, content);
    try output.appendSlice(allocator, "<|im_end|>\n");
}

fn stripHistoricalThinking(content: []const u8) []const u8 {
    const close_tag = "</think>";
    const close_index = std.mem.indexOf(u8, content, close_tag) orelse return content;
    return std.mem.trimLeft(u8, content[close_index + close_tag.len ..], "\n");
}

test "single user prompt matches qwen thinking template" {
    const testing = std.testing;
    const prompt = try renderSingleUserPromptAlloc(testing.allocator, "Hello", .enabled);
    defer testing.allocator.free(prompt);

    try testing.expectEqualStrings(
        "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n",
        prompt,
    );

    var tokenizer = try qwen_bpe.Tokenizer.loadFromModelDir(testing.allocator, "models/Qwen3-0.6B");
    defer tokenizer.deinit();

    const ids = try tokenizer.encodeAlloc(testing.allocator, prompt);
    defer testing.allocator.free(ids);
    try testing.expectEqualSlices(u32, &[_]u32{ 151644, 872, 198, 9707, 151645, 198, 151644, 77091, 198 }, ids);
}

test "single user prompt matches qwen non-thinking template" {
    const testing = std.testing;
    const prompt = try renderSingleUserPromptAlloc(testing.allocator, "Hello", .disabled);
    defer testing.allocator.free(prompt);

    try testing.expectEqualStrings(
        "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        prompt,
    );

    var tokenizer = try qwen_bpe.Tokenizer.loadFromModelDir(testing.allocator, "models/Qwen3-0.6B");
    defer tokenizer.deinit();

    const ids = try tokenizer.encodeAlloc(testing.allocator, prompt);
    defer testing.allocator.free(ids);
    try testing.expectEqualSlices(u32, &[_]u32{ 151644, 872, 198, 9707, 151645, 198, 151644, 77091, 198, 151667, 271, 151668, 271 }, ids);
}

test "multi-message prompt matches qwen template and strips historical thinking" {
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

    var tokenizer = try qwen_bpe.Tokenizer.loadFromModelDir(testing.allocator, "models/Qwen3-0.6B");
    defer tokenizer.deinit();

    const ids = try tokenizer.encodeAlloc(testing.allocator, prompt);
    defer testing.allocator.free(ids);
    try testing.expectEqualSlices(u32, &[_]u32{
        151644, 8948, 198, 2610, 525, 50537, 13, 151645, 198,
        151644, 872, 198, 13048, 151645, 198,
        151644, 77091, 198, 9707, 1052, 151645, 198,
        151644, 872, 198, 5847, 151645, 198,
        151644, 77091, 198, 151667, 271, 151668, 271,
    }, ids);
}
