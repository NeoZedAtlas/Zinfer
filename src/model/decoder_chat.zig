const std = @import("std");
const decoder_runtime = @import("decoder_runtime.zig");
const qwen_chat_template = @import("../tokenizer/qwen_chat_template.zig");

pub const ThinkingMode = qwen_chat_template.ThinkingMode;
pub const Role = qwen_chat_template.Role;
pub const ToolCall = qwen_chat_template.ToolCall;
pub const Message = qwen_chat_template.Message;

pub fn renderMessagesPromptAlloc(
    allocator: std.mem.Allocator,
    architecture: decoder_runtime.Architecture,
    messages: []const Message,
    mode: ThinkingMode,
) ![]u8 {
    return switch (architecture) {
        .qwen3 => qwen_chat_template.renderMessagesPromptAlloc(allocator, messages, mode),
    };
}

pub fn renderSingleUserPromptAlloc(
    allocator: std.mem.Allocator,
    architecture: decoder_runtime.Architecture,
    user_text: []const u8,
    mode: ThinkingMode,
) ![]u8 {
    return switch (architecture) {
        .qwen3 => qwen_chat_template.renderSingleUserPromptAlloc(allocator, user_text, mode),
    };
}

pub fn assistantHistoryContent(
    architecture: decoder_runtime.Architecture,
    content: []const u8,
) []const u8 {
    return switch (architecture) {
        .qwen3 => qwen_chat_template.assistantHistoryContent(content),
    };
}
