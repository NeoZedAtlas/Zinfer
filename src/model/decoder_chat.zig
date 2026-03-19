const std = @import("std");
const decoder_family = @import("decoder_family.zig");
const decoder_runtime = @import("decoder_runtime.zig");

pub const ThinkingMode = decoder_family.ThinkingMode;
pub const Role = decoder_family.Role;
pub const ToolCall = decoder_family.ToolCall;
pub const Message = decoder_family.Message;

pub fn renderMessagesPromptAlloc(
    allocator: std.mem.Allocator,
    architecture: decoder_runtime.Architecture,
    messages: []const Message,
    mode: ThinkingMode,
) ![]u8 {
    return try decoder_family.renderMessagesPromptAlloc(allocator, architecture, messages, mode);
}

pub fn renderSingleUserPromptAlloc(
    allocator: std.mem.Allocator,
    architecture: decoder_runtime.Architecture,
    user_text: []const u8,
    mode: ThinkingMode,
) ![]u8 {
    return try decoder_family.renderSingleUserPromptAlloc(allocator, architecture, user_text, mode);
}

pub fn assistantHistoryContent(
    architecture: decoder_runtime.Architecture,
    content: []const u8,
) []const u8 {
    return decoder_family.assistantHistoryContent(architecture, content);
}
