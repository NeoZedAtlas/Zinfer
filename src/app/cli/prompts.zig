const std = @import("std");
const decoder_family = @import("../../model/runtime/decoder_family.zig");

pub fn buildSingleUserPromptAlloc(
    allocator: std.mem.Allocator,
    architecture: decoder_family.Architecture,
    user_text: []const u8,
    system_prompt: ?[]const u8,
    thinking_mode: decoder_family.ThinkingMode,
) ![]u8 {
    if (system_prompt) |system| {
        const messages = [_]decoder_family.Message{
            .{ .role = .system, .content = system },
            .{ .role = .user, .content = user_text },
        };
        return decoder_family.renderMessagesPromptAlloc(allocator, architecture, &messages, thinking_mode);
    }
    return decoder_family.renderSingleUserPromptAlloc(allocator, architecture, user_text, thinking_mode);
}

pub fn buildMessagesPromptAlloc(
    allocator: std.mem.Allocator,
    architecture: decoder_family.Architecture,
    messages: []const decoder_family.Message,
    system_prompt: ?[]const u8,
    thinking_mode: decoder_family.ThinkingMode,
) ![]u8 {
    if (system_prompt == null) {
        return decoder_family.renderMessagesPromptAlloc(allocator, architecture, messages, thinking_mode);
    }

    const system = system_prompt.?;
    const needs_prepend = messages.len == 0 or messages[0].role != .system;
    if (!needs_prepend) {
        return decoder_family.renderMessagesPromptAlloc(allocator, architecture, messages, thinking_mode);
    }

    const expanded = try allocator.alloc(decoder_family.Message, messages.len + 1);
    defer allocator.free(expanded);
    expanded[0] = .{ .role = .system, .content = system };
    @memcpy(expanded[1..], messages);
    return decoder_family.renderMessagesPromptAlloc(allocator, architecture, expanded, thinking_mode);
}

pub fn thinkingModeName(mode: decoder_family.ThinkingMode) []const u8 {
    return switch (mode) {
        .enabled => "think",
        .disabled => "no-think",
    };
}
