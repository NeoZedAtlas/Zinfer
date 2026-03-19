const std = @import("std");
const decoder_runtime = @import("decoder_runtime.zig");
const qwen_chat_template = @import("../tokenizer/qwen_chat_template.zig");

pub const ThinkingMode = qwen_chat_template.ThinkingMode;
pub const Role = qwen_chat_template.Role;
pub const ToolCall = qwen_chat_template.ToolCall;
pub const Message = qwen_chat_template.Message;

const RegistryEntry = struct {
    render_messages_prompt_alloc: *const fn (std.mem.Allocator, []const Message, ThinkingMode) anyerror![]u8,
    render_single_user_prompt_alloc: *const fn (std.mem.Allocator, []const u8, ThinkingMode) anyerror![]u8,
    assistant_history_content: *const fn ([]const u8) []const u8,
};

fn entryForArchitecture(architecture: decoder_runtime.Architecture) RegistryEntry {
    return switch (architecture) {
        .qwen3 => .{
            .render_messages_prompt_alloc = qwen_chat_template.renderMessagesPromptAlloc,
            .render_single_user_prompt_alloc = qwen_chat_template.renderSingleUserPromptAlloc,
            .assistant_history_content = qwen_chat_template.assistantHistoryContent,
        },
    };
}

pub fn renderMessagesPromptAlloc(
    allocator: std.mem.Allocator,
    architecture: decoder_runtime.Architecture,
    messages: []const Message,
    mode: ThinkingMode,
) ![]u8 {
    return try entryForArchitecture(architecture).render_messages_prompt_alloc(allocator, messages, mode);
}

pub fn renderSingleUserPromptAlloc(
    allocator: std.mem.Allocator,
    architecture: decoder_runtime.Architecture,
    user_text: []const u8,
    mode: ThinkingMode,
) ![]u8 {
    return try entryForArchitecture(architecture).render_single_user_prompt_alloc(allocator, user_text, mode);
}

pub fn assistantHistoryContent(
    architecture: decoder_runtime.Architecture,
    content: []const u8,
) []const u8 {
    return entryForArchitecture(architecture).assistant_history_content(content);
}
