const std = @import("std");
const qwen_bpe = @import("../tokenizer/qwen_bpe.zig");
const qwen_chat_template = @import("../tokenizer/qwen_chat_template.zig");

pub const Architecture = enum {
    qwen3,

    pub fn name(self: Architecture) []const u8 {
        return entryForArchitecture(self).model_type;
    }
};

pub const ThinkingMode = qwen_chat_template.ThinkingMode;
pub const Role = qwen_chat_template.Role;
pub const ToolCall = qwen_chat_template.ToolCall;
pub const Message = qwen_chat_template.Message;

pub const Tokenizer = union(Architecture) {
    qwen3: qwen_bpe.Tokenizer,

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
    load_tokenizer: *const fn (std.mem.Allocator, []const u8) anyerror!Tokenizer,
    render_messages_prompt_alloc: *const fn (std.mem.Allocator, []const Message, ThinkingMode) anyerror![]u8,
    render_single_user_prompt_alloc: *const fn (std.mem.Allocator, []const u8, ThinkingMode) anyerror![]u8,
    assistant_history_content: *const fn ([]const u8) []const u8,
};

pub fn detectArchitecture(model_type: []const u8) ?Architecture {
    inline for (std.meta.tags(Architecture)) |tag| {
        if (std.mem.eql(u8, model_type, entryForArchitecture(tag).model_type)) {
            return tag;
        }
    }
    return null;
}

pub fn loadTokenizerFromModelDir(
    backing_allocator: std.mem.Allocator,
    architecture: Architecture,
    model_dir: []const u8,
) !Tokenizer {
    return Tokenizer.loadFromModelDir(backing_allocator, architecture, model_dir);
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
            .load_tokenizer = loadQwen3TokenizerFromModelDir,
            .render_messages_prompt_alloc = qwen_chat_template.renderMessagesPromptAlloc,
            .render_single_user_prompt_alloc = qwen_chat_template.renderSingleUserPromptAlloc,
            .assistant_history_content = qwen_chat_template.assistantHistoryContent,
        },
    };
}

fn loadQwen3TokenizerFromModelDir(
    backing_allocator: std.mem.Allocator,
    model_dir: []const u8,
) !Tokenizer {
    return .{
        .qwen3 = try qwen_bpe.Tokenizer.loadFromModelDir(backing_allocator, model_dir),
    };
}

test "family detects qwen3 model type" {
    const testing = std.testing;
    try testing.expectEqual(Architecture.qwen3, detectArchitecture("qwen3").?);
    try testing.expect(detectArchitecture("unknown-model") == null);
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
