const std = @import("std");
const decoder_family = @import("../../model/runtime/decoder_family.zig");

pub const LoadedChatMessages = struct {
    arena: std.heap.ArenaAllocator,
    items: []decoder_family.Message,

    pub fn deinit(self: *LoadedChatMessages) void {
        self.arena.deinit();
    }
};

pub fn loadChatMessages(
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

pub fn readFileAllocAtPath(
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

pub fn parseChatRole(text: []const u8) !decoder_family.Role {
    if (std.mem.eql(u8, text, "system")) return .system;
    if (std.mem.eql(u8, text, "user")) return .user;
    if (std.mem.eql(u8, text, "assistant")) return .assistant;
    if (std.mem.eql(u8, text, "tool")) return .tool;
    return error.InvalidChatRole;
}
