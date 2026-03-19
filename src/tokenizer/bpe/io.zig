const std = @import("std");
const types = @import("types.zig");

pub fn loadVocab(
    allocator: std.mem.Allocator,
    path: []const u8,
    vocab: *std.StringHashMapUnmanaged(u32),
) !usize {
    const bytes = try std.fs.cwd().readFileAlloc(allocator, path, 32 * 1024 * 1024);
    const root = try std.json.parseFromSliceLeaky(std.json.Value, allocator, bytes, .{});
    if (root != .object) return error.InvalidVocab;

    var count: usize = 0;
    var it = root.object.iterator();
    while (it.next()) |entry| {
        const value = switch (entry.value_ptr.*) {
            .integer => |n| n,
            else => return error.InvalidVocab,
        };
        try vocab.put(allocator, entry.key_ptr.*, @intCast(value));
        count += 1;
    }
    return count;
}

pub fn loadMerges(
    allocator: std.mem.Allocator,
    path: []const u8,
    merges: *std.StringHashMapUnmanaged(u32),
) !void {
    const bytes = try std.fs.cwd().readFileAlloc(allocator, path, 16 * 1024 * 1024);
    var lines = std.mem.splitScalar(u8, bytes, '\n');
    var rank: u32 = 0;
    while (lines.next()) |line_raw| {
        const line = std.mem.trim(u8, line_raw, " \r\t");
        if (line.len == 0) continue;
        if (line[0] == '#') continue;
        try merges.put(allocator, line, rank);
        rank += 1;
    }
}

pub fn loadSpecialTokens(
    allocator: std.mem.Allocator,
    path: []const u8,
    vocab: *std.StringHashMapUnmanaged(u32),
) ![]const types.SpecialToken {
    const bytes = try std.fs.cwd().readFileAlloc(allocator, path, 4 * 1024 * 1024);
    const root = try std.json.parseFromSliceLeaky(std.json.Value, allocator, bytes, .{});
    if (root != .object) return error.InvalidTokenizerConfig;

    const decoder_value = root.object.get("added_tokens_decoder") orelse return error.InvalidTokenizerConfig;
    if (decoder_value != .object) return error.InvalidTokenizerConfig;

    var tokens = std.ArrayListUnmanaged(types.SpecialToken).empty;
    defer tokens.deinit(allocator);

    var it = decoder_value.object.iterator();
    while (it.next()) |entry| {
        const id = try std.fmt.parseInt(u32, entry.key_ptr.*, 10);
        if (entry.value_ptr.* != .object) return error.InvalidTokenizerConfig;
        const content_value = entry.value_ptr.*.object.get("content") orelse return error.InvalidTokenizerConfig;
        if (content_value != .string) return error.InvalidTokenizerConfig;

        try vocab.put(allocator, content_value.string, id);
        try tokens.append(allocator, .{
            .id = id,
            .content = content_value.string,
        });
    }

    std.sort.block(types.SpecialToken, tokens.items, {}, struct {
        fn lessThan(_: void, lhs: types.SpecialToken, rhs: types.SpecialToken) bool {
            return lhs.content.len > rhs.content.len;
        }
    }.lessThan);

    return tokens.toOwnedSlice(allocator);
}

pub fn computeMaxId(vocab: std.StringHashMapUnmanaged(u32), special_tokens: []const types.SpecialToken) usize {
    var max_id: usize = 0;
    var it = vocab.iterator();
    while (it.next()) |entry| {
        max_id = @max(max_id, entry.value_ptr.*);
    }
    for (special_tokens) |token| {
        max_id = @max(max_id, token.id);
    }
    return max_id;
}
