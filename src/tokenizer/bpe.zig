const std = @import("std");
const byte_mapping = @import("bpe/byte_mapping.zig");
const io = @import("bpe/io.zig");
const merge = @import("bpe/merge.zig");
const pretokenize = @import("bpe/pretokenize.zig");
const types = @import("bpe/types.zig");

pub const SpecialToken = types.SpecialToken;

pub const Tokenizer = struct {
    arena: std.heap.ArenaAllocator,
    vocab: std.StringHashMapUnmanaged(u32),
    id_to_token: []const ?[]const u8,
    merges: std.StringHashMapUnmanaged(u32),
    special_tokens: []const SpecialToken,
    byte_encoder: [256][]const u8,
    byte_decoder: std.AutoHashMapUnmanaged(u21, u8),

    pub fn deinit(self: *Tokenizer) void {
        self.arena.deinit();
    }

    pub fn loadFromModelDir(backing_allocator: std.mem.Allocator, model_dir: []const u8) !Tokenizer {
        var arena = std.heap.ArenaAllocator.init(backing_allocator);
        errdefer arena.deinit();
        const allocator = arena.allocator();

        const vocab_path = try std.fs.path.join(allocator, &.{ model_dir, "vocab.json" });
        defer allocator.free(vocab_path);
        const merges_path = try std.fs.path.join(allocator, &.{ model_dir, "merges.txt" });
        defer allocator.free(merges_path);
        const tokenizer_config_path = try std.fs.path.join(allocator, &.{ model_dir, "tokenizer_config.json" });
        defer allocator.free(tokenizer_config_path);

        var byte_encoder: [256][]const u8 = undefined;
        var byte_decoder: std.AutoHashMapUnmanaged(u21, u8) = .empty;
        try byte_mapping.init(allocator, &byte_encoder, &byte_decoder);

        var vocab: std.StringHashMapUnmanaged(u32) = .empty;
        _ = try io.loadVocab(allocator, vocab_path, &vocab);
        var merges: std.StringHashMapUnmanaged(u32) = .empty;
        try io.loadMerges(allocator, merges_path, &merges);

        const special_tokens = try io.loadSpecialTokens(allocator, tokenizer_config_path, &vocab);
        const max_id = io.computeMaxId(vocab, special_tokens);
        const id_to_token = try allocator.alloc(?[]const u8, max_id + 1);
        @memset(id_to_token, null);

        var vocab_it = vocab.iterator();
        while (vocab_it.next()) |entry| {
            id_to_token[entry.value_ptr.*] = entry.key_ptr.*;
        }

        return .{
            .arena = arena,
            .vocab = vocab,
            .id_to_token = id_to_token,
            .merges = merges,
            .special_tokens = special_tokens,
            .byte_encoder = byte_encoder,
            .byte_decoder = byte_decoder,
        };
    }

    pub fn encodeAlloc(self: *const Tokenizer, backing_allocator: std.mem.Allocator, text: []const u8) ![]u32 {
        var arena = std.heap.ArenaAllocator.init(backing_allocator);
        defer arena.deinit();
        const allocator = arena.allocator();

        var ids: std.ArrayListUnmanaged(u32) = .empty;
        defer ids.deinit(backing_allocator);

        var index: usize = 0;
        while (index < text.len) {
            if (pretokenize.matchSpecialToken(self.special_tokens, text[index..])) |matched| {
                try ids.append(backing_allocator, matched.id);
                index += matched.content.len;
                continue;
            }

            const segment = pretokenize.nextSegment(self.special_tokens, text, &index) orelse break;
            if (segment.len == 0) continue;
            try self.encodeSegment(allocator, backing_allocator, segment, &ids);
        }

        return ids.toOwnedSlice(backing_allocator);
    }

    pub fn decodeAlloc(self: *const Tokenizer, allocator: std.mem.Allocator, ids: []const u32) ![]u8 {
        var bytes = std.ArrayListUnmanaged(u8).empty;
        defer bytes.deinit(allocator);

        for (ids) |id| {
            const token = self.tokenForId(id) orelse return error.UnknownTokenId;
            if (self.isSpecialTokenId(id)) {
                try bytes.appendSlice(allocator, token);
                continue;
            }

            var it = try std.unicode.Utf8View.init(token);
            var utf8_it = it.iterator();
            while (utf8_it.nextCodepoint()) |cp| {
                const b = self.byte_decoder.get(cp) orelse return error.InvalidByteDecoderEntry;
                try bytes.append(allocator, b);
            }
        }

        const raw = try bytes.toOwnedSlice(allocator);
        errdefer allocator.free(raw);
        if (std.unicode.utf8ValidateSlice(raw)) return raw;

        const fixed = std.unicode.wtf8ToUtf8LossyAlloc(allocator, raw) catch |err| switch (err) {
            error.InvalidWtf8 => try utf8LossyAlloc(allocator, raw),
            else => return err,
        };
        allocator.free(raw);
        return fixed;
    }

    fn encodeSegment(
        self: *const Tokenizer,
        temp_allocator: std.mem.Allocator,
        output_allocator: std.mem.Allocator,
        segment: []const u8,
        ids: *std.ArrayListUnmanaged(u32),
    ) !void {
        var byte_encoded = std.ArrayListUnmanaged(u8).empty;
        defer byte_encoded.deinit(temp_allocator);
        for (segment) |byte| {
            try byte_encoded.appendSlice(temp_allocator, self.byte_encoder[byte]);
        }

        const encoded = try byte_encoded.toOwnedSlice(temp_allocator);
        const pieces = try merge.apply(temp_allocator, self.merges, encoded);
        for (pieces) |piece| {
            const id = self.vocab.get(piece) orelse return error.TokenNotFound;
            try ids.append(output_allocator, id);
        }
    }

    fn tokenForId(self: *const Tokenizer, id: u32) ?[]const u8 {
        if (id >= self.id_to_token.len) return null;
        return self.id_to_token[id];
    }

    fn isSpecialTokenId(self: *const Tokenizer, id: u32) bool {
        for (self.special_tokens) |token| {
            if (token.id == id) return true;
        }
        return false;
    }
};

fn utf8LossyAlloc(allocator: std.mem.Allocator, raw: []const u8) ![]u8 {
    var output = std.ArrayListUnmanaged(u8).empty;
    defer output.deinit(allocator);

    var index: usize = 0;
    while (index < raw.len) {
        const expected_len = std.unicode.utf8ByteSequenceLength(raw[index]) catch {
            try output.appendSlice(allocator, "\xEF\xBF\xBD");
            index += 1;
            continue;
        };
        if (index + expected_len <= raw.len and std.unicode.utf8ValidateSlice(raw[index .. index + expected_len])) {
            try output.appendSlice(allocator, raw[index .. index + expected_len]);
            index += expected_len;
            continue;
        }

        try output.appendSlice(allocator, "\xEF\xBF\xBD");
        index += 1;
    }

    return output.toOwnedSlice(allocator);
}

test "bpe tokenizer encodes known samples" {
    const testing = std.testing;
    var tokenizer = try Tokenizer.loadFromModelDir(testing.allocator, "models/Qwen3-0.6B");
    defer tokenizer.deinit();

    {
        const ids = try tokenizer.encodeAlloc(testing.allocator, "Hello");
        defer testing.allocator.free(ids);
        try testing.expectEqualSlices(u32, &[_]u32{9707}, ids);
    }
    {
        const ids = try tokenizer.encodeAlloc(testing.allocator, " hello");
        defer testing.allocator.free(ids);
        try testing.expectEqualSlices(u32, &[_]u32{23811}, ids);
    }
    {
        const ids = try tokenizer.encodeAlloc(testing.allocator, "<|im_start|>user\nHello<|im_end|>\n");
        defer testing.allocator.free(ids);
        try testing.expectEqualSlices(u32, &[_]u32{ 151644, 872, 198, 9707, 151645, 198 }, ids);
    }
    {
        const ids = try tokenizer.encodeAlloc(testing.allocator, "<think>test</think>");
        defer testing.allocator.free(ids);
        try testing.expectEqualSlices(u32, &[_]u32{ 151667, 1944, 151668 }, ids);
    }
}

test "bpe tokenizer decodes known samples" {
    const testing = std.testing;
    var tokenizer = try Tokenizer.loadFromModelDir(testing.allocator, "models/Qwen3-0.6B");
    defer tokenizer.deinit();

    const text = try tokenizer.decodeAlloc(testing.allocator, &[_]u32{ 151644, 872, 198, 9707, 151645, 198 });
    defer testing.allocator.free(text);
    try testing.expectEqualStrings("<|im_start|>user\nHello<|im_end|>\n", text);
}
