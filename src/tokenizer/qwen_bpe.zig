const std = @import("std");

pub const SpecialToken = struct {
    id: u32,
    content: []const u8,
};

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
        try initByteMappings(allocator, &byte_encoder, &byte_decoder);

        var vocab: std.StringHashMapUnmanaged(u32) = .empty;
        const base_vocab_size = try loadVocab(allocator, vocab_path, &vocab);
        var merges: std.StringHashMapUnmanaged(u32) = .empty;
        try loadMerges(allocator, merges_path, &merges);

        const special_tokens = try loadSpecialTokens(allocator, tokenizer_config_path, &vocab);
        const max_id = try computeMaxId(vocab, special_tokens);
        const id_to_token = try allocator.alloc(?[]const u8, max_id + 1);
        @memset(id_to_token, null);

        var vocab_it = vocab.iterator();
        while (vocab_it.next()) |entry| {
            id_to_token[entry.value_ptr.*] = entry.key_ptr.*;
        }

        _ = base_vocab_size;
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
            if (self.matchSpecialToken(text[index..])) |matched| {
                try ids.append(backing_allocator, matched.id);
                index += matched.content.len;
                continue;
            }

            const segment = self.nextPretokenizedSegment(text, &index) orelse break;
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
        if (std.unicode.utf8ValidateSlice(raw)) return raw;

        const fixed = try std.unicode.wtf8ToUtf8LossyAlloc(allocator, raw);
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
        const pieces = try self.applyBpe(temp_allocator, encoded);
        for (pieces) |piece| {
            const id = self.vocab.get(piece) orelse return error.TokenNotFound;
            try ids.append(output_allocator, id);
        }
    }

    fn applyBpe(self: *const Tokenizer, allocator: std.mem.Allocator, encoded: []const u8) ![]const []const u8 {
        var pieces = std.ArrayListUnmanaged([]const u8).empty;
        defer pieces.deinit(allocator);

        var view = try std.unicode.Utf8View.init(encoded);
        var it = view.iterator();
        while (it.nextCodepointSlice()) |cp_slice| {
            try pieces.append(allocator, cp_slice);
        }

        while (pieces.items.len >= 2) {
            var best_rank: ?u32 = null;
            var best_index: usize = 0;

            for (0..pieces.items.len - 1) |i| {
                const rank = self.mergeRank(allocator, pieces.items[i], pieces.items[i + 1]) orelse continue;
                if (best_rank == null or rank < best_rank.?) {
                    best_rank = rank;
                    best_index = i;
                }
            }

            if (best_rank == null) break;

            const merged = try std.mem.concat(allocator, u8, &.{ pieces.items[best_index], pieces.items[best_index + 1] });
            pieces.items[best_index] = merged;
            _ = pieces.orderedRemove(best_index + 1);
        }

        return pieces.toOwnedSlice(allocator);
    }

    fn mergeRank(self: *const Tokenizer, allocator: std.mem.Allocator, left: []const u8, right: []const u8) ?u32 {
        const key = std.fmt.allocPrint(allocator, "{s} {s}", .{ left, right }) catch return null;
        defer allocator.free(key);
        return self.merges.get(key);
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

    fn matchSpecialToken(self: *const Tokenizer, input: []const u8) ?SpecialToken {
        var best: ?SpecialToken = null;
        for (self.special_tokens) |token| {
            if (input.len < token.content.len) continue;
            if (!std.mem.eql(u8, input[0..token.content.len], token.content)) continue;
            if (best == null or token.content.len > best.?.content.len) {
                best = token;
            }
        }
        return best;
    }

    fn nextPretokenizedSegment(self: *const Tokenizer, text: []const u8, index: *usize) ?[]const u8 {
        if (index.* >= text.len) return null;

        const start = index.*;
        const first_len = std.unicode.utf8ByteSequenceLength(text[start]) catch return null;
        const first_slice = text[start .. start + first_len];
        const first_cp = std.unicode.utf8Decode(first_slice) catch return null;

        if (isNewline(first_cp)) {
            var end = start + first_len;
            while (end < text.len) {
                if (self.matchSpecialToken(text[end..]) != null) break;
                const len = std.unicode.utf8ByteSequenceLength(text[end]) catch break;
                const slice = text[end .. end + len];
                const cp = std.unicode.utf8Decode(slice) catch break;
                if (!isNewline(cp)) break;
                end += len;
            }
            index.* = end;
            return text[start..end];
        }

        if (isHorizontalWhitespace(first_cp)) {
            var end = start;
            while (end < text.len) {
                if (self.matchSpecialToken(text[end..]) != null) break;
                const len = std.unicode.utf8ByteSequenceLength(text[end]) catch break;
                const slice = text[end .. end + len];
                const cp = std.unicode.utf8Decode(slice) catch break;
                if (!isHorizontalWhitespace(cp)) break;
                end += len;
            }
            if (end < text.len) {
                if (self.matchSpecialToken(text[end..]) != null) {
                    index.* = end;
                    return text[start..end];
                }
                const len = std.unicode.utf8ByteSequenceLength(text[end]) catch 0;
                if (len != 0) {
                    const cp = std.unicode.utf8Decode(text[end .. end + len]) catch 0;
                    if (isLetterLike(cp)) {
                        if (end - start > 1) {
                            index.* = end - 1;
                            return text[start .. end - 1];
                        }
                        var word_end = end + len;
                        while (word_end < text.len) {
                            if (self.matchSpecialToken(text[word_end..]) != null) break;
                            const next_len = std.unicode.utf8ByteSequenceLength(text[word_end]) catch break;
                            const next_slice = text[word_end .. word_end + next_len];
                            const next_cp = std.unicode.utf8Decode(next_slice) catch break;
                            if (!isLetterLike(next_cp)) break;
                            word_end += next_len;
                        }
                        index.* = word_end;
                        return text[start..word_end];
                    }
                    if (!isDigitLike(cp) and !isNewline(cp) and !isHorizontalWhitespace(cp)) {
                        if (end - start > 1) {
                            index.* = end - 1;
                            return text[start .. end - 1];
                        }
                        var punct_end = end + len;
                        while (punct_end < text.len) {
                            if (self.matchSpecialToken(text[punct_end..]) != null) break;
                            const next_len = std.unicode.utf8ByteSequenceLength(text[punct_end]) catch break;
                            const next_slice = text[punct_end .. punct_end + next_len];
                            const next_cp = std.unicode.utf8Decode(next_slice) catch break;
                            if (isDigitLike(next_cp) or isLetterLike(next_cp) or isNewline(next_cp) or isHorizontalWhitespace(next_cp)) break;
                            punct_end += next_len;
                        }
                        while (punct_end < text.len) {
                            if (self.matchSpecialToken(text[punct_end..]) != null) break;
                            const next_len = std.unicode.utf8ByteSequenceLength(text[punct_end]) catch break;
                            const next_slice = text[punct_end .. punct_end + next_len];
                            const next_cp = std.unicode.utf8Decode(next_slice) catch break;
                            if (!isNewline(next_cp)) break;
                            punct_end += next_len;
                        }
                        index.* = punct_end;
                        return text[start..punct_end];
                    }
                }
            }
            index.* = end;
            return text[start..end];
        }

        if (first_slice.len == 1 and first_slice[0] == '\'') {
            if (matchContraction(text[start..])) |contr_len| {
                index.* = start + contr_len;
                return text[start..index.*];
            }
        }

        if (isLetterLike(first_cp)) {
            var end = start + first_len;
            while (end < text.len) {
                if (self.matchSpecialToken(text[end..]) != null) break;
                const len = std.unicode.utf8ByteSequenceLength(text[end]) catch break;
                const slice = text[end .. end + len];
                const cp = std.unicode.utf8Decode(slice) catch break;
                if (!isLetterLike(cp)) break;
                end += len;
            }
            index.* = end;
            return text[start..end];
        }

        if (isDigitLike(first_cp)) {
            var end = start + first_len;
            while (end < text.len) {
                if (self.matchSpecialToken(text[end..]) != null) break;
                const len = std.unicode.utf8ByteSequenceLength(text[end]) catch break;
                const slice = text[end .. end + len];
                const cp = std.unicode.utf8Decode(slice) catch break;
                if (!isDigitLike(cp)) break;
                end += len;
            }
            index.* = end;
            return text[start..end];
        }

        var end = start + first_len;
        while (end < text.len) {
            if (self.matchSpecialToken(text[end..]) != null) break;
            const len = std.unicode.utf8ByteSequenceLength(text[end]) catch break;
            const slice = text[end .. end + len];
            const cp = std.unicode.utf8Decode(slice) catch break;
            if (isNewline(cp) or isHorizontalWhitespace(cp) or isLetterLike(cp) or isDigitLike(cp)) break;
            end += len;
        }
        while (end < text.len) {
            if (self.matchSpecialToken(text[end..]) != null) break;
            const len = std.unicode.utf8ByteSequenceLength(text[end]) catch break;
            const slice = text[end .. end + len];
            const cp = std.unicode.utf8Decode(slice) catch break;
            if (!isNewline(cp)) break;
            end += len;
        }
        index.* = end;
        return text[start..end];
    }
};

fn initByteMappings(
    allocator: std.mem.Allocator,
    byte_encoder: *[256][]const u8,
    byte_decoder: *std.AutoHashMapUnmanaged(u21, u8),
) !void {
    var bs = std.ArrayListUnmanaged(u32).empty;
    defer bs.deinit(allocator);
    var cs = std.ArrayListUnmanaged(u32).empty;
    defer cs.deinit(allocator);

    var b: u32 = '!';
    while (b <= '~') : (b += 1) {
        try bs.append(allocator, b);
        try cs.append(allocator, b);
    }
    b = 0xA1;
    while (b <= 0xAC) : (b += 1) {
        try bs.append(allocator, b);
        try cs.append(allocator, b);
    }
    b = 0xAE;
    while (b <= 0xFF) : (b += 1) {
        try bs.append(allocator, b);
        try cs.append(allocator, b);
    }

    var n: u32 = 0;
    for (0..256) |byte_value| {
        if (!containsCode(bs.items, @intCast(byte_value))) {
            try bs.append(allocator, @intCast(byte_value));
            try cs.append(allocator, 256 + n);
            n += 1;
        }
    }

    for (bs.items, cs.items) |byte_value, codepoint| {
        var buffer: [4]u8 = undefined;
        const len = try std.unicode.utf8Encode(@intCast(codepoint), &buffer);
        const string = try allocator.dupe(u8, buffer[0..len]);
        byte_encoder[@intCast(byte_value)] = string;
        try byte_decoder.put(allocator, @intCast(codepoint), @intCast(byte_value));
    }
}

fn containsCode(values: []const u32, needle: u32) bool {
    for (values) |value| {
        if (value == needle) return true;
    }
    return false;
}

fn loadVocab(
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

fn loadMerges(
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

fn loadSpecialTokens(
    allocator: std.mem.Allocator,
    path: []const u8,
    vocab: *std.StringHashMapUnmanaged(u32),
) ![]const SpecialToken {
    const bytes = try std.fs.cwd().readFileAlloc(allocator, path, 4 * 1024 * 1024);
    const root = try std.json.parseFromSliceLeaky(std.json.Value, allocator, bytes, .{});
    if (root != .object) return error.InvalidTokenizerConfig;

    const decoder_value = root.object.get("added_tokens_decoder") orelse return error.InvalidTokenizerConfig;
    if (decoder_value != .object) return error.InvalidTokenizerConfig;

    var tokens = std.ArrayListUnmanaged(SpecialToken).empty;
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

    std.sort.block(SpecialToken, tokens.items, {}, struct {
        fn lessThan(_: void, lhs: SpecialToken, rhs: SpecialToken) bool {
            return lhs.content.len > rhs.content.len;
        }
    }.lessThan);

    return tokens.toOwnedSlice(allocator);
}

fn computeMaxId(vocab: std.StringHashMapUnmanaged(u32), special_tokens: []const SpecialToken) !usize {
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

fn matchContraction(input: []const u8) ?usize {
    const contractions = [_][]const u8{ "'s", "'t", "'re", "'ve", "'m", "'ll", "'d" };
    for (contractions) |item| {
        if (input.len >= item.len and std.ascii.eqlIgnoreCase(input[0..item.len], item)) {
            return item.len;
        }
    }
    return null;
}

fn isHorizontalWhitespace(cp: u21) bool {
    return switch (cp) {
        ' ', '\t' => true,
        else => false,
    };
}

fn isNewline(cp: u21) bool {
    return cp == '\n' or cp == '\r';
}

fn isDigitLike(cp: u21) bool {
    return cp >= '0' and cp <= '9';
}

fn isLetterLike(cp: u21) bool {
    if (cp < 128) return std.ascii.isAlphabetic(@intCast(cp));
    if (cp >= 0x4E00 and cp <= 0x9FFF) return true;
    if (cp >= 0x3400 and cp <= 0x4DBF) return true;
    if (cp >= 0x3040 and cp <= 0x30FF) return true;
    if (cp >= 0xAC00 and cp <= 0xD7AF) return true;
    if (cp >= 0x0400 and cp <= 0x052F) return true;
    if (cp >= 0x00C0 and cp <= 0x024F) return true;
    return !isHorizontalWhitespace(cp) and !isNewline(cp) and !isLikelyPunctuation(cp);
}

fn isLikelyPunctuation(cp: u21) bool {
    return switch (cp) {
        0x2000...0x206F,
        0x3000...0x303F,
        0xFF00...0xFF0F,
        0xFF1A...0xFF20,
        0xFF3B...0xFF40,
        0xFF5B...0xFF65,
        => true,
        else => false,
    };
}

test "qwen tokenizer encodes known samples" {
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

test "qwen tokenizer decodes known samples" {
    const testing = std.testing;
    var tokenizer = try Tokenizer.loadFromModelDir(testing.allocator, "models/Qwen3-0.6B");
    defer tokenizer.deinit();

    const text = try tokenizer.decodeAlloc(testing.allocator, &[_]u32{ 151644, 872, 198, 9707, 151645, 198 });
    defer testing.allocator.free(text);
    try testing.expectEqualStrings("<|im_start|>user\nHello<|im_end|>\n", text);
}
