const std = @import("std");
const types = @import("types.zig");

pub fn matchSpecialToken(special_tokens: []const types.SpecialToken, input: []const u8) ?types.SpecialToken {
    var best: ?types.SpecialToken = null;
    for (special_tokens) |token| {
        if (input.len < token.content.len) continue;
        if (!std.mem.eql(u8, input[0..token.content.len], token.content)) continue;
        if (best == null or token.content.len > best.?.content.len) {
            best = token;
        }
    }
    return best;
}

pub fn nextSegment(special_tokens: []const types.SpecialToken, text: []const u8, index: *usize) ?[]const u8 {
    if (index.* >= text.len) return null;

    const start = index.*;
    const first_len = std.unicode.utf8ByteSequenceLength(text[start]) catch return null;
    const first_slice = text[start .. start + first_len];
    const first_cp = std.unicode.utf8Decode(first_slice) catch return null;

    if (isNewline(first_cp)) {
        var end = start + first_len;
        while (end < text.len) {
            if (matchSpecialToken(special_tokens, text[end..]) != null) break;
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
            if (matchSpecialToken(special_tokens, text[end..]) != null) break;
            const len = std.unicode.utf8ByteSequenceLength(text[end]) catch break;
            const slice = text[end .. end + len];
            const cp = std.unicode.utf8Decode(slice) catch break;
            if (!isHorizontalWhitespace(cp)) break;
            end += len;
        }
        if (end < text.len) {
            if (matchSpecialToken(special_tokens, text[end..]) != null) {
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
                        if (matchSpecialToken(special_tokens, text[word_end..]) != null) break;
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
                        if (matchSpecialToken(special_tokens, text[punct_end..]) != null) break;
                        const next_len = std.unicode.utf8ByteSequenceLength(text[punct_end]) catch break;
                        const next_slice = text[punct_end .. punct_end + next_len];
                        const next_cp = std.unicode.utf8Decode(next_slice) catch break;
                        if (isDigitLike(next_cp) or isLetterLike(next_cp) or isNewline(next_cp) or isHorizontalWhitespace(next_cp)) break;
                        punct_end += next_len;
                    }
                    while (punct_end < text.len) {
                        if (matchSpecialToken(special_tokens, text[punct_end..]) != null) break;
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
            if (matchSpecialToken(special_tokens, text[end..]) != null) break;
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
            if (matchSpecialToken(special_tokens, text[end..]) != null) break;
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
        if (matchSpecialToken(special_tokens, text[end..]) != null) break;
        const len = std.unicode.utf8ByteSequenceLength(text[end]) catch break;
        const slice = text[end .. end + len];
        const cp = std.unicode.utf8Decode(slice) catch break;
        if (isNewline(cp) or isHorizontalWhitespace(cp) or isLetterLike(cp) or isDigitLike(cp)) break;
        end += len;
    }
    while (end < text.len) {
        if (matchSpecialToken(special_tokens, text[end..]) != null) break;
        const len = std.unicode.utf8ByteSequenceLength(text[end]) catch break;
        const slice = text[end .. end + len];
        const cp = std.unicode.utf8Decode(slice) catch break;
        if (!isNewline(cp)) break;
        end += len;
    }
    index.* = end;
    return text[start..end];
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
