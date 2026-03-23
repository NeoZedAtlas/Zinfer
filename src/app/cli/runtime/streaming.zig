const std = @import("std");
const GenerateOptions = @import("../args.zig").GenerateOptions;
const decoder_family = @import("../../../model/runtime/decoder_family.zig");

pub const StopAnalysis = struct {
    printable_len: usize,
    stop_hit: bool,
    response_len: usize,
};

pub fn analyzeGeneratedText(text: []const u8, stop_sequences: [][]const u8) StopAnalysis {
    var max_overlap: usize = 0;

    for (stop_sequences) |stop_sequence| {
        if (stop_sequence.len == 0) continue;
        if (std.mem.endsWith(u8, text, stop_sequence)) {
            return .{
                .printable_len = text.len - stop_sequence.len,
                .stop_hit = true,
                .response_len = text.len - stop_sequence.len,
            };
        }

        const max_candidate = @min(text.len, stop_sequence.len - 1);
        var overlap = max_candidate;
        while (overlap > 0) : (overlap -= 1) {
            if (std.mem.eql(u8, text[text.len - overlap ..], stop_sequence[0..overlap])) {
                max_overlap = @max(max_overlap, overlap);
                break;
            }
        }
    }

    return .{
        .printable_len = text.len - max_overlap,
        .stop_hit = false,
        .response_len = text.len,
    };
}

pub fn analyzeAndMaybeStream(
    allocator: std.mem.Allocator,
    tokenizer: *decoder_family.Tokenizer,
    generated_ids: []const u32,
    options: GenerateOptions,
    stdout: anytype,
    streamed_len: *usize,
) !?[]u8 {
    if (!options.stream_output and options.stop_sequences.len == 0) return null;

    const decoded = tokenizer.decodeAlloc(allocator, generated_ids) catch |err| switch (err) {
        error.InvalidWtf8 => return null,
        else => return err,
    };
    defer allocator.free(decoded);

    const analysis = analyzeGeneratedText(decoded, options.stop_sequences);
    if (options.stream_output and analysis.printable_len > streamed_len.*) {
        try stdout.writeAll(decoded[streamed_len.*..analysis.printable_len]);
        streamed_len.* = analysis.printable_len;
    }
    if (!analysis.stop_hit) return null;

    return try allocator.dupe(u8, decoded[0..analysis.response_len]);
}

test "analyzeGeneratedText trims full stop sequence" {
    const testing = std.testing;

    const stops: []const []const u8 = &[_][]const u8{"today?"};
    const analysis = analyzeGeneratedText("Hello today?", @constCast(stops));
    try testing.expect(analysis.stop_hit);
    try testing.expectEqual(@as(usize, 6), analysis.printable_len);
    try testing.expectEqual(@as(usize, 6), analysis.response_len);
}

test "analyzeGeneratedText holds back partial stop prefix" {
    const testing = std.testing;

    const stops: []const []const u8 = &[_][]const u8{"today?"};
    const analysis = analyzeGeneratedText("Hello to", @constCast(stops));
    try testing.expect(!analysis.stop_hit);
    try testing.expectEqual(@as(usize, 6), analysis.printable_len);
    try testing.expectEqual(@as(usize, 8), analysis.response_len);
}
