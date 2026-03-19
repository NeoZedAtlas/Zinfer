const std = @import("std");
const decoder_family = @import("decoder_family.zig");

pub const Tokenizer = decoder_family.Tokenizer;

test "decoder tokenizer loads qwen3 and roundtrips prompt text" {
    const testing = std.testing;

    var tokenizer = try decoder_family.loadTokenizerFromModelDir(testing.allocator, .qwen3, "models/Qwen3-0.6B");
    defer tokenizer.deinit();

    const ids = try tokenizer.encodeAlloc(testing.allocator, "<|im_start|>user\nHello<|im_end|>\n");
    defer testing.allocator.free(ids);
    try testing.expectEqualSlices(u32, &[_]u32{ 151644, 872, 198, 9707, 151645, 198 }, ids);

    const text = try tokenizer.decodeAlloc(testing.allocator, ids);
    defer testing.allocator.free(text);
    try testing.expectEqualStrings("<|im_start|>user\nHello<|im_end|>\n", text);
}
