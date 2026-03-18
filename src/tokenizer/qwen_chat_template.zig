const std = @import("std");
const qwen_bpe = @import("qwen_bpe.zig");

pub const ThinkingMode = enum {
    enabled,
    disabled,
};

pub fn renderSingleUserPromptAlloc(
    allocator: std.mem.Allocator,
    user_text: []const u8,
    mode: ThinkingMode,
) ![]u8 {
    return switch (mode) {
        .enabled => std.fmt.allocPrint(
            allocator,
            "<|im_start|>user\n{s}<|im_end|>\n<|im_start|>assistant\n",
            .{user_text},
        ),
        .disabled => std.fmt.allocPrint(
            allocator,
            "<|im_start|>user\n{s}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
            .{user_text},
        ),
    };
}

test "single user prompt matches qwen thinking template" {
    const testing = std.testing;
    const prompt = try renderSingleUserPromptAlloc(testing.allocator, "Hello", .enabled);
    defer testing.allocator.free(prompt);

    try testing.expectEqualStrings(
        "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n",
        prompt,
    );

    var tokenizer = try qwen_bpe.Tokenizer.loadFromModelDir(testing.allocator, "models/Qwen3-0.6B");
    defer tokenizer.deinit();

    const ids = try tokenizer.encodeAlloc(testing.allocator, prompt);
    defer testing.allocator.free(ids);
    try testing.expectEqualSlices(u32, &[_]u32{ 151644, 872, 198, 9707, 151645, 198, 151644, 77091, 198 }, ids);
}

test "single user prompt matches qwen non-thinking template" {
    const testing = std.testing;
    const prompt = try renderSingleUserPromptAlloc(testing.allocator, "Hello", .disabled);
    defer testing.allocator.free(prompt);

    try testing.expectEqualStrings(
        "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        prompt,
    );

    var tokenizer = try qwen_bpe.Tokenizer.loadFromModelDir(testing.allocator, "models/Qwen3-0.6B");
    defer tokenizer.deinit();

    const ids = try tokenizer.encodeAlloc(testing.allocator, prompt);
    defer testing.allocator.free(ids);
    try testing.expectEqualSlices(u32, &[_]u32{ 151644, 872, 198, 9707, 151645, 198, 151644, 77091, 198, 151667, 271, 151668, 271 }, ids);
}
