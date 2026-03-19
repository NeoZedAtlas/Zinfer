const std = @import("std");

pub const eos_token_ids: []const u32 = &[_]u32{ 151645, 151643 };
pub const default_stop_sequences: []const []const u8 = &[_][]const u8{
    "<|im_end|>",
};

pub fn isEosToken(token_id: usize) bool {
    for (eos_token_ids) |eos_id| {
        if (token_id == eos_id) return true;
    }
    return false;
}

test "adapter generation policy recognizes eos tokens" {
    const testing = std.testing;
    try testing.expect(isEosToken(151645));
    try testing.expect(isEosToken(151643));
    try testing.expect(!isEosToken(42));
}
