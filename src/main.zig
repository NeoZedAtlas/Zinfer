const std = @import("std");
const cli = @import("cli.zig");

pub fn main() !void {
    var gpa_state = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa_state.deinit();

    const gpa = gpa_state.allocator();
    try cli.run(gpa);
}

test {
    std.testing.refAllDecls(@This());
    _ = @import("kernel/cpu.zig");
    _ = @import("kernel/attention.zig");
    _ = @import("model/kv_cache.zig");
    _ = @import("model/qwen3_block.zig");
    _ = @import("model/qwen3_model.zig");
    _ = @import("model/qwen3_attention.zig");
    _ = @import("model/decoder_runtime.zig");
    _ = @import("model/decoder_chat.zig");
    _ = @import("model/decoder_tokenizer.zig");
    _ = @import("tokenizer/qwen_bpe.zig");
    _ = @import("tokenizer/qwen_chat_template.zig");
    _ = @import("sampling/sampler.zig");
}
