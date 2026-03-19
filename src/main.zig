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
    _ = @import("model/decoder_cache.zig");
    _ = @import("model/decoder_only_stack.zig");
    _ = @import("model/chat_types.zig");
    _ = @import("model/decoder_registry.zig");
    _ = @import("model/logits.zig");
    _ = @import("model/gqa_attention.zig");
    _ = @import("model/rmsnorm_gqa_swiglu_block.zig");
    _ = @import("model/adapters/qwen3/weights.zig");
    _ = @import("model/adapters/qwen3/layout.zig");
    _ = @import("model/adapters/qwen3/family.zig");
    _ = @import("model/adapters/qwen3/generation_policy.zig");
    _ = @import("model/decoder_types.zig");
    _ = @import("model/decoder_family.zig");
    _ = @import("model/weights_layout.zig");
    _ = @import("model/adapters/qwen3/chat_template.zig");
    _ = @import("tokenizer/bpe.zig");
    _ = @import("sampling/sampler.zig");
}
