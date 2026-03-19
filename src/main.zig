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
    _ = @import("model/logits.zig");
    _ = @import("model/gqa_attention.zig");
    _ = @import("model/rmsnorm_gqa_swiglu_block.zig");
    _ = @import("model/adapters/qwen3/block.zig");
    _ = @import("model/adapters/qwen3/runtime.zig");
    _ = @import("model/adapters/qwen3/spec.zig");
    _ = @import("model/adapters/qwen3/weights.zig");
    _ = @import("model/adapters/qwen3/generation_policy.zig");
    _ = @import("model/decoder_family.zig");
    _ = @import("model/decoder_runtime.zig");
    _ = @import("model/weights_layout.zig");
    _ = @import("model/adapters/qwen3/tokenizer.zig");
    _ = @import("model/adapters/qwen3/chat_template.zig");
    _ = @import("sampling/sampler.zig");
}
