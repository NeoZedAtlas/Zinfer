const std = @import("std");
const cli = @import("app/cli.zig");

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
    _ = @import("model/runtime/kv_cache.zig");
    _ = @import("model/runtime/decoder_cache.zig");
    _ = @import("model/layers/decoder_only_stack.zig");
    _ = @import("model/runtime/chat_types.zig");
    _ = @import("model/runtime/decoder_registry.zig");
    _ = @import("model/layers/logits.zig");
    _ = @import("model/layers/gqa_attention.zig");
    _ = @import("model/layers/rmsnorm_gqa_swiglu_block.zig");
    _ = @import("model/families/qwen3/weights.zig");
    _ = @import("model/families/qwen3/layout.zig");
    _ = @import("model/families/qwen3/family.zig");
    _ = @import("model/families/qwen3/generation_policy.zig");
    _ = @import("model/runtime/decoder_types.zig");
    _ = @import("model/runtime/decoder_family.zig");
    _ = @import("model/runtime/optimized_decoder.zig");
    _ = @import("model/runtime/optimized_kv_cache.zig");
    _ = @import("model/layers/weights_layout.zig");
    _ = @import("model/families/qwen3/chat_template.zig");
    _ = @import("tensor/backend.zig");
    _ = @import("tokenizer/bpe.zig");
    _ = @import("tensor/quantized.zig");
    _ = @import("sampling/sampler.zig");
}
