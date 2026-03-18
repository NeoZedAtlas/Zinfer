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
    _ = @import("model/qwen3_attention.zig");
}
