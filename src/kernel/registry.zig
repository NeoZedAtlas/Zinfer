const types = @import("registry/types.zig");
const resolve_mod = @import("registry/resolve.zig");

pub const ShapeTag = types.ShapeTag;
pub const GemvOp = types.GemvOp;
pub const AttentionQ8Layout = types.AttentionQ8Layout;
pub const KernelBackend = types.KernelBackend;
pub const IsaTag = types.IsaTag;
pub const KernelSpec = types.KernelSpec;
pub const Entry = types.Entry;

pub const shapeForWidth = types.shapeForWidth;
pub const backendForGemvOp = types.backendForGemvOp;
pub const activeIsa = types.activeIsa;

pub const resolve = resolve_mod.resolve;
pub const resolveGemvRow = resolve_mod.resolveGemvRow;
pub const resolveAttentionQ8Decode = resolve_mod.resolveAttentionQ8Decode;

const std = @import("std");

test "registry resolves hot gemv row as specialized entry" {
    const entry = resolve(.{ .gemv_row = .{ .op = .q6_row, .cols = 3072 } });
    try std.testing.expectEqualStrings("gemv_q6_3072", entry.name);
    try std.testing.expectEqual(ShapeTag.qwen3_intermediate_3072, entry.shape);
    try std.testing.expectEqual(KernelBackend.q6, entry.backend);
    try std.testing.expect(entry.specialized);
    try std.testing.expect(entry.layout == null);
}

test "registry preserves attention layout and backend" {
    const entry = resolve(.{ .attention_q8_decode = .{
        .head_dim = 128,
        .layout = .head_major,
    } });
    try std.testing.expectEqualStrings("attn_q8_decode_head_major_128", entry.name);
    try std.testing.expectEqual(ShapeTag.qwen3_head_dim_128, entry.shape);
    try std.testing.expectEqual(KernelBackend.q8, entry.backend);
    try std.testing.expectEqual(AttentionQ8Layout.head_major, entry.layout.?);
    try std.testing.expect(entry.specialized);
}
