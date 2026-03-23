const types = @import("types.zig");

pub fn resolve(spec: types.KernelSpec) types.Entry {
    return switch (spec) {
        .gemv_row => |row| resolveGemvRow(row.op, row.cols),
        .attention_q8_decode => |attn| resolveAttentionQ8Decode(attn.head_dim, attn.layout),
    };
}

pub fn resolveGemvRow(op: types.GemvOp, cols: usize) types.Entry {
    const shape = types.shapeForWidth(cols);
    return .{
        .name = gemvName(op, shape),
        .shape = shape,
        .backend = types.backendForGemvOp(op),
        .layout = null,
        .isa = types.activeIsa(),
        .specialized = shape != .generic,
    };
}

pub fn resolveAttentionQ8Decode(head_dim: usize, layout: types.AttentionQ8Layout) types.Entry {
    const shape = types.shapeForWidth(head_dim);
    return .{
        .name = attentionQ8Name(shape, layout),
        .shape = shape,
        .backend = .q8,
        .layout = layout,
        .isa = types.activeIsa(),
        .specialized = shape != .generic,
    };
}

fn gemvName(op: types.GemvOp, shape: types.ShapeTag) []const u8 {
    return switch (op) {
        .f32_row => switch (shape) {
            .qwen3_hidden_1024 => "gemv_f32_1024",
            .qwen3_intermediate_3072 => "gemv_f32_3072",
            else => "gemv_f32_generic",
        },
        .bf16_row => switch (shape) {
            .qwen3_hidden_1024 => "gemv_bf16_1024",
            .qwen3_intermediate_3072 => "gemv_bf16_3072",
            else => "gemv_bf16_generic",
        },
        .q8_row => switch (shape) {
            .qwen3_hidden_1024 => "gemv_q8_1024",
            .qwen3_intermediate_3072 => "gemv_q8_3072",
            else => "gemv_q8_generic",
        },
        .q6_row => switch (shape) {
            .qwen3_hidden_1024 => "gemv_q6_1024",
            .qwen3_intermediate_3072 => "gemv_q6_3072",
            else => "gemv_q6_generic",
        },
        .q4_row => switch (shape) {
            .qwen3_hidden_1024 => "gemv_q4_1024",
            .qwen3_intermediate_3072 => "gemv_q4_3072",
            else => "gemv_q4_generic",
        },
    };
}

fn attentionQ8Name(shape: types.ShapeTag, layout: types.AttentionQ8Layout) []const u8 {
    return switch (layout) {
        .token_major => switch (shape) {
            .qwen3_head_dim_128 => "attn_q8_decode_token_major_128",
            else => "attn_q8_decode_token_major_generic",
        },
        .head_major => switch (shape) {
            .qwen3_head_dim_128 => "attn_q8_decode_head_major_128",
            else => "attn_q8_decode_head_major_generic",
        },
        .paged_head_major => switch (shape) {
            .qwen3_head_dim_128 => "attn_q8_decode_paged_head_major_128",
            else => "attn_q8_decode_paged_head_major_generic",
        },
    };
}
