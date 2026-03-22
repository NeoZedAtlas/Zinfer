pub const ShapeTag = enum {
    generic,
    qwen3_hidden_1024,
    qwen3_intermediate_3072,
    qwen3_head_dim_128,
};

pub const GemvOp = enum {
    f32_row,
    bf16_row,
    q8_row,
    q6_row,
    q4_row,
};

pub const AttentionQ8Layout = enum {
    token_major,
    head_major,
};

pub const KernelSpec = union(enum) {
    gemv_row: struct {
        op: GemvOp,
        cols: usize,
    },
    attention_q8_decode: struct {
        head_dim: usize,
        layout: AttentionQ8Layout,
    },
};

pub const Entry = struct {
    name: []const u8,
    shape: ShapeTag,
};

pub fn resolve(spec: KernelSpec) Entry {
    return switch (spec) {
        .gemv_row => |row| resolveGemvRow(row.op, row.cols),
        .attention_q8_decode => |attn| resolveAttentionQ8Decode(attn.head_dim, attn.layout),
    };
}

pub fn resolveGemvRow(op: GemvOp, cols: usize) Entry {
    return .{
        .name = gemvName(op, cols),
        .shape = shapeForWidth(cols),
    };
}

pub fn resolveAttentionQ8Decode(head_dim: usize, layout: AttentionQ8Layout) Entry {
    return .{
        .name = switch (layout) {
            .token_major => switch (shapeForWidth(head_dim)) {
                .qwen3_head_dim_128 => "attn_q8_decode_token_major_128",
                else => "attn_q8_decode_token_major_generic",
            },
            .head_major => switch (shapeForWidth(head_dim)) {
                .qwen3_head_dim_128 => "attn_q8_decode_head_major_128",
                else => "attn_q8_decode_head_major_generic",
            },
        },
        .shape = shapeForWidth(head_dim),
    };
}

pub fn shapeForWidth(width: usize) ShapeTag {
    return switch (width) {
        1024 => .qwen3_hidden_1024,
        3072 => .qwen3_intermediate_3072,
        128 => .qwen3_head_dim_128,
        else => .generic,
    };
}

fn gemvName(op: GemvOp, cols: usize) []const u8 {
    return switch (op) {
        .f32_row => switch (shapeForWidth(cols)) {
            .qwen3_hidden_1024 => "gemv_f32_1024",
            .qwen3_intermediate_3072 => "gemv_f32_3072",
            else => "gemv_f32_generic",
        },
        .bf16_row => switch (shapeForWidth(cols)) {
            .qwen3_hidden_1024 => "gemv_bf16_1024",
            .qwen3_intermediate_3072 => "gemv_bf16_3072",
            else => "gemv_bf16_generic",
        },
        .q8_row => switch (shapeForWidth(cols)) {
            .qwen3_hidden_1024 => "gemv_q8_1024",
            .qwen3_intermediate_3072 => "gemv_q8_3072",
            else => "gemv_q8_generic",
        },
        .q6_row => switch (shapeForWidth(cols)) {
            .qwen3_hidden_1024 => "gemv_q6_1024",
            .qwen3_intermediate_3072 => "gemv_q6_3072",
            else => "gemv_q6_generic",
        },
        .q4_row => switch (shapeForWidth(cols)) {
            .qwen3_hidden_1024 => "gemv_q4_1024",
            .qwen3_intermediate_3072 => "gemv_q4_3072",
            else => "gemv_q4_generic",
        },
    };
}
