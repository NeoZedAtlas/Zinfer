const builtin = @import("builtin");

pub const ShapeTag = enum {
    generic,
    qwen3_hidden_1024,
    qwen3_intermediate_3072,
    qwen3_head_dim_128,

    pub fn name(self: ShapeTag) []const u8 {
        return switch (self) {
            .generic => "generic",
            .qwen3_hidden_1024 => "qwen3_hidden_1024",
            .qwen3_intermediate_3072 => "qwen3_intermediate_3072",
            .qwen3_head_dim_128 => "qwen3_head_dim_128",
        };
    }
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
    paged_head_major,

    pub fn name(self: AttentionQ8Layout) []const u8 {
        return switch (self) {
            .token_major => "token_major",
            .head_major => "head_major",
            .paged_head_major => "paged_head_major",
        };
    }
};

pub const KernelBackend = enum {
    f32,
    bf16,
    q8,
    q6,
    q4,

    pub fn name(self: KernelBackend) []const u8 {
        return switch (self) {
            .f32 => "f32",
            .bf16 => "bf16",
            .q8 => "q8",
            .q6 => "q6",
            .q4 => "q4",
        };
    }
};

pub const IsaTag = enum {
    generic,
    x86_64,
    aarch64,

    pub fn name(self: IsaTag) []const u8 {
        return switch (self) {
            .generic => "generic",
            .x86_64 => "x86_64",
            .aarch64 => "aarch64",
        };
    }
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
    backend: KernelBackend,
    layout: ?AttentionQ8Layout,
    isa: IsaTag,
    specialized: bool,
};

pub fn shapeForWidth(width: usize) ShapeTag {
    return switch (width) {
        1024 => .qwen3_hidden_1024,
        3072 => .qwen3_intermediate_3072,
        128 => .qwen3_head_dim_128,
        else => .generic,
    };
}

pub fn backendForGemvOp(op: GemvOp) KernelBackend {
    return switch (op) {
        .f32_row => .f32,
        .bf16_row => .bf16,
        .q8_row => .q8,
        .q6_row => .q6,
        .q4_row => .q4,
    };
}

pub fn activeIsa() IsaTag {
    return switch (builtin.target.cpu.arch) {
        .x86, .x86_64 => .x86_64,
        .aarch64 => .aarch64,
        else => .generic,
    };
}
