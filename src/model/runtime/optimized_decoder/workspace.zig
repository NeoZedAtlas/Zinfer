const std = @import("std");
const attention = @import("../../../kernel/attention/attention.zig");
const decoder_family = @import("../decoder_family.zig");

pub const Workspace = struct {
    allocator: std.mem.Allocator,
    hidden_a: []f32,
    hidden_b: []f32,
    normed: []f32,
    q_proj: []f32,
    k_proj: []f32,
    v_proj: []f32,
    attn_flat: []f32,
    scores: []f32,
    attn_out: []f32,
    post_attn: []f32,
    gate: []f32,
    up: []f32,
    mlp_out: []f32,
    logits: []f32,
    io_scratch: []u8,
    rope_table: attention.RoPETable,

    pub fn init(
        allocator: std.mem.Allocator,
        cfg: decoder_family.DecoderConfig,
        max_seq_len: usize,
        io_scratch_bytes: usize,
    ) !Workspace {
        const kv_width = cfg.num_key_value_heads * cfg.head_dim;
        var rope_table = try attention.RoPETable.init(
            allocator,
            max_seq_len,
            cfg.head_dim,
            @floatCast(cfg.rope_theta),
        );
        errdefer rope_table.deinit();
        return .{
            .allocator = allocator,
            .hidden_a = try allocator.alloc(f32, cfg.hidden_size),
            .hidden_b = try allocator.alloc(f32, cfg.hidden_size),
            .normed = try allocator.alloc(f32, cfg.hidden_size),
            .q_proj = try allocator.alloc(f32, cfg.num_attention_heads * cfg.head_dim),
            .k_proj = try allocator.alloc(f32, kv_width),
            .v_proj = try allocator.alloc(f32, kv_width),
            .attn_flat = try allocator.alloc(f32, cfg.num_attention_heads * cfg.head_dim),
            .scores = try allocator.alloc(f32, max_seq_len),
            .attn_out = try allocator.alloc(f32, cfg.hidden_size),
            .post_attn = try allocator.alloc(f32, cfg.hidden_size),
            .gate = try allocator.alloc(f32, cfg.intermediate_size),
            .up = try allocator.alloc(f32, cfg.intermediate_size),
            .mlp_out = try allocator.alloc(f32, cfg.hidden_size),
            .logits = try allocator.alloc(f32, cfg.vocab_size),
            .io_scratch = try allocator.alloc(u8, io_scratch_bytes),
            .rope_table = rope_table,
        };
    }

    pub fn deinit(self: *Workspace) void {
        self.allocator.free(self.hidden_a);
        self.allocator.free(self.hidden_b);
        self.allocator.free(self.normed);
        self.allocator.free(self.q_proj);
        self.allocator.free(self.k_proj);
        self.allocator.free(self.v_proj);
        self.allocator.free(self.attn_flat);
        self.allocator.free(self.scores);
        self.allocator.free(self.attn_out);
        self.allocator.free(self.post_attn);
        self.allocator.free(self.gate);
        self.allocator.free(self.up);
        self.allocator.free(self.mlp_out);
        self.allocator.free(self.logits);
        self.allocator.free(self.io_scratch);
        self.rope_table.deinit();
    }
};
