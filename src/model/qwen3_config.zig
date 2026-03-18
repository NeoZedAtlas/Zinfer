const std = @import("std");

pub const Qwen3Config = struct {
    architectures: []const []const u8 = &.{},
    head_dim: usize,
    hidden_size: usize,
    intermediate_size: usize,
    max_position_embeddings: usize,
    model_type: []const u8,
    num_attention_heads: usize,
    num_hidden_layers: usize,
    num_key_value_heads: usize,
    rms_norm_eps: f64,
    rope_theta: f64,
    tie_word_embeddings: bool,
    torch_dtype: []const u8,
    vocab_size: usize,

    pub fn firstArchitecture(self: Qwen3Config) ?[]const u8 {
        if (self.architectures.len == 0) return null;
        return self.architectures[0];
    }
};

pub const ParsedConfig = struct {
    arena: std.heap.ArenaAllocator,
    value: Qwen3Config,

    pub fn deinit(self: *ParsedConfig) void {
        self.arena.deinit();
    }
};

pub fn loadFromFile(backing_allocator: std.mem.Allocator, path: []const u8) !ParsedConfig {
    var arena = std.heap.ArenaAllocator.init(backing_allocator);
    errdefer arena.deinit();

    const allocator = arena.allocator();
    const bytes = try std.fs.cwd().readFileAlloc(allocator, path, 1024 * 1024);
    const config = try std.json.parseFromSliceLeaky(Qwen3Config, allocator, bytes, .{
        .ignore_unknown_fields = true,
    });

    return .{
        .arena = arena,
        .value = config,
    };
}
