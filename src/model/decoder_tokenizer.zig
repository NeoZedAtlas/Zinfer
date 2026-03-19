const std = @import("std");
const decoder_runtime = @import("decoder_runtime.zig");
const qwen_bpe = @import("../tokenizer/qwen_bpe.zig");

pub const Tokenizer = union(decoder_runtime.Architecture) {
    qwen3: qwen_bpe.Tokenizer,

    pub fn loadFromModelDir(
        backing_allocator: std.mem.Allocator,
        architecture: decoder_runtime.Architecture,
        model_dir: []const u8,
    ) !Tokenizer {
        return try entryForArchitecture(architecture).load_from_model_dir(backing_allocator, model_dir);
    }

    pub fn deinit(self: *Tokenizer) void {
        switch (self.*) {
            inline else => |*tokenizer| tokenizer.deinit(),
        }
    }

    pub fn encodeAlloc(self: *const Tokenizer, allocator: std.mem.Allocator, text: []const u8) ![]u32 {
        return switch (self.*) {
            inline else => |*tokenizer| tokenizer.encodeAlloc(allocator, text),
        };
    }

    pub fn decodeAlloc(self: *const Tokenizer, allocator: std.mem.Allocator, ids: []const u32) ![]u8 {
        return switch (self.*) {
            inline else => |*tokenizer| tokenizer.decodeAlloc(allocator, ids),
        };
    }
};

const RegistryEntry = struct {
    load_from_model_dir: *const fn (std.mem.Allocator, []const u8) anyerror!Tokenizer,
};

fn entryForArchitecture(architecture: decoder_runtime.Architecture) RegistryEntry {
    return switch (architecture) {
        .qwen3 => .{
            .load_from_model_dir = loadQwen3TokenizerFromModelDir,
        },
    };
}

fn loadQwen3TokenizerFromModelDir(
    backing_allocator: std.mem.Allocator,
    model_dir: []const u8,
) !Tokenizer {
    return .{
        .qwen3 = try qwen_bpe.Tokenizer.loadFromModelDir(backing_allocator, model_dir),
    };
}

test "decoder tokenizer loads qwen3 and roundtrips prompt text" {
    const testing = std.testing;

    var tokenizer = try Tokenizer.loadFromModelDir(testing.allocator, .qwen3, "models/Qwen3-0.6B");
    defer tokenizer.deinit();

    const ids = try tokenizer.encodeAlloc(testing.allocator, "<|im_start|>user\nHello<|im_end|>\n");
    defer testing.allocator.free(ids);
    try testing.expectEqualSlices(u32, &[_]u32{ 151644, 872, 198, 9707, 151645, 198 }, ids);

    const text = try tokenizer.decodeAlloc(testing.allocator, ids);
    defer testing.allocator.free(text);
    try testing.expectEqualStrings("<|im_start|>user\nHello<|im_end|>\n", text);
}
