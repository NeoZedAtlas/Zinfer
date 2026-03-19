const std = @import("std");
const decoder_family = @import("decoder_family.zig");
const tensor_store = @import("../tensor/store.zig");

pub const Architecture = decoder_family.Architecture;
pub const DecoderConfig = decoder_family.DecoderConfig;
pub const ParsedConfig = decoder_family.ParsedConfig;
pub const ModelCache = decoder_family.ModelCache;
pub const TopLogit = decoder_family.TopLogit;

pub fn loadConfigFromFile(backing_allocator: std.mem.Allocator, path: []const u8) !ParsedConfig {
    return try decoder_family.loadConfigFromFile(backing_allocator, path);
}

pub fn forwardTokenId(
    allocator: std.mem.Allocator,
    store: *const tensor_store.TensorStore,
    cfg: DecoderConfig,
    cache: *ModelCache,
    token_id: usize,
) ![]f32 {
    return try decoder_family.forwardTokenId(allocator, store, cfg, cache, token_id);
}

pub fn prefillTokenIds(
    allocator: std.mem.Allocator,
    store: *const tensor_store.TensorStore,
    cfg: DecoderConfig,
    cache: *ModelCache,
    token_ids: []const usize,
) ![]f32 {
    return try decoder_family.prefillTokenIds(allocator, store, cfg, cache, token_ids);
}

pub fn topKLogitsAlloc(
    allocator: std.mem.Allocator,
    cfg: DecoderConfig,
    logits: []const f32,
    k: usize,
) ![]TopLogit {
    return try decoder_family.topKLogitsAlloc(allocator, cfg.architecture, logits, k);
}

pub fn argMaxLogit(cfg: DecoderConfig, logits: []const f32) !usize {
    return try decoder_family.argMaxLogit(cfg.architecture, logits);
}

pub fn isEosToken(cfg: DecoderConfig, token_id: usize) bool {
    return decoder_family.isEosToken(cfg.architecture, token_id);
}

pub fn effectiveStopSequencesAlloc(
    allocator: std.mem.Allocator,
    cfg: DecoderConfig,
    extra_stop_sequences: [][]const u8,
) ![][]const u8 {
    return try decoder_family.effectiveStopSequencesAlloc(allocator, cfg.architecture, extra_stop_sequences);
}

test "decoder config loads qwen3 architecture" {
    const testing = std.testing;

    var parsed = try loadConfigFromFile(testing.allocator, "models/Qwen3-0.6B/config.json");
    defer parsed.deinit();

    try testing.expectEqual(Architecture.qwen3, parsed.value.architecture);
    try testing.expectEqualStrings("qwen3", parsed.value.model_type);
}
