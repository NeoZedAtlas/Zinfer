const std = @import("std");
const decoder_family = @import("../decoder_family.zig");
const tensor_backend = @import("../../../tensor/backends/backend.zig");
const weights_layout = @import("../../layers/weights_layout.zig");

pub fn allocVector(
    backend: *tensor_backend.Backend,
    allocator: std.mem.Allocator,
    name: []const u8,
    len: usize,
    io_scratch_bytes: usize,
) ![]f32 {
    const output = try allocator.alloc(f32, len);
    errdefer allocator.free(output);
    const scratch = try allocator.alloc(u8, io_scratch_bytes);
    defer allocator.free(scratch);
    try backend.readVectorInto(name, output, scratch);
    return output;
}

pub fn resolveMatrixTensor(
    backend: *tensor_backend.Backend,
    allocator: std.mem.Allocator,
    cfg: decoder_family.DecoderConfig,
    layer_index: usize,
    kind: weights_layout.LayerTensorKind,
) !tensor_backend.Backend.TensorHandle {
    const name = try decoder_family.layerTensorNameAlloc(allocator, cfg.architecture, layer_index, kind);
    defer allocator.free(name);
    return try backend.resolveTensor(name);
}

pub fn maxIoScratchBytes(cfg: decoder_family.DecoderConfig) usize {
    return @max(cfg.hidden_size, cfg.intermediate_size) * 4;
}
