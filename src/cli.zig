const std = @import("std");
const safetensors = @import("format/safetensors.zig");
const qwen3_config = @import("model/qwen3_config.zig");
const tensor_store = @import("tensor/store.zig");

const default_model_dir = "models/Qwen3-0.6B";

pub fn run(allocator: std.mem.Allocator) !void {
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len == 1) {
        try inspectConfig(allocator, default_model_dir);
        return;
    }

    const command = args[1];
    if (std.mem.eql(u8, command, "inspect-config")) {
        const model_dir = if (args.len >= 3) args[2] else default_model_dir;
        try inspectConfig(allocator, model_dir);
        return;
    }

    if (std.mem.eql(u8, command, "inspect-weights")) {
        const model_dir = if (args.len >= 3) args[2] else default_model_dir;
        try inspectWeights(allocator, model_dir);
        return;
    }

    if (std.mem.eql(u8, command, "inspect-tensor")) {
        if (args.len == 3) {
            try inspectTensor(allocator, default_model_dir, args[2], 8);
            return;
        }
        if (args.len >= 4) {
            const count = if (args.len >= 5) try std.fmt.parseInt(usize, args[4], 10) else 8;
            try inspectTensor(allocator, args[2], args[3], count);
            return;
        }
        try printUsage();
        return error.InvalidCommand;
    }

    if (std.mem.eql(u8, command, "probe-linear")) {
        if (args.len == 3) {
            try probeLinear(allocator, default_model_dir, args[2], 0, 8);
            return;
        }
        if (args.len == 4) {
            const input_index = try std.fmt.parseInt(usize, args[3], 10);
            try probeLinear(allocator, default_model_dir, args[2], input_index, 8);
            return;
        }
        if (args.len == 5) {
            const input_index = try std.fmt.parseInt(usize, args[3], 10);
            const rows_to_print = try std.fmt.parseInt(usize, args[4], 10);
            try probeLinear(allocator, default_model_dir, args[2], input_index, rows_to_print);
            return;
        }
        if (args.len >= 6) {
            const input_index = try std.fmt.parseInt(usize, args[4], 10);
            const rows_to_print = try std.fmt.parseInt(usize, args[5], 10);
            try probeLinear(allocator, args[2], args[3], input_index, rows_to_print);
            return;
        }
        try printUsage();
        return error.InvalidCommand;
    }

    if (std.mem.eql(u8, command, "--help") or std.mem.eql(u8, command, "-h")) {
        try printUsage();
        return;
    }

    std.log.err("unknown command: {s}", .{command});
    try printUsage();
    return error.InvalidCommand;
}

fn inspectConfig(allocator: std.mem.Allocator, model_dir: []const u8) !void {
    const config_path = try std.fs.path.join(allocator, &.{ model_dir, "config.json" });
    defer allocator.free(config_path);

    var parsed = try qwen3_config.loadFromFile(allocator, config_path);
    defer parsed.deinit();

    const cfg = parsed.value;
    const stdout = std.fs.File.stdout().deprecatedWriter();

    try stdout.print("Zinfer model inspection\n", .{});
    try stdout.print("model_dir: {s}\n", .{model_dir});
    try stdout.print("model_type: {s}\n", .{cfg.model_type});
    try stdout.print("architectures[0]: {s}\n", .{cfg.firstArchitecture() orelse "<none>"});
    try stdout.print("hidden_size: {d}\n", .{cfg.hidden_size});
    try stdout.print("intermediate_size: {d}\n", .{cfg.intermediate_size});
    try stdout.print("num_hidden_layers: {d}\n", .{cfg.num_hidden_layers});
    try stdout.print("num_attention_heads: {d}\n", .{cfg.num_attention_heads});
    try stdout.print("num_key_value_heads: {d}\n", .{cfg.num_key_value_heads});
    try stdout.print("head_dim: {d}\n", .{cfg.head_dim});
    try stdout.print("vocab_size: {d}\n", .{cfg.vocab_size});
    try stdout.print("max_position_embeddings: {d}\n", .{cfg.max_position_embeddings});
    try stdout.print("rope_theta: {d}\n", .{@as(u64, @intFromFloat(cfg.rope_theta))});
    try stdout.print("rms_norm_eps: {e}\n", .{cfg.rms_norm_eps});
    try stdout.print("torch_dtype: {s}\n", .{cfg.torch_dtype});
    try stdout.print("tie_word_embeddings: {any}\n", .{cfg.tie_word_embeddings});
}

fn printUsage() !void {
    const stdout = std.fs.File.stdout().deprecatedWriter();
    try stdout.writeAll(
        \\Usage:
        \\  zinfer
        \\  zinfer inspect-config [model_dir]
        \\  zinfer inspect-weights [model_dir]
        \\  zinfer inspect-tensor <tensor_name>
        \\  zinfer inspect-tensor [model_dir] <tensor_name> [count]
        \\  zinfer probe-linear <tensor_name> [input_index] [rows_to_print]
        \\  zinfer probe-linear [model_dir] <tensor_name> <input_index> <rows_to_print>
        \\
        \\Defaults:
        \\  model_dir = models/Qwen3-0.6B
        \\
    );
}

fn inspectWeights(allocator: std.mem.Allocator, model_dir: []const u8) !void {
    const weights_path = try std.fs.path.join(allocator, &.{ model_dir, "model.safetensors" });
    defer allocator.free(weights_path);

    var parsed = try safetensors.loadFromFile(allocator, weights_path);
    defer parsed.deinit();

    const stdout = std.fs.File.stdout().deprecatedWriter();
    try stdout.print("Zinfer weights inspection\n", .{});
    try stdout.print("weights_path: {s}\n", .{weights_path});
    try stdout.print("file_size: {d}\n", .{parsed.file_size});
    try stdout.print("header_len: {d}\n", .{parsed.header_len});
    try stdout.print("data_start: {d}\n", .{parsed.data_start});
    try stdout.print("tensor_count: {d}\n", .{parsed.tensorCount()});

    if (parsed.metadata.count() > 0) {
        var metadata_it = parsed.metadata.iterator();
        while (metadata_it.next()) |entry| {
            try stdout.print("metadata.{s}: {s}\n", .{ entry.key_ptr.*, entry.value_ptr.* });
        }
    }

    const sample_names = [_][]const u8{
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.norm.weight",
        "lm_head.weight",
    };

    for (sample_names) |name| {
        if (parsed.getTensor(name)) |tensor| {
            try printTensorSummary(stdout, name, tensor);
        }
    }
}

fn printTensorSummary(
    stdout: anytype,
    name: []const u8,
    tensor: safetensors.TensorInfo,
) !void {
    try stdout.print("tensor: {s}\n", .{name});
    try stdout.print("  dtype: {s}\n", .{tensor.dtype.name()});
    try stdout.print("  rank: {d}\n", .{tensor.rank()});
    try stdout.print("  shape: [", .{});
    for (tensor.shape, 0..) |dim, idx| {
        if (idx != 0) try stdout.print(", ", .{});
        try stdout.print("{d}", .{dim});
    }
    try stdout.print("]\n", .{});
    try stdout.print("  data_offsets: [{d}, {d}]\n", .{ tensor.data_offsets[0], tensor.data_offsets[1] });
    try stdout.print("  absolute_offset: {d}\n", .{tensor.absolute_offset});
    try stdout.print("  byte_len: {d}\n", .{tensor.byteLen()});
}

fn inspectTensor(
    allocator: std.mem.Allocator,
    model_dir: []const u8,
    tensor_name: []const u8,
    count: usize,
) !void {
    const weights_path = try std.fs.path.join(allocator, &.{ model_dir, "model.safetensors" });
    defer allocator.free(weights_path);

    var store = try tensor_store.TensorStore.open(allocator, weights_path);
    defer store.deinit();

    const tensor = store.getTensor(tensor_name) orelse return error.TensorNotFound;
    const values = try store.readElementsAsF32Alloc(tensor_name, 0, count);
    defer allocator.free(values);

    const stdout = std.fs.File.stdout().deprecatedWriter();
    try stdout.print("Zinfer tensor inspection\n", .{});
    try stdout.print("tensor: {s}\n", .{tensor_name});
    try stdout.print("weights_path: {s}\n", .{weights_path});
    try stdout.print("dtype: {s}\n", .{tensor.dtype.name()});
    try stdout.print("shape: [", .{});
    for (tensor.shape, 0..) |dim, idx| {
        if (idx != 0) try stdout.print(", ", .{});
        try stdout.print("{d}", .{dim});
    }
    try stdout.print("]\n", .{});
    try stdout.print("first_values: [", .{});
    for (values, 0..) |value, idx| {
        if (idx != 0) try stdout.print(", ", .{});
        try stdout.print("{d:.6}", .{value});
    }
    try stdout.print("]\n", .{});
}

fn probeLinear(
    allocator: std.mem.Allocator,
    model_dir: []const u8,
    tensor_name: []const u8,
    input_index: usize,
    rows_to_print: usize,
) !void {
    const weights_path = try std.fs.path.join(allocator, &.{ model_dir, "model.safetensors" });
    defer allocator.free(weights_path);

    var store = try tensor_store.TensorStore.open(allocator, weights_path);
    defer store.deinit();

    const tensor = store.getTensor(tensor_name) orelse return error.TensorNotFound;
    if (tensor.rank() != 2) return error.InvalidTensorRank;

    const rows = std.math.cast(usize, tensor.shape[0]) orelse return error.DimensionTooLarge;
    const cols = std.math.cast(usize, tensor.shape[1]) orelse return error.DimensionTooLarge;
    if (input_index >= cols) return error.InputIndexOutOfBounds;

    const input = try allocator.alloc(f32, cols);
    defer allocator.free(input);
    @memset(input, 0.0);
    input[input_index] = 1.0;

    const output = try allocator.alloc(f32, rows);
    defer allocator.free(output);
    try store.matmulVecByName(output, tensor_name, input);

    const stdout = std.fs.File.stdout().deprecatedWriter();
    try stdout.print("Zinfer linear probe\n", .{});
    try stdout.print("tensor: {s}\n", .{tensor_name});
    try stdout.print("weights_path: {s}\n", .{weights_path});
    try stdout.print("shape: [{d}, {d}]\n", .{ rows, cols });
    try stdout.print("input_index: {d}\n", .{input_index});
    try stdout.print("first_outputs: [", .{});
    const limit = @min(rows_to_print, output.len);
    for (output[0..limit], 0..) |value, idx| {
        if (idx != 0) try stdout.print(", ", .{});
        try stdout.print("{d:.6}", .{value});
    }
    try stdout.print("]\n", .{});
}
