const std = @import("std");
const qwen3_config = @import("model/qwen3_config.zig");

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
        \\
        \\Defaults:
        \\  model_dir = models/Qwen3-0.6B
        \\
    );
}
