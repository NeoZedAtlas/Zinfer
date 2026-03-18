const std = @import("std");
const safetensors = @import("format/safetensors.zig");
const kv_cache = @import("model/kv_cache.zig");
const qwen3_block = @import("model/qwen3_block.zig");
const qwen3_model = @import("model/qwen3_model.zig");
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

    if (std.mem.eql(u8, command, "probe-block")) {
        if (args.len == 2) {
            try probeBlock(allocator, default_model_dir, 0, 0, 8);
            return;
        }
        if (args.len == 3) {
            const layer_index = try std.fmt.parseInt(usize, args[2], 10);
            try probeBlock(allocator, default_model_dir, layer_index, 0, 8);
            return;
        }
        if (args.len == 4) {
            const layer_index = try std.fmt.parseInt(usize, args[2], 10);
            const input_index = try std.fmt.parseInt(usize, args[3], 10);
            try probeBlock(allocator, default_model_dir, layer_index, input_index, 8);
            return;
        }
        if (args.len == 5) {
            const layer_index = try std.fmt.parseInt(usize, args[2], 10);
            const input_index = try std.fmt.parseInt(usize, args[3], 10);
            const count = try std.fmt.parseInt(usize, args[4], 10);
            try probeBlock(allocator, default_model_dir, layer_index, input_index, count);
            return;
        }
        if (args.len >= 6) {
            const layer_index = try std.fmt.parseInt(usize, args[3], 10);
            const input_index = try std.fmt.parseInt(usize, args[4], 10);
            const count = try std.fmt.parseInt(usize, args[5], 10);
            try probeBlock(allocator, args[2], layer_index, input_index, count);
            return;
        }
        try printUsage();
        return error.InvalidCommand;
    }

    if (std.mem.eql(u8, command, "probe-model")) {
        if (args.len == 2) {
            try probeModel(allocator, default_model_dir, 0, 8);
            return;
        }
        if (args.len == 3) {
            const token_id = try std.fmt.parseInt(usize, args[2], 10);
            try probeModel(allocator, default_model_dir, token_id, 8);
            return;
        }
        if (args.len == 4) {
            const token_id = try std.fmt.parseInt(usize, args[2], 10);
            const top_k = try std.fmt.parseInt(usize, args[3], 10);
            try probeModel(allocator, default_model_dir, token_id, top_k);
            return;
        }
        if (args.len >= 5) {
            const token_id = try std.fmt.parseInt(usize, args[3], 10);
            const top_k = try std.fmt.parseInt(usize, args[4], 10);
            try probeModel(allocator, args[2], token_id, top_k);
            return;
        }
        try printUsage();
        return error.InvalidCommand;
    }

    if (std.mem.eql(u8, command, "generate-token-ids")) {
        if (args.len == 2) {
            try generateTokenIds(allocator, default_model_dir, "0", 5);
            return;
        }
        if (args.len == 3) {
            try generateTokenIds(allocator, default_model_dir, args[2], 5);
            return;
        }
        if (args.len == 4) {
            const steps = try std.fmt.parseInt(usize, args[3], 10);
            try generateTokenIds(allocator, default_model_dir, args[2], steps);
            return;
        }
        if (args.len >= 5) {
            const steps = try std.fmt.parseInt(usize, args[4], 10);
            try generateTokenIds(allocator, args[2], args[3], steps);
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
        \\  zinfer probe-block [layer_index] [input_index] [count]
        \\  zinfer probe-block [model_dir] <layer_index> <input_index> <count>
        \\  zinfer probe-model [token_id] [top_k]
        \\  zinfer probe-model [model_dir] <token_id> <top_k>
        \\  zinfer generate-token-ids [seed_ids_csv] [steps]
        \\  zinfer generate-token-ids [model_dir] <seed_ids_csv> <steps>
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

fn probeBlock(
    allocator: std.mem.Allocator,
    model_dir: []const u8,
    layer_index: usize,
    input_index: usize,
    count: usize,
) !void {
    const config_path = try std.fs.path.join(allocator, &.{ model_dir, "config.json" });
    defer allocator.free(config_path);
    var parsed_config = try qwen3_config.loadFromFile(allocator, config_path);
    defer parsed_config.deinit();
    const cfg = parsed_config.value;

    if (input_index >= cfg.hidden_size) return error.InputIndexOutOfBounds;

    const weights_path = try std.fs.path.join(allocator, &.{ model_dir, "model.safetensors" });
    defer allocator.free(weights_path);

    var store = try tensor_store.TensorStore.open(allocator, weights_path);
    defer store.deinit();

    var cache = try kv_cache.LayerKVCache.init(
        allocator,
        1,
        cfg.num_key_value_heads,
        cfg.head_dim,
    );
    defer cache.deinit();

    const hidden_in = try allocator.alloc(f32, cfg.hidden_size);
    defer allocator.free(hidden_in);
    @memset(hidden_in, 0.0);
    hidden_in[input_index] = 1.0;

    const hidden_out = try allocator.alloc(f32, cfg.hidden_size);
    defer allocator.free(hidden_out);

    const spec = qwen3_block.Qwen3BlockSpec{
        .layer_index = layer_index,
        .hidden_size = cfg.hidden_size,
        .intermediate_size = cfg.intermediate_size,
        .num_attention_heads = cfg.num_attention_heads,
        .num_key_value_heads = cfg.num_key_value_heads,
        .head_dim = cfg.head_dim,
        .rope_theta = @floatCast(cfg.rope_theta),
        .rms_norm_eps = @floatCast(cfg.rms_norm_eps),
    };
    try qwen3_block.forwardSingleToken(allocator, &store, spec, &cache, hidden_in, hidden_out);

    const stdout = std.fs.File.stdout().deprecatedWriter();
    try stdout.print("Zinfer block probe\n", .{});
    try stdout.print("layer_index: {d}\n", .{layer_index});
    try stdout.print("input_index: {d}\n", .{input_index});
    try stdout.print("cache_len: {d}\n", .{cache.len});
    try stdout.print("first_outputs: [", .{});
    const limit = @min(count, hidden_out.len);
    for (hidden_out[0..limit], 0..) |value, idx| {
        if (idx != 0) try stdout.print(", ", .{});
        try stdout.print("{d:.6}", .{value});
    }
    try stdout.print("]\n", .{});
}

fn probeModel(
    allocator: std.mem.Allocator,
    model_dir: []const u8,
    token_id: usize,
    top_k: usize,
) !void {
    const config_path = try std.fs.path.join(allocator, &.{ model_dir, "config.json" });
    defer allocator.free(config_path);
    var parsed_config = try qwen3_config.loadFromFile(allocator, config_path);
    defer parsed_config.deinit();
    const cfg = parsed_config.value;

    const weights_path = try std.fs.path.join(allocator, &.{ model_dir, "model.safetensors" });
    defer allocator.free(weights_path);

    var store = try tensor_store.TensorStore.open(allocator, weights_path);
    defer store.deinit();

    var cache = try qwen3_model.ModelCache.init(
        allocator,
        cfg.num_hidden_layers,
        1,
        cfg.num_key_value_heads,
        cfg.head_dim,
    );
    defer cache.deinit();

    const logits = try qwen3_model.forwardTokenId(allocator, &store, cfg, &cache, token_id);
    defer allocator.free(logits);
    const top = try qwen3_model.topKLogitsAlloc(allocator, logits, top_k);
    defer allocator.free(top);

    const stdout = std.fs.File.stdout().deprecatedWriter();
    try stdout.print("Zinfer model probe\n", .{});
    try stdout.print("token_id: {d}\n", .{token_id});
    for (top) |entry| {
        try stdout.print("top_logit token={d} value={d:.6}\n", .{ entry.token_id, entry.logit });
    }
}

fn generateTokenIds(
    allocator: std.mem.Allocator,
    model_dir: []const u8,
    seed_ids_csv: []const u8,
    steps: usize,
) !void {
    const seed_ids = try parseTokenIdsAlloc(allocator, seed_ids_csv);
    defer allocator.free(seed_ids);
    if (seed_ids.len == 0) return error.EmptySeed;

    const config_path = try std.fs.path.join(allocator, &.{ model_dir, "config.json" });
    defer allocator.free(config_path);
    var parsed_config = try qwen3_config.loadFromFile(allocator, config_path);
    defer parsed_config.deinit();
    const cfg = parsed_config.value;

    const weights_path = try std.fs.path.join(allocator, &.{ model_dir, "model.safetensors" });
    defer allocator.free(weights_path);

    var store = try tensor_store.TensorStore.open(allocator, weights_path);
    defer store.deinit();

    var cache = try qwen3_model.ModelCache.init(
        allocator,
        cfg.num_hidden_layers,
        seed_ids.len + steps,
        cfg.num_key_value_heads,
        cfg.head_dim,
    );
    defer cache.deinit();

    const generated = try allocator.alloc(usize, seed_ids.len + steps);
    defer allocator.free(generated);
    @memcpy(generated[0..seed_ids.len], seed_ids);
    var generated_len = seed_ids.len;

    var last_token = seed_ids[0];
    var last_logits: ?[]f32 = null;
    for (seed_ids) |token_id| {
        const logits = try qwen3_model.forwardTokenId(allocator, &store, cfg, &cache, token_id);
        if (last_logits) |buffer| allocator.free(buffer);
        last_logits = logits;
        last_token = token_id;
    }
    defer if (last_logits) |buffer| allocator.free(buffer);

    for (0..steps) |_| {
        const current_logits = last_logits orelse return error.MissingPromptLogits;
        const next_token = try qwen3_model.argMaxLogit(current_logits);
        generated[generated_len] = next_token;
        generated_len += 1;
        last_token = next_token;
        allocator.free(current_logits);
        last_logits = try qwen3_model.forwardTokenId(allocator, &store, cfg, &cache, last_token);
    }

    const stdout = std.fs.File.stdout().deprecatedWriter();
    try stdout.print("Zinfer generated token ids\n", .{});
    try stdout.print("seed: {s}\n", .{seed_ids_csv});
    try stdout.print("steps: {d}\n", .{steps});
    try stdout.print("tokens: [", .{});
    for (generated[0..generated_len], 0..) |token_id, idx| {
        if (idx != 0) try stdout.print(", ", .{});
        try stdout.print("{d}", .{token_id});
    }
    try stdout.print("]\n", .{});
}

fn parseTokenIdsAlloc(
    allocator: std.mem.Allocator,
    csv: []const u8,
) ![]usize {
    var list: std.ArrayListUnmanaged(usize) = .empty;
    defer list.deinit(allocator);

    var it = std.mem.splitScalar(u8, csv, ',');
    while (it.next()) |part| {
        const trimmed = std.mem.trim(u8, part, " ");
        if (trimmed.len == 0) continue;
        try list.append(allocator, try std.fmt.parseInt(usize, trimmed, 10));
    }

    return list.toOwnedSlice(allocator);
}
