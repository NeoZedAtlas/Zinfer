const std = @import("std");
const safetensors = @import("format/safetensors.zig");
const kv_cache = @import("model/kv_cache.zig");
const optimized_kv_cache = @import("model/optimized_kv_cache.zig");
const decoder_family = @import("model/decoder_family.zig");
const optimized_decoder = @import("model/optimized_decoder.zig");
const tensor_backend = @import("tensor/backend.zig");
const quantized = @import("tensor/quantized.zig");
const tensor_store = @import("tensor/store.zig");
const sampler = @import("sampling/sampler.zig");

const default_model_dir = "models/Qwen3-0.6B";

const GenerateOptions = struct {
    max_new_tokens: usize,
    thinking_mode: decoder_family.ThinkingMode,
    system_prompt: ?[]const u8,
    sampling: sampler.SamplingConfig,
    seed: u64,
    stream_output: bool,
    stop_sequences: [][]const u8,
    backend_scheme: tensor_backend.Scheme,
    thread_count: usize,

    fn deinit(self: *GenerateOptions, allocator: std.mem.Allocator) void {
        allocator.free(self.stop_sequences);
    }
};

const ParsedGenerateInvocation = struct {
    model_dir: []const u8,
    user_text: []const u8,
    options: GenerateOptions,

    fn deinit(self: *ParsedGenerateInvocation, allocator: std.mem.Allocator) void {
        self.options.deinit(allocator);
    }
};

const ParsedGenerateChatInvocation = struct {
    model_dir: []const u8,
    messages_json_path: []const u8,
    options: GenerateOptions,

    fn deinit(self: *ParsedGenerateChatInvocation, allocator: std.mem.Allocator) void {
        self.options.deinit(allocator);
    }
};

const ParsedChatInvocation = struct {
    model_dir: []const u8,
    options: GenerateOptions,
    load_path: ?[]const u8,
    save_path: ?[]const u8,

    fn deinit(self: *ParsedChatInvocation, allocator: std.mem.Allocator) void {
        self.options.deinit(allocator);
    }
};

const ParsedBenchInvocation = struct {
    model_dir: []const u8,
    user_text: []const u8,
    options: GenerateOptions,

    fn deinit(self: *ParsedBenchInvocation, allocator: std.mem.Allocator) void {
        self.options.deinit(allocator);
    }
};

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

    if (std.mem.eql(u8, command, "bench")) {
        var invocation = try parseBenchInvocation(allocator, args);
        defer invocation.deinit(allocator);
        try benchPrompt(allocator, invocation.model_dir, invocation.user_text, invocation.options);
        return;
    }

    if (std.mem.eql(u8, command, "quantize")) {
        if (args.len == 3) {
            try quantizeModelDir(allocator, default_model_dir, args[2]);
            return;
        }
        if (args.len >= 4) {
            try quantizeModelDir(allocator, args[3], args[2]);
            return;
        }
        try printUsage();
        return error.InvalidCommand;
    }

    if (std.mem.eql(u8, command, "tokenize")) {
        if (args.len == 3) {
            try tokenizeText(allocator, default_model_dir, args[2]);
            return;
        }
        if (args.len >= 4) {
            try tokenizeText(allocator, args[2], args[3]);
            return;
        }
        try printUsage();
        return error.InvalidCommand;
    }

    if (std.mem.eql(u8, command, "decode-ids")) {
        if (args.len == 3) {
            try decodeIds(allocator, default_model_dir, args[2]);
            return;
        }
        if (args.len >= 4) {
            try decodeIds(allocator, args[2], args[3]);
            return;
        }
        try printUsage();
        return error.InvalidCommand;
    }

    if (std.mem.eql(u8, command, "generate")) {
        var invocation = try parseGenerateInvocation(allocator, args);
        defer invocation.deinit(allocator);
        try generateText(allocator, invocation.model_dir, invocation.user_text, invocation.options);
        return;
    }

    if (std.mem.eql(u8, command, "generate-chat")) {
        var invocation = try parseGenerateChatInvocation(allocator, args);
        defer invocation.deinit(allocator);
        try generateChatFromFile(allocator, invocation.model_dir, invocation.messages_json_path, invocation.options);
        return;
    }

    if (std.mem.eql(u8, command, "chat")) {
        var invocation = try parseChatInvocation(allocator, args);
        defer invocation.deinit(allocator);
        try chatLoop(
            allocator,
            invocation.model_dir,
            invocation.options,
            invocation.load_path,
            invocation.save_path,
        );
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

    var parsed = try decoder_family.loadConfigFromFile(allocator, config_path);
    defer parsed.deinit();

    const cfg = parsed.value;
    const stdout = std.fs.File.stdout().deprecatedWriter();

    try stdout.print("Zinfer model inspection\n", .{});
    try stdout.print("model_dir: {s}\n", .{model_dir});
    try stdout.print("model_type: {s}\n", .{cfg.model_type});
    try stdout.print("architecture: {s}\n", .{cfg.architecture.name()});
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
        \\  zinfer bench <text> [max_new_tokens]
        \\  zinfer bench [model_dir] <text> <max_new_tokens>
        \\  zinfer quantize <q8|q4>
        \\  zinfer quantize <q8|q4> [model_dir]
        \\  zinfer tokenize <text>
        \\  zinfer tokenize [model_dir] <text>
        \\  zinfer decode-ids <ids_csv>
        \\  zinfer decode-ids [model_dir] <ids_csv>
        \\  zinfer generate <text> [max_new_tokens] [think|no-think] [flags...]
        \\  zinfer generate [model_dir] <text> <max_new_tokens> [think|no-think] [flags...]
        \\  zinfer generate-chat <messages_json_path> [max_new_tokens] [think|no-think] [flags...]
        \\  zinfer generate-chat [model_dir] <messages_json_path> <max_new_tokens> [think|no-think] [flags...]
        \\  zinfer chat [max_new_tokens] [think|no-think] [flags...]
        \\  zinfer chat [model_dir] [max_new_tokens] [think|no-think] [flags...]
        \\
        \\Defaults:
        \\  model_dir = models/Qwen3-0.6B
        \\  generate max_new_tokens = 64
        \\  generate-chat/chat max_new_tokens = 128
        \\
        \\Flags:
        \\  --system <text>
        \\  --seed <u64>
        \\  --temperature <f32>
        \\  --top-p <f32>
        \\  --top-k <usize>
        \\  --min-p <f32>
        \\  --presence-penalty <f32>
        \\  --frequency-penalty <f32>
        \\  --repetition-penalty <f32>
        \\  --stop <text>           (repeatable)
        \\  --backend <auto|bf16|q8|q4>
        \\  --threads <usize>       (0 = auto)
        \\  --stream
        \\  --load <path>           (chat only)
        \\  --save <path>           (chat only)
        \\
    );
}

fn parseGenerateInvocation(
    allocator: std.mem.Allocator,
    args: []const []const u8,
) !ParsedGenerateInvocation {
    if (args.len < 3) return error.InvalidCommand;

    var model_dir: []const u8 = default_model_dir;
    var user_text: []const u8 = undefined;
    var start_flags: usize = undefined;
    var options = initGenerateOptions(.disabled, 64);

    if (args.len >= 5) {
        if (std.fmt.parseInt(usize, args[4], 10)) |max_new_tokens| {
            model_dir = args[2];
            user_text = args[3];
            options.max_new_tokens = max_new_tokens;
            start_flags = 5;
        } else |_| {
            model_dir = default_model_dir;
            user_text = args[2];
            start_flags = 3;
            if (args.len > start_flags and !isFlagArg(args[start_flags])) {
                options.max_new_tokens = try std.fmt.parseInt(usize, args[start_flags], 10);
                start_flags += 1;
            }
        }
    } else {
        model_dir = default_model_dir;
        user_text = args[2];
        start_flags = 3;
        if (args.len > start_flags and !isFlagArg(args[start_flags])) {
            options.max_new_tokens = try std.fmt.parseInt(usize, args[start_flags], 10);
            start_flags += 1;
        }
    }

    if (args.len > start_flags and !isFlagArg(args[start_flags])) {
        options.thinking_mode = try parseThinkingMode(args[start_flags]);
        start_flags += 1;
    }
    options.sampling = defaultSamplingConfig(options.thinking_mode);
    try parseGenerateFlags(allocator, args[start_flags..], &options);

    return .{
        .model_dir = model_dir,
        .user_text = user_text,
        .options = options,
    };
}

fn parseGenerateChatInvocation(
    allocator: std.mem.Allocator,
    args: []const []const u8,
) !ParsedGenerateChatInvocation {
    if (args.len < 3) return error.InvalidCommand;

    var model_dir: []const u8 = default_model_dir;
    var messages_json_path: []const u8 = undefined;
    var start_flags: usize = undefined;
    var options = initGenerateOptions(.disabled, 128);

    if (args.len >= 5) {
        if (std.fmt.parseInt(usize, args[4], 10)) |max_new_tokens| {
            model_dir = args[2];
            messages_json_path = args[3];
            options.max_new_tokens = max_new_tokens;
            start_flags = 5;
        } else |_| {
            messages_json_path = args[2];
            start_flags = 3;
            if (args.len > start_flags and !isFlagArg(args[start_flags])) {
                options.max_new_tokens = try std.fmt.parseInt(usize, args[start_flags], 10);
                start_flags += 1;
            }
        }
    } else {
        messages_json_path = args[2];
        start_flags = 3;
        if (args.len > start_flags and !isFlagArg(args[start_flags])) {
            options.max_new_tokens = try std.fmt.parseInt(usize, args[start_flags], 10);
            start_flags += 1;
        }
    }

    if (args.len > start_flags and !isFlagArg(args[start_flags])) {
        options.thinking_mode = try parseThinkingMode(args[start_flags]);
        start_flags += 1;
    }
    options.sampling = defaultSamplingConfig(options.thinking_mode);
    try parseGenerateFlags(allocator, args[start_flags..], &options);

    return .{
        .model_dir = model_dir,
        .messages_json_path = messages_json_path,
        .options = options,
    };
}

fn parseChatInvocation(
    allocator: std.mem.Allocator,
    args: []const []const u8,
) !ParsedChatInvocation {
    var model_dir: []const u8 = default_model_dir;
    var start_flags: usize = 2;
    var options = initGenerateOptions(.disabled, 128);
    var load_path: ?[]const u8 = null;
    var save_path: ?[]const u8 = null;

    if (args.len > 2 and !isFlagArg(args[2])) {
        if (std.fmt.parseInt(usize, args[2], 10)) |max_new_tokens| {
            options.max_new_tokens = max_new_tokens;
            start_flags = 3;
        } else |_| {
            model_dir = args[2];
            start_flags = 3;
            if (args.len > 3 and !isFlagArg(args[3])) {
                options.max_new_tokens = try std.fmt.parseInt(usize, args[3], 10);
                start_flags = 4;
            }
        }
    }

    if (args.len > start_flags and !isFlagArg(args[start_flags])) {
        options.thinking_mode = try parseThinkingMode(args[start_flags]);
        start_flags += 1;
    }
    options.sampling = defaultSamplingConfig(options.thinking_mode);
    try parseChatFlags(allocator, args[start_flags..], &options, &load_path, &save_path);

    return .{
        .model_dir = model_dir,
        .options = options,
        .load_path = load_path,
        .save_path = save_path,
    };
}

fn parseBenchInvocation(
    allocator: std.mem.Allocator,
    args: []const []const u8,
) !ParsedBenchInvocation {
    if (args.len < 3) return error.InvalidCommand;

    var model_dir: []const u8 = default_model_dir;
    var user_text: []const u8 = undefined;
    var start_flags: usize = undefined;
    var options = initGenerateOptions(.disabled, 16);

    if (args.len >= 5) {
        if (std.fmt.parseInt(usize, args[4], 10)) |max_new_tokens| {
            model_dir = args[2];
            user_text = args[3];
            options.max_new_tokens = max_new_tokens;
            start_flags = 5;
        } else |_| {
            model_dir = default_model_dir;
            user_text = args[2];
            start_flags = 3;
            if (args.len > start_flags and !isFlagArg(args[start_flags])) {
                options.max_new_tokens = try std.fmt.parseInt(usize, args[start_flags], 10);
                start_flags += 1;
            }
        }
    } else {
        model_dir = default_model_dir;
        user_text = args[2];
        start_flags = 3;
        if (args.len > start_flags and !isFlagArg(args[start_flags])) {
            options.max_new_tokens = try std.fmt.parseInt(usize, args[start_flags], 10);
            start_flags += 1;
        }
    }

    try parseGenerateFlags(allocator, args[start_flags..], &options);
    return .{
        .model_dir = model_dir,
        .user_text = user_text,
        .options = options,
    };
}

fn initGenerateOptions(mode: decoder_family.ThinkingMode, max_new_tokens: usize) GenerateOptions {
    return .{
        .max_new_tokens = max_new_tokens,
        .thinking_mode = mode,
        .system_prompt = null,
        .sampling = defaultSamplingConfig(mode),
        .seed = 0,
        .stream_output = false,
        .stop_sequences = &.{},
        .backend_scheme = .auto,
        .thread_count = 0,
    };
}

fn defaultSamplingConfig(mode: decoder_family.ThinkingMode) sampler.SamplingConfig {
    return switch (mode) {
        .enabled => .{
            .temperature = 0.6,
            .top_k = 20,
            .top_p = 0.95,
            .min_p = 0.0,
            .presence_penalty = 0.0,
            .frequency_penalty = 0.0,
            .repetition_penalty = 1.1,
        },
        .disabled => .{
            .temperature = 0.7,
            .top_k = 20,
            .top_p = 0.8,
            .min_p = 0.0,
            .presence_penalty = 0.0,
            .frequency_penalty = 0.0,
            .repetition_penalty = 1.1,
        },
    };
}

fn isFlagArg(arg: []const u8) bool {
    return std.mem.startsWith(u8, arg, "--");
}

fn parseGenerateFlags(
    allocator: std.mem.Allocator,
    args: []const []const u8,
    options: *GenerateOptions,
) !void {
    var stop_sequences = std.ArrayListUnmanaged([]const u8).empty;
    errdefer stop_sequences.deinit(allocator);

    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (!isFlagArg(arg)) return error.InvalidCommand;

        if (std.mem.eql(u8, arg, "--system")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            options.system_prompt = args[i];
            continue;
        }
        if (std.mem.eql(u8, arg, "--seed")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            options.seed = try std.fmt.parseInt(u64, args[i], 10);
            continue;
        }
        if (std.mem.eql(u8, arg, "--temperature")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            options.sampling.temperature = try std.fmt.parseFloat(f32, args[i]);
            continue;
        }
        if (std.mem.eql(u8, arg, "--top-p")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            options.sampling.top_p = try std.fmt.parseFloat(f32, args[i]);
            continue;
        }
        if (std.mem.eql(u8, arg, "--top-k")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            options.sampling.top_k = try std.fmt.parseInt(usize, args[i], 10);
            continue;
        }
        if (std.mem.eql(u8, arg, "--min-p")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            options.sampling.min_p = try std.fmt.parseFloat(f32, args[i]);
            continue;
        }
        if (std.mem.eql(u8, arg, "--presence-penalty")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            options.sampling.presence_penalty = try std.fmt.parseFloat(f32, args[i]);
            continue;
        }
        if (std.mem.eql(u8, arg, "--frequency-penalty")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            options.sampling.frequency_penalty = try std.fmt.parseFloat(f32, args[i]);
            continue;
        }
        if (std.mem.eql(u8, arg, "--repetition-penalty")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            options.sampling.repetition_penalty = try std.fmt.parseFloat(f32, args[i]);
            continue;
        }
        if (std.mem.eql(u8, arg, "--stop")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            try stop_sequences.append(allocator, args[i]);
            continue;
        }
        if (std.mem.eql(u8, arg, "--backend")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            options.backend_scheme = try parseBackendScheme(args[i]);
            continue;
        }
        if (std.mem.eql(u8, arg, "--threads")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            options.thread_count = try std.fmt.parseInt(usize, args[i], 10);
            continue;
        }
        if (std.mem.eql(u8, arg, "--stream")) {
            options.stream_output = true;
            continue;
        }

        return error.UnknownFlag;
    }

    options.stop_sequences = try stop_sequences.toOwnedSlice(allocator);
}

fn parseBackendScheme(text: []const u8) !tensor_backend.Scheme {
    if (std.mem.eql(u8, text, "auto")) return .auto;
    if (std.mem.eql(u8, text, "bf16")) return .bf16;
    if (std.mem.eql(u8, text, "q8")) return .q8;
    if (std.mem.eql(u8, text, "q4")) return .q4;
    return error.InvalidBackendScheme;
}

fn parseChatFlags(
    allocator: std.mem.Allocator,
    args: []const []const u8,
    options: *GenerateOptions,
    load_path: *?[]const u8,
    save_path: *?[]const u8,
) !void {
    var filtered = std.ArrayListUnmanaged([]const u8).empty;
    defer filtered.deinit(allocator);

    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "--load")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            load_path.* = args[i];
            continue;
        }
        if (std.mem.eql(u8, arg, "--save")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            save_path.* = args[i];
            continue;
        }
        try filtered.append(allocator, arg);
    }

    try parseGenerateFlags(allocator, filtered.items, options);
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
    var parsed_config = try decoder_family.loadConfigFromFile(allocator, config_path);
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

    try decoder_family.forwardSingleBlock(allocator, &store, cfg, layer_index, &cache, hidden_in, hidden_out);

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
    var parsed_config = try decoder_family.loadConfigFromFile(allocator, config_path);
    defer parsed_config.deinit();
    const cfg = parsed_config.value;

    const weights_path = try std.fs.path.join(allocator, &.{ model_dir, "model.safetensors" });
    defer allocator.free(weights_path);

    var store = try tensor_store.TensorStore.open(allocator, weights_path);
    defer store.deinit();

    var cache = try decoder_family.ModelCache.init(
        allocator,
        cfg.num_hidden_layers,
        1,
        cfg.num_key_value_heads,
        cfg.head_dim,
    );
    defer cache.deinit();

    const logits = try decoder_family.forwardTokenId(allocator, &store, cfg, &cache, token_id);
    defer allocator.free(logits);
    const top = try decoder_family.topKLogitsAlloc(allocator, logits, top_k);
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
    var parsed_config = try decoder_family.loadConfigFromFile(allocator, config_path);
    defer parsed_config.deinit();
    const cfg = parsed_config.value;

    const weights_path = try std.fs.path.join(allocator, &.{ model_dir, "model.safetensors" });
    defer allocator.free(weights_path);

    var store = try tensor_store.TensorStore.open(allocator, weights_path);
    defer store.deinit();

    var cache = try decoder_family.ModelCache.init(
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
        const logits = try decoder_family.forwardTokenId(allocator, &store, cfg, &cache, token_id);
        if (last_logits) |buffer| allocator.free(buffer);
        last_logits = logits;
        last_token = token_id;
    }
    defer if (last_logits) |buffer| allocator.free(buffer);

    for (0..steps) |_| {
        const current_logits = last_logits orelse return error.MissingPromptLogits;
        const next_token = try decoder_family.argMaxLogit(current_logits);
        generated[generated_len] = next_token;
        generated_len += 1;
        last_token = next_token;
        allocator.free(current_logits);
        last_logits = try decoder_family.forwardTokenId(allocator, &store, cfg, &cache, last_token);
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

fn benchPrompt(
    allocator: std.mem.Allocator,
    model_dir: []const u8,
    user_text: []const u8,
    options: GenerateOptions,
) !void {
    var runtime = try GeneratorRuntime.init(allocator, model_dir, options.backend_scheme, options.thread_count);
    defer runtime.deinit();
    const cfg = runtime.model.cfg;

    const prompt = try buildSingleUserPromptAlloc(
        allocator,
        cfg.architecture,
        user_text,
        options.system_prompt,
        options.thinking_mode,
    );
    defer allocator.free(prompt);

    var tokenize_timer = try std.time.Timer.start();
    const prompt_ids_u32 = try runtime.tokenizer.encodeAlloc(allocator, prompt);
    defer allocator.free(prompt_ids_u32);
    const tokenize_ns = tokenize_timer.read();
    if (prompt_ids_u32.len == 0) return error.EmptyPrompt;

    const prompt_ids = try allocator.alloc(usize, prompt_ids_u32.len);
    defer allocator.free(prompt_ids);
    for (prompt_ids_u32, 0..) |token_id, idx| {
        prompt_ids[idx] = token_id;
    }

    var cache = try optimized_kv_cache.ModelCache.init(
        allocator,
        cfg.num_hidden_layers,
        prompt_ids.len + options.max_new_tokens,
        cfg.num_key_value_heads,
        cfg.head_dim,
    );
    defer cache.deinit();
    var workspace = try runtime.model.initWorkspace(prompt_ids.len + options.max_new_tokens);
    defer workspace.deinit();

    var prefill_timer = try std.time.Timer.start();
    const last_logits = try runtime.model.prefillTokenIds(&workspace, &cache, prompt_ids);
    const prefill_ns = prefill_timer.read();

    var decode_timer = try std.time.Timer.start();
    var decoded_tokens: usize = 0;
    var current_logits = last_logits;
    for (0..options.max_new_tokens) |_| {
        const next_token = try decoder_family.argMaxLogit(current_logits);
        if (decoder_family.isEosToken(cfg.architecture, next_token)) {
            break;
        }

        current_logits = try runtime.model.forwardTokenId(&workspace, &cache, next_token);
        decoded_tokens += 1;
    }
    const decode_ns = decode_timer.read();

    const weights_size = try weightArtifactSize(allocator, model_dir, runtime.model.backendName());
    const kv_cache_bytes = estimateKvCacheBytes(cfg, prompt_ids.len + options.max_new_tokens, true);
    const stdout = std.fs.File.stdout().deprecatedWriter();
    try stdout.print("Zinfer benchmark\n", .{});
    try stdout.print("model_dir: {s}\n", .{model_dir});
    try stdout.print("backend: {s}\n", .{runtime.model.backendName()});
    try stdout.print("threads: {d}\n", .{runtime.model.thread_count});
    try stdout.print("prompt_tokens: {d}\n", .{prompt_ids.len});
    try stdout.print("decode_tokens: {d}\n", .{decoded_tokens});
    try stdout.print("tokenize_ms: {d:.3}\n", .{nsToMs(tokenize_ns)});
    try stdout.print("prefill_ms: {d:.3}\n", .{nsToMs(prefill_ns)});
    try stdout.print("decode_ms: {d:.3}\n", .{nsToMs(decode_ns)});
    try stdout.print("prefill_tok_s: {d:.3}\n", .{tokensPerSecond(prompt_ids.len, prefill_ns)});
    try stdout.print("decode_tok_s: {d:.3}\n", .{tokensPerSecond(decoded_tokens, decode_ns)});
    try stdout.print("weights_bytes: {d}\n", .{weights_size});
    try stdout.print("weights_mib: {d:.3}\n", .{bytesToMiB(weights_size)});
    try stdout.print("kv_cache_bytes: {d}\n", .{kv_cache_bytes});
    try stdout.print("kv_cache_mib: {d:.3}\n", .{bytesToMiB(kv_cache_bytes)});
}

fn quantizeModelDir(
    allocator: std.mem.Allocator,
    model_dir: []const u8,
    scheme_text: []const u8,
) !void {
    const scheme: quantized.Scheme = if (std.mem.eql(u8, scheme_text, "q8"))
        .q8
    else if (std.mem.eql(u8, scheme_text, "q4"))
        .q4
    else
        return error.InvalidQuantizationScheme;

    const input_path = try std.fs.path.join(allocator, &.{ model_dir, "model.safetensors" });
    defer allocator.free(input_path);
    const output_path = try std.fs.path.join(allocator, &.{ model_dir, scheme.fileName() });
    defer allocator.free(output_path);

    const stdout = std.fs.File.stdout().deprecatedWriter();
    try stdout.print("Zinfer quantize\n", .{});
    try stdout.print("model_dir: {s}\n", .{model_dir});
    try stdout.print("scheme: {s}\n", .{scheme.name()});
    try stdout.print("output: {s}\n", .{output_path});

    var timer = try std.time.Timer.start();
    try quantized.quantizeModel(allocator, input_path, output_path, scheme);
    const elapsed_ns = timer.read();
    const output_size = try fileSizeAtPath(output_path);

    try stdout.print("elapsed_ms: {d:.3}\n", .{nsToMs(elapsed_ns)});
    try stdout.print("output_bytes: {d}\n", .{output_size});
    try stdout.print("output_mib: {d:.3}\n", .{bytesToMiB(output_size)});
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

fn fileSizeAtPath(path: []const u8) !u64 {
    const file = if (std.fs.path.isAbsolute(path))
        try std.fs.openFileAbsolute(path, .{})
    else
        try std.fs.cwd().openFile(path, .{});
    defer file.close();
    const stat = try file.stat();
    return stat.size;
}

fn weightArtifactSize(allocator: std.mem.Allocator, model_dir: []const u8, backend_name: []const u8) !u64 {
    const file_name = if (std.mem.eql(u8, backend_name, "q4"))
        "model.q4.zinfer"
    else if (std.mem.eql(u8, backend_name, "q8"))
        "model.q8.zinfer"
    else
        "model.safetensors";
    const path = try std.fs.path.join(allocator, &.{ model_dir, file_name });
    defer allocator.free(path);
    return fileSizeAtPath(path);
}

fn estimateKvCacheBytes(cfg: decoder_family.DecoderConfig, max_seq_len: usize, use_optimized_bf16_cache: bool) u64 {
    if (use_optimized_bf16_cache) {
        return optimized_kv_cache.estimateBytes(
            cfg.num_hidden_layers,
            max_seq_len,
            cfg.num_key_value_heads,
            cfg.head_dim,
        );
    }

    const total = @as(u128, cfg.num_hidden_layers) *
        @as(u128, max_seq_len) *
        @as(u128, cfg.num_key_value_heads) *
        @as(u128, cfg.head_dim) *
        2 *
        @sizeOf(f32);
    return @intCast(total);
}

fn nsToMs(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1_000_000.0;
}

fn bytesToMiB(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / (1024.0 * 1024.0);
}

fn tokensPerSecond(token_count: usize, elapsed_ns: u64) f64 {
    if (token_count == 0 or elapsed_ns == 0) return 0.0;
    return @as(f64, @floatFromInt(token_count)) * 1_000_000_000.0 / @as(f64, @floatFromInt(elapsed_ns));
}

fn tokenizeText(
    allocator: std.mem.Allocator,
    model_dir: []const u8,
    text: []const u8,
) !void {
    const config_path = try std.fs.path.join(allocator, &.{ model_dir, "config.json" });
    defer allocator.free(config_path);

    var parsed_config = try decoder_family.loadConfigFromFile(allocator, config_path);
    defer parsed_config.deinit();

    var tokenizer = try decoder_family.loadTokenizerFromModelDir(
        allocator,
        parsed_config.value.architecture,
        model_dir,
    );
    defer tokenizer.deinit();

    const ids = try tokenizer.encodeAlloc(allocator, text);
    defer allocator.free(ids);

    const stdout = std.fs.File.stdout().deprecatedWriter();
    try stdout.print("Zinfer tokenize\n", .{});
    try stdout.print("text: {s}\n", .{text});
    try stdout.print("ids: [", .{});
    for (ids, 0..) |id, idx| {
        if (idx != 0) try stdout.print(", ", .{});
        try stdout.print("{d}", .{id});
    }
    try stdout.print("]\n", .{});
}

fn decodeIds(
    allocator: std.mem.Allocator,
    model_dir: []const u8,
    ids_csv: []const u8,
) !void {
    const config_path = try std.fs.path.join(allocator, &.{ model_dir, "config.json" });
    defer allocator.free(config_path);

    var parsed_config = try decoder_family.loadConfigFromFile(allocator, config_path);
    defer parsed_config.deinit();

    var tokenizer = try decoder_family.loadTokenizerFromModelDir(
        allocator,
        parsed_config.value.architecture,
        model_dir,
    );
    defer tokenizer.deinit();

    const ids_usize = try parseTokenIdsAlloc(allocator, ids_csv);
    defer allocator.free(ids_usize);
    const ids = try allocator.alloc(u32, ids_usize.len);
    defer allocator.free(ids);
    for (ids_usize, 0..) |value, idx| {
        ids[idx] = std.math.cast(u32, value) orelse return error.TokenIdOutOfRange;
    }

    const text = try tokenizer.decodeAlloc(allocator, ids);
    defer allocator.free(text);

    const stdout = std.fs.File.stdout().deprecatedWriter();
    try stdout.print("Zinfer decode\n", .{});
    try stdout.print("ids: {s}\n", .{ids_csv});
    try stdout.print("text: {s}\n", .{text});
}

fn buildSingleUserPromptAlloc(
    allocator: std.mem.Allocator,
    architecture: decoder_family.Architecture,
    user_text: []const u8,
    system_prompt: ?[]const u8,
    thinking_mode: decoder_family.ThinkingMode,
) ![]u8 {
    if (system_prompt) |system| {
        const messages = [_]decoder_family.Message{
            .{ .role = .system, .content = system },
            .{ .role = .user, .content = user_text },
        };
        return decoder_family.renderMessagesPromptAlloc(allocator, architecture, &messages, thinking_mode);
    }
    return decoder_family.renderSingleUserPromptAlloc(allocator, architecture, user_text, thinking_mode);
}

fn buildMessagesPromptAlloc(
    allocator: std.mem.Allocator,
    architecture: decoder_family.Architecture,
    messages: []const decoder_family.Message,
    system_prompt: ?[]const u8,
    thinking_mode: decoder_family.ThinkingMode,
) ![]u8 {
    if (system_prompt == null) {
        return decoder_family.renderMessagesPromptAlloc(allocator, architecture, messages, thinking_mode);
    }

    const system = system_prompt.?;
    const needs_prepend = messages.len == 0 or messages[0].role != .system;
    if (!needs_prepend) {
        return decoder_family.renderMessagesPromptAlloc(allocator, architecture, messages, thinking_mode);
    }

    const expanded = try allocator.alloc(decoder_family.Message, messages.len + 1);
    defer allocator.free(expanded);
    expanded[0] = .{ .role = .system, .content = system };
    @memcpy(expanded[1..], messages);
    return decoder_family.renderMessagesPromptAlloc(allocator, architecture, expanded, thinking_mode);
}

const StopAnalysis = struct {
    printable_len: usize,
    stop_hit: bool,
    response_len: usize,
};

fn analyzeGeneratedText(text: []const u8, stop_sequences: [][]const u8) StopAnalysis {
    var max_overlap: usize = 0;

    for (stop_sequences) |stop_sequence| {
        if (stop_sequence.len == 0) continue;
        if (std.mem.endsWith(u8, text, stop_sequence)) {
            return .{
                .printable_len = text.len - stop_sequence.len,
                .stop_hit = true,
                .response_len = text.len - stop_sequence.len,
            };
        }

        const max_candidate = @min(text.len, stop_sequence.len - 1);
        var overlap = max_candidate;
        while (overlap > 0) : (overlap -= 1) {
            if (std.mem.eql(u8, text[text.len - overlap ..], stop_sequence[0..overlap])) {
                max_overlap = @max(max_overlap, overlap);
                break;
            }
        }
    }

    return .{
        .printable_len = text.len - max_overlap,
        .stop_hit = false,
        .response_len = text.len,
    };
}

fn analyzeAndMaybeStream(
    allocator: std.mem.Allocator,
    tokenizer: *decoder_family.Tokenizer,
    generated_ids: []const u32,
    options: GenerateOptions,
    stdout: anytype,
    streamed_len: *usize,
) !?[]u8 {
    if (!options.stream_output and options.stop_sequences.len == 0) return null;

    const decoded = tokenizer.decodeAlloc(allocator, generated_ids) catch |err| switch (err) {
        error.InvalidWtf8 => return null,
        else => return err,
    };
    defer allocator.free(decoded);

    const analysis = analyzeGeneratedText(decoded, options.stop_sequences);
    if (options.stream_output and analysis.printable_len > streamed_len.*) {
        try stdout.writeAll(decoded[streamed_len.*..analysis.printable_len]);
        streamed_len.* = analysis.printable_len;
    }
    if (!analysis.stop_hit) return null;

    return try allocator.dupe(u8, decoded[0..analysis.response_len]);
}

fn generateText(
    allocator: std.mem.Allocator,
    model_dir: []const u8,
    user_text: []const u8,
    options: GenerateOptions,
) !void {
    var runtime = try GeneratorRuntime.init(allocator, model_dir, options.backend_scheme, options.thread_count);
    defer runtime.deinit();
    const stdout = std.fs.File.stdout().deprecatedWriter();

    try stdout.print("Zinfer generate\n", .{});
    try stdout.print("mode: {s}\n", .{thinkingModeName(options.thinking_mode)});
    try stdout.print("prompt: {s}\n", .{user_text});
    try stdout.writeAll("response: ");

    const prompt = try buildSingleUserPromptAlloc(
        allocator,
        runtime.model.cfg.architecture,
        user_text,
        options.system_prompt,
        options.thinking_mode,
    );
    defer allocator.free(prompt);

    const response = try runtime.generateFromPrompt(prompt, options);
    defer allocator.free(response);
    if (!options.stream_output) {
        try stdout.print("{s}", .{response});
    }
    try stdout.writeAll("\n");
}

fn generateChatFromFile(
    allocator: std.mem.Allocator,
    model_dir: []const u8,
    messages_json_path: []const u8,
    options: GenerateOptions,
) !void {
    var runtime = try GeneratorRuntime.init(allocator, model_dir, options.backend_scheme, options.thread_count);
    defer runtime.deinit();
    const stdout = std.fs.File.stdout().deprecatedWriter();

    try stdout.print("Zinfer generate-chat\n", .{});
    try stdout.print("mode: {s}\n", .{thinkingModeName(options.thinking_mode)});
    try stdout.print("messages_path: {s}\n", .{messages_json_path});
    try stdout.writeAll("response: ");

    var messages = try loadChatMessages(allocator, messages_json_path);
    defer messages.deinit();

    const prompt = try buildMessagesPromptAlloc(
        allocator,
        runtime.model.cfg.architecture,
        messages.items,
        options.system_prompt,
        options.thinking_mode,
    );
    defer allocator.free(prompt);

    const response = try runtime.generateFromPrompt(prompt, options);
    defer allocator.free(response);
    if (!options.stream_output) {
        try stdout.print("{s}", .{response});
    }
    try stdout.writeAll("\n");
}

fn parseThinkingMode(text: []const u8) !decoder_family.ThinkingMode {
    if (std.mem.eql(u8, text, "think")) return .enabled;
    if (std.mem.eql(u8, text, "no-think")) return .disabled;
    return error.InvalidThinkingMode;
}

fn thinkingModeName(mode: decoder_family.ThinkingMode) []const u8 {
    return switch (mode) {
        .enabled => "think",
        .disabled => "no-think",
    };
}

const LoadedChatMessages = struct {
    arena: std.heap.ArenaAllocator,
    items: []decoder_family.Message,

    fn deinit(self: *LoadedChatMessages) void {
        self.arena.deinit();
    }
};

const SessionMetadata = struct {
    model_dir: []const u8,
    options: GenerateOptions,
};

fn loadChatMessages(
    backing_allocator: std.mem.Allocator,
    path: []const u8,
) !LoadedChatMessages {
    var arena = std.heap.ArenaAllocator.init(backing_allocator);
    errdefer arena.deinit();
    const allocator = arena.allocator();

    const bytes = try readFileAllocAtPath(allocator, path, 1024 * 1024);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, bytes, .{});
    const messages_value = switch (parsed.value) {
        .array => parsed.value,
        .object => parsed.value.object.get("messages") orelse return error.InvalidMessagesJson,
        else => return error.InvalidMessagesJson,
    };
    if (messages_value != .array) return error.InvalidMessagesJson;

    const messages = try allocator.alloc(decoder_family.Message, messages_value.array.items.len);

    for (messages_value.array.items, 0..) |item, idx| {
        if (item != .object) return error.InvalidMessagesJson;

        const role_value = item.object.get("role") orelse return error.InvalidMessagesJson;
        const content_value = item.object.get("content") orelse return error.InvalidMessagesJson;
        if (role_value != .string or content_value != .string) return error.InvalidMessagesJson;

        var tool_calls: []const decoder_family.ToolCall = &.{};
        if (item.object.get("tool_calls")) |tool_calls_value| {
            if (tool_calls_value != .array) return error.InvalidMessagesJson;
            const parsed_tool_calls = try allocator.alloc(decoder_family.ToolCall, tool_calls_value.array.items.len);
            for (tool_calls_value.array.items, 0..) |tool_call_item, tool_idx| {
                if (tool_call_item != .object) return error.InvalidMessagesJson;
                const name_value = tool_call_item.object.get("name") orelse return error.InvalidMessagesJson;
                const args_value = tool_call_item.object.get("arguments") orelse return error.InvalidMessagesJson;
                if (name_value != .string or args_value != .string) return error.InvalidMessagesJson;
                parsed_tool_calls[tool_idx] = .{
                    .name = try allocator.dupe(u8, name_value.string),
                    .arguments_json = try allocator.dupe(u8, args_value.string),
                };
            }
            tool_calls = parsed_tool_calls;
        }

        messages[idx] = .{
            .role = try parseChatRole(role_value.string),
            .content = try allocator.dupe(u8, content_value.string),
            .tool_calls = tool_calls,
        };
    }

    return .{
        .arena = arena,
        .items = messages,
    };
}

fn readFileAllocAtPath(
    allocator: std.mem.Allocator,
    path: []const u8,
    max_bytes: usize,
) ![]u8 {
    if (std.fs.path.isAbsolute(path)) {
        const file = try std.fs.openFileAbsolute(path, .{});
        defer file.close();
        return file.readToEndAlloc(allocator, max_bytes);
    }
    return std.fs.cwd().readFileAlloc(allocator, path, max_bytes);
}

fn parseChatRole(text: []const u8) !decoder_family.Role {
    if (std.mem.eql(u8, text, "system")) return .system;
    if (std.mem.eql(u8, text, "user")) return .user;
    if (std.mem.eql(u8, text, "assistant")) return .assistant;
    if (std.mem.eql(u8, text, "tool")) return .tool;
    return error.InvalidChatRole;
}

const GeneratorRuntime = struct {
    allocator: std.mem.Allocator,
    tokenizer: decoder_family.Tokenizer,
    model: optimized_decoder.Runtime,

    fn init(
        allocator: std.mem.Allocator,
        model_dir: []const u8,
        backend_scheme: tensor_backend.Scheme,
        thread_count: usize,
    ) !GeneratorRuntime {
        const config_path = try std.fs.path.join(allocator, &.{ model_dir, "config.json" });
        defer allocator.free(config_path);
        var parsed_config = try decoder_family.loadConfigFromFile(allocator, config_path);
        defer parsed_config.deinit();

        var tokenizer = try decoder_family.loadTokenizerFromModelDir(
            allocator,
            parsed_config.value.architecture,
            model_dir,
        );
        errdefer tokenizer.deinit();

        var model = try optimized_decoder.Runtime.init(
            allocator,
            model_dir,
            backend_scheme,
            if (thread_count == 0) null else thread_count,
        );
        errdefer model.deinit();

        return .{
            .allocator = allocator,
            .tokenizer = tokenizer,
            .model = model,
        };
    }

    fn deinit(self: *GeneratorRuntime) void {
        self.model.deinit();
        self.tokenizer.deinit();
    }

    fn generateFromPrompt(
        self: *GeneratorRuntime,
        prompt: []const u8,
        options: GenerateOptions,
    ) ![]u8 {
        const prompt_ids_u32 = try self.tokenizer.encodeAlloc(self.allocator, prompt);
        defer self.allocator.free(prompt_ids_u32);
        if (prompt_ids_u32.len == 0) return error.EmptyPrompt;

        const prompt_ids = try self.allocator.alloc(usize, prompt_ids_u32.len);
        defer self.allocator.free(prompt_ids);
        for (prompt_ids_u32, 0..) |token_id, idx| {
            prompt_ids[idx] = token_id;
        }

        const cfg = self.model.cfg;
        var cache = try optimized_kv_cache.ModelCache.init(
            self.allocator,
            cfg.num_hidden_layers,
            prompt_ids.len + options.max_new_tokens,
            cfg.num_key_value_heads,
            cfg.head_dim,
        );
        defer cache.deinit();
        var workspace = try self.model.initWorkspace(prompt_ids.len + options.max_new_tokens);
        defer workspace.deinit();

        const effective_stop_sequences = try decoder_family.effectiveStopSequencesAlloc(
            self.allocator,
            cfg.architecture,
            options.stop_sequences,
        );
        defer self.allocator.free(effective_stop_sequences);

        var current_logits = try self.model.prefillTokenIds(&workspace, &cache, prompt_ids);

        var generated = std.ArrayListUnmanaged(u32).empty;
        defer generated.deinit(self.allocator);

        var history_ids = std.ArrayListUnmanaged(usize).empty;
        defer history_ids.deinit(self.allocator);
        try history_ids.appendSlice(self.allocator, prompt_ids);
        const stdout = std.fs.File.stdout().deprecatedWriter();
        var streamed_len: usize = 0;
        var prng = std.Random.DefaultPrng.init(options.seed);
        for (0..options.max_new_tokens) |_| {
            const next_token = try sampler.sampleToken(self.allocator, prng.random(), current_logits, history_ids.items, options.sampling);
            if (decoder_family.isEosToken(cfg.architecture, next_token)) {
                break;
            }

            try generated.append(self.allocator, std.math.cast(u32, next_token) orelse return error.TokenIdOutOfRange);
            try history_ids.append(self.allocator, next_token);
            var effective_options = options;
            effective_options.stop_sequences = effective_stop_sequences;
            if (try analyzeAndMaybeStream(self.allocator, &self.tokenizer, generated.items, effective_options, stdout, &streamed_len)) |trimmed| {
                return trimmed;
            }

            current_logits = try self.model.forwardTokenId(&workspace, &cache, next_token);
        }

        const response = try self.tokenizer.decodeAlloc(self.allocator, generated.items);
        if (options.stream_output and response.len > streamed_len) {
            try stdout.writeAll(response[streamed_len..]);
        }
        return response;
    }
};

fn chatLoop(
    allocator: std.mem.Allocator,
    model_dir: []const u8,
    options: GenerateOptions,
    load_path: ?[]const u8,
    save_path: ?[]const u8,
) !void {
    var runtime = try GeneratorRuntime.init(allocator, model_dir, options.backend_scheme, options.thread_count);
    defer runtime.deinit();

    var history = ChatHistory.init(allocator);
    defer history.deinit();

    const stdout = std.fs.File.stdout().deprecatedWriter();
    const stdin = std.fs.File.stdin();
    const reader = stdin.deprecatedReader();

    try stdout.print("Zinfer chat\n", .{});
    try stdout.print("model_dir: {s}\n", .{model_dir});
    try stdout.print("mode: {s}\n", .{thinkingModeName(options.thinking_mode)});
    try stdout.writeAll("commands: /exit /quit /clear /save <path> /load <path>\n");

    if (load_path) |path| {
        try history.loadFromFile(path);
    }
    if (options.system_prompt) |system_prompt| {
        try history.prependSystemIfMissing(system_prompt);
    }

    while (true) {
        try stdout.writeAll("user> ");
        const line_opt = try reader.readUntilDelimiterOrEofAlloc(allocator, '\n', 64 * 1024);
        if (line_opt == null) break;
        defer allocator.free(line_opt.?);

        const line = std.mem.trimRight(u8, line_opt.?, "\r\n");
        if (line.len == 0) continue;

        if (std.mem.eql(u8, line, "/exit") or std.mem.eql(u8, line, "/quit")) break;
        if (std.mem.eql(u8, line, "/clear")) {
            history.clear();
            if (options.system_prompt) |system_prompt| {
                try history.prependSystemIfMissing(system_prompt);
            }
            try stdout.writeAll("history cleared\n");
            continue;
        }
        if (std.mem.startsWith(u8, line, "/save ")) {
            const path = std.mem.trim(u8, line["/save ".len..], " ");
            if (path.len == 0) return error.MissingFlagValue;
            try history.saveToFile(path, .{ .model_dir = model_dir, .options = options });
            try stdout.print("saved: {s}\n", .{path});
            continue;
        }
        if (std.mem.startsWith(u8, line, "/load ")) {
            const path = std.mem.trim(u8, line["/load ".len..], " ");
            if (path.len == 0) return error.MissingFlagValue;
            try history.loadFromFile(path);
            if (options.system_prompt) |system_prompt| {
                try history.prependSystemIfMissing(system_prompt);
            }
            try stdout.print("loaded: {s}\n", .{path});
            continue;
        }

        try history.append(.user, line);

        const prompt = try decoder_family.renderMessagesPromptAlloc(allocator, runtime.model.cfg.architecture, history.items(), options.thinking_mode);
        defer allocator.free(prompt);

        try stdout.writeAll("assistant> ");
        const response = try runtime.generateFromPrompt(prompt, options);
        defer allocator.free(response);

        if (!options.stream_output) {
            try stdout.print("{s}", .{response});
        }
        try stdout.writeAll("\n");
        try history.append(.assistant, decoder_family.assistantHistoryContent(runtime.model.cfg.architecture, response));
    }

    if (save_path) |path| {
        try history.saveToFile(path, .{ .model_dir = model_dir, .options = options });
    }
}

const ChatHistory = struct {
    allocator: std.mem.Allocator,
    messages: std.ArrayListUnmanaged(decoder_family.Message),

    fn init(allocator: std.mem.Allocator) ChatHistory {
        return .{
            .allocator = allocator,
            .messages = .empty,
        };
    }

    fn deinit(self: *ChatHistory) void {
        self.clear();
        self.messages.deinit(self.allocator);
    }

    fn clear(self: *ChatHistory) void {
        for (self.messages.items) |message| {
            self.allocator.free(message.content);
            if (message.tool_calls.len != 0) {
                for (message.tool_calls) |tool_call| {
                    self.allocator.free(tool_call.name);
                    self.allocator.free(tool_call.arguments_json);
                }
                self.allocator.free(message.tool_calls);
            }
        }
        self.messages.clearRetainingCapacity();
    }

    fn append(self: *ChatHistory, role: decoder_family.Role, content: []const u8) !void {
        try self.appendMessage(.{
            .role = role,
            .content = content,
        });
    }

    fn appendMessage(self: *ChatHistory, message: decoder_family.Message) !void {
        const owned_content = try self.allocator.dupe(u8, message.content);
        errdefer self.allocator.free(owned_content);

        var owned_tool_calls: []const decoder_family.ToolCall = &.{};
        if (message.tool_calls.len != 0) {
            const copied = try self.allocator.alloc(decoder_family.ToolCall, message.tool_calls.len);
            errdefer self.allocator.free(copied);

            var copied_len: usize = 0;
            errdefer {
                for (copied[0..copied_len]) |tool_call| {
                    self.allocator.free(tool_call.name);
                    self.allocator.free(tool_call.arguments_json);
                }
            }

            for (message.tool_calls, 0..) |tool_call, idx| {
                copied[idx] = .{
                    .name = try self.allocator.dupe(u8, tool_call.name),
                    .arguments_json = try self.allocator.dupe(u8, tool_call.arguments_json),
                };
                copied_len += 1;
            }
            owned_tool_calls = copied;
        }
        errdefer if (owned_tool_calls.len != 0) {
            for (owned_tool_calls) |tool_call| {
                self.allocator.free(tool_call.name);
                self.allocator.free(tool_call.arguments_json);
            }
            self.allocator.free(owned_tool_calls);
        };

        try self.messages.append(self.allocator, .{
            .role = message.role,
            .content = owned_content,
            .tool_calls = owned_tool_calls,
        });
    }

    fn prependSystemIfMissing(self: *ChatHistory, content: []const u8) !void {
        if (self.messages.items.len != 0 and self.messages.items[0].role == .system) return;

        const owned = try self.allocator.dupe(u8, content);
        errdefer self.allocator.free(owned);
        try self.messages.insert(self.allocator, 0, .{
            .role = .system,
            .content = owned,
        });
    }

    fn saveToFile(self: *const ChatHistory, path: []const u8, metadata: SessionMetadata) !void {
        const file = if (std.fs.path.isAbsolute(path))
            try std.fs.createFileAbsolute(path, .{ .truncate = true })
        else
            try std.fs.cwd().createFile(path, .{ .truncate = true });
        defer file.close();
        const writer = file.deprecatedWriter();

        try writer.writeAll("{\n");
        try writer.writeAll("  \"version\": 1,\n");
        try writer.writeAll("  \"kind\": \"zinfer_chat_session\",\n");
        try writer.writeAll("  \"saved_unix\": ");
        try writer.print("{d}", .{std.time.timestamp()});
        try writer.writeAll(",\n");
        try writer.writeAll("  \"model_dir\": ");
        try writer.print("{f}", .{std.json.fmt(metadata.model_dir, .{})});
        try writer.writeAll(",\n");
        try writer.writeAll("  \"options\": {\n");
        try writer.writeAll("    \"thinking_mode\": ");
        try writer.print("{f}", .{std.json.fmt(thinkingModeName(metadata.options.thinking_mode), .{})});
        try writer.writeAll(",\n");
        try writer.writeAll("    \"max_new_tokens\": ");
        try writer.print("{d}", .{metadata.options.max_new_tokens});
        try writer.writeAll(",\n");
        try writer.writeAll("    \"seed\": ");
        try writer.print("{d}", .{metadata.options.seed});
        try writer.writeAll(",\n");
        try writer.writeAll("    \"backend\": ");
        try writer.print("{f}", .{std.json.fmt(metadata.options.backend_scheme.name(), .{})});
        try writer.writeAll(",\n");
        try writer.writeAll("    \"threads\": ");
        try writer.print("{d}", .{metadata.options.thread_count});
        try writer.writeAll(",\n");
        try writer.writeAll("    \"stream_output\": ");
        try writer.writeAll(if (metadata.options.stream_output) "true" else "false");
        try writer.writeAll(",\n");
        try writer.writeAll("    \"system_prompt\": ");
        if (metadata.options.system_prompt) |system_prompt| {
            try writer.print("{f}", .{std.json.fmt(system_prompt, .{})});
        } else {
            try writer.writeAll("null");
        }
        try writer.writeAll(",\n");
        try writer.writeAll("    \"sampling\": {\n");
        try writer.writeAll("      \"temperature\": ");
        try writer.print("{d}", .{metadata.options.sampling.temperature});
        try writer.writeAll(",\n");
        try writer.writeAll("      \"top_p\": ");
        try writer.print("{d}", .{metadata.options.sampling.top_p});
        try writer.writeAll(",\n");
        try writer.writeAll("      \"top_k\": ");
        try writer.print("{d}", .{metadata.options.sampling.top_k});
        try writer.writeAll(",\n");
        try writer.writeAll("      \"min_p\": ");
        try writer.print("{d}", .{metadata.options.sampling.min_p});
        try writer.writeAll(",\n");
        try writer.writeAll("      \"presence_penalty\": ");
        try writer.print("{d}", .{metadata.options.sampling.presence_penalty});
        try writer.writeAll(",\n");
        try writer.writeAll("      \"frequency_penalty\": ");
        try writer.print("{d}", .{metadata.options.sampling.frequency_penalty});
        try writer.writeAll(",\n");
        try writer.writeAll("      \"repetition_penalty\": ");
        try writer.print("{d}", .{metadata.options.sampling.repetition_penalty});
        try writer.writeAll("\n");
        try writer.writeAll("    },\n");
        try writer.writeAll("    \"stop_sequences\": [");
        for (metadata.options.stop_sequences, 0..) |stop_sequence, idx| {
            if (idx != 0) try writer.writeAll(", ");
            try writer.print("{f}", .{std.json.fmt(stop_sequence, .{})});
        }
        try writer.writeAll("]\n");
        try writer.writeAll("  },\n");
        try writer.writeAll("  \"messages\": [\n");
        for (self.messages.items, 0..) |message, idx| {
            if (idx != 0) try writer.writeAll(",\n");
            try writer.writeAll("  {\"role\": ");
            try writer.print("{f}", .{std.json.fmt(message.role.name(), .{})});
            try writer.writeAll(", \"content\": ");
            try writer.print("{f}", .{std.json.fmt(message.content, .{})});
            if (message.tool_calls.len != 0) {
                try writer.writeAll(", \"tool_calls\": [");
                for (message.tool_calls, 0..) |tool_call, tool_idx| {
                    if (tool_idx != 0) try writer.writeAll(", ");
                    try writer.writeAll("{\"name\": ");
                    try writer.print("{f}", .{std.json.fmt(tool_call.name, .{})});
                    try writer.writeAll(", \"arguments\": ");
                    try writer.print("{f}", .{std.json.fmt(tool_call.arguments_json, .{})});
                    try writer.writeAll("}");
                }
                try writer.writeAll("]");
            }
            try writer.writeAll("}");
        }
        try writer.writeAll("\n  ]\n}\n");
    }

    fn loadFromFile(self: *ChatHistory, path: []const u8) !void {
        self.clear();

        var loaded = try loadChatMessages(self.allocator, path);
        defer loaded.deinit();

        for (loaded.items) |message| {
            try self.appendMessage(message);
        }
    }

    fn items(self: *const ChatHistory) []const decoder_family.Message {
        return self.messages.items;
    }
};

test "analyzeGeneratedText trims full stop sequence" {
    const testing = std.testing;

    const stops: []const []const u8 = &[_][]const u8{"today?"};
    const analysis = analyzeGeneratedText("Hello today?", @constCast(stops));
    try testing.expect(analysis.stop_hit);
    try testing.expectEqual(@as(usize, 6), analysis.printable_len);
    try testing.expectEqual(@as(usize, 6), analysis.response_len);
}

test "analyzeGeneratedText holds back partial stop prefix" {
    const testing = std.testing;

    const stops: []const []const u8 = &[_][]const u8{"today?"};
    const analysis = analyzeGeneratedText("Hello to", @constCast(stops));
    try testing.expect(!analysis.stop_hit);
    try testing.expectEqual(@as(usize, 6), analysis.printable_len);
    try testing.expectEqual(@as(usize, 8), analysis.response_len);
}

test "chat history session save and load preserves tool calls" {
    const testing = std.testing;

    var temp_dir = testing.tmpDir(.{});
    defer temp_dir.cleanup();

    const temp_root = try temp_dir.dir.realpathAlloc(testing.allocator, ".");
    defer testing.allocator.free(temp_root);
    const session_path = try std.fs.path.join(testing.allocator, &.{ temp_root, "session.json" });
    defer testing.allocator.free(session_path);

    var history = ChatHistory.init(testing.allocator);
    defer history.deinit();

    try history.append(.system, "You are terse.");
    try history.append(.user, "Use the weather tool.");
    try history.appendMessage(.{
        .role = .assistant,
        .content = "",
            .tool_calls = &[_]decoder_family.ToolCall{
                .{ .name = "lookup_weather", .arguments_json = "{\"city\":\"Shanghai\"}" },
            },
        });
    try history.append(.tool, "{\"temp\":22}");

    const stop_sequences = [_][]const u8{"</tool_response>"};
    try history.saveToFile(session_path, .{
        .model_dir = "models/Qwen3-0.6B",
        .options = .{
            .max_new_tokens = 64,
            .thinking_mode = .disabled,
            .system_prompt = "You are terse.",
            .sampling = defaultSamplingConfig(.disabled),
            .seed = 7,
            .stream_output = true,
            .stop_sequences = @constCast(stop_sequences[0..]),
            .backend_scheme = .q4,
            .thread_count = 8,
        },
    });

    const saved = try readFileAllocAtPath(testing.allocator, session_path, 64 * 1024);
    defer testing.allocator.free(saved);
    try testing.expect(std.mem.indexOf(u8, saved, "\"kind\": \"zinfer_chat_session\"") != null);
    try testing.expect(std.mem.indexOf(u8, saved, "\"tool_calls\"") != null);
    try testing.expect(std.mem.indexOf(u8, saved, "\"model_dir\": \"models/Qwen3-0.6B\"") != null);

    var loaded = ChatHistory.init(testing.allocator);
    defer loaded.deinit();
    try loaded.loadFromFile(session_path);

    try testing.expectEqual(@as(usize, 4), loaded.items().len);
    try testing.expectEqual(decoder_family.Role.system, loaded.items()[0].role);
    try testing.expectEqual(decoder_family.Role.assistant, loaded.items()[2].role);
    try testing.expectEqual(decoder_family.Role.tool, loaded.items()[3].role);
    try testing.expectEqual(@as(usize, 1), loaded.items()[2].tool_calls.len);
    try testing.expectEqualStrings("lookup_weather", loaded.items()[2].tool_calls[0].name);
    try testing.expectEqualStrings("{\"city\":\"Shanghai\"}", loaded.items()[2].tool_calls[0].arguments_json);
}
