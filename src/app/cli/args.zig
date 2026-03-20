const std = @import("std");
const optimized_kv_cache = @import("../../model/runtime/optimized_kv_cache.zig");
const decoder_family = @import("../../model/runtime/decoder_family.zig");
const tensor_backend = @import("../../tensor/backend.zig");
const sampler = @import("../../sampling/sampler.zig");

pub const default_model_dir = "models/Qwen3-0.6B";

pub const GenerateOptions = struct {
    max_new_tokens: usize,
    thinking_mode: decoder_family.ThinkingMode,
    system_prompt: ?[]const u8,
    sampling: sampler.SamplingConfig,
    seed: u64,
    stream_output: bool,
    stop_sequences: [][]const u8,
    backend_scheme: tensor_backend.Scheme,
    kv_cache_scheme: optimized_kv_cache.Scheme,
    thread_count: usize,

    pub fn deinit(self: *GenerateOptions, allocator: std.mem.Allocator) void {
        allocator.free(self.stop_sequences);
    }
};

pub const ParsedGenerateInvocation = struct {
    model_dir: []const u8,
    user_text: []const u8,
    options: GenerateOptions,

    pub fn deinit(self: *ParsedGenerateInvocation, allocator: std.mem.Allocator) void {
        self.options.deinit(allocator);
    }
};

pub const ParsedGenerateChatInvocation = struct {
    model_dir: []const u8,
    messages_json_path: []const u8,
    options: GenerateOptions,

    pub fn deinit(self: *ParsedGenerateChatInvocation, allocator: std.mem.Allocator) void {
        self.options.deinit(allocator);
    }
};

pub const ParsedChatInvocation = struct {
    model_dir: []const u8,
    options: GenerateOptions,
    load_path: ?[]const u8,
    save_path: ?[]const u8,

    pub fn deinit(self: *ParsedChatInvocation, allocator: std.mem.Allocator) void {
        self.options.deinit(allocator);
    }
};

pub const ParsedBenchInvocation = struct {
    model_dir: []const u8,
    user_text: []const u8,
    options: GenerateOptions,

    pub fn deinit(self: *ParsedBenchInvocation, allocator: std.mem.Allocator) void {
        self.options.deinit(allocator);
    }
};

pub fn parseGenerateInvocation(
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

pub fn parseGenerateChatInvocation(
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

pub fn parseChatInvocation(
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

pub fn parseBenchInvocation(
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

pub fn defaultSamplingConfig(mode: decoder_family.ThinkingMode) sampler.SamplingConfig {
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
        .kv_cache_scheme = .auto,
        .thread_count = 0,
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
        if (std.mem.eql(u8, arg, "--kv-cache")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            options.kv_cache_scheme = try parseKvCacheScheme(args[i]);
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
    if (std.mem.eql(u8, text, "q6")) return .q6;
    if (std.mem.eql(u8, text, "q8")) return .q8;
    if (std.mem.eql(u8, text, "q4")) return .q4;
    return error.InvalidBackendScheme;
}

fn parseKvCacheScheme(text: []const u8) !optimized_kv_cache.Scheme {
    if (std.mem.eql(u8, text, "auto")) return .auto;
    if (std.mem.eql(u8, text, "bf16")) return .bf16;
    if (std.mem.eql(u8, text, "q8")) return .q8;
    return error.InvalidKvCacheScheme;
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

fn parseThinkingMode(text: []const u8) !decoder_family.ThinkingMode {
    if (std.mem.eql(u8, text, "think")) return .enabled;
    if (std.mem.eql(u8, text, "no-think")) return .disabled;
    return error.InvalidThinkingMode;
}
