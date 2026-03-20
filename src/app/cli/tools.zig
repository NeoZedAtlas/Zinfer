const std = @import("std");
const cli_prompts = @import("prompts.zig");
const cli_token_ids = @import("token_ids.zig");
const optimized_kv_cache = @import("../../model/optimized_kv_cache.zig");
const decoder_family = @import("../../model/decoder_family.zig");
const optimized_decoder = @import("../../model/optimized_decoder.zig");
const quantized = @import("../../tensor/quantized.zig");
const tensor_backend = @import("../../tensor/backend.zig");
const GenerateOptions = @import("args.zig").GenerateOptions;

pub fn benchPrompt(
    allocator: std.mem.Allocator,
    model_dir: []const u8,
    user_text: []const u8,
    options: GenerateOptions,
) !void {
    var runtime = try GeneratorRuntime.init(allocator, model_dir, options.backend_scheme, options.thread_count);
    defer runtime.deinit();
    const cfg = runtime.model.cfg;
    const resolved_kv_cache_scheme = optimized_kv_cache.resolveScheme(options.kv_cache_scheme, runtime.model.backendName());

    const prompt = try cli_prompts.buildSingleUserPromptAlloc(
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
        resolved_kv_cache_scheme,
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

    const weights_size = runtime.model.artifactBytes();
    const kv_cache_bytes = estimateKvCacheBytes(cfg, prompt_ids.len + options.max_new_tokens, resolved_kv_cache_scheme);
    const stdout = std.fs.File.stdout().deprecatedWriter();
    try stdout.print("Zinfer benchmark\n", .{});
    try stdout.print("model_dir: {s}\n", .{model_dir});
    try stdout.print("backend: {s}\n", .{runtime.model.backendName()});
    try stdout.print("kv_cache: {s}\n", .{resolved_kv_cache_scheme.name()});
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

pub fn quantizeModelDir(
    allocator: std.mem.Allocator,
    model_dir: []const u8,
    scheme_text: []const u8,
) !void {
    const scheme: quantized.Scheme = if (std.mem.eql(u8, scheme_text, "q8"))
        .q8
    else if (std.mem.eql(u8, scheme_text, "q6"))
        .q6
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

pub fn tokenizeText(
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

pub fn decodeIds(
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

    const ids_usize = try cli_token_ids.parseTokenIdsAlloc(allocator, ids_csv);
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

fn fileSizeAtPath(path: []const u8) !u64 {
    const file = if (std.fs.path.isAbsolute(path))
        try std.fs.openFileAbsolute(path, .{})
    else
        try std.fs.cwd().openFile(path, .{});
    defer file.close();
    const stat = try file.stat();
    return stat.size;
}

fn estimateKvCacheBytes(
    cfg: decoder_family.DecoderConfig,
    max_seq_len: usize,
    kv_cache_scheme: optimized_kv_cache.Scheme,
) u64 {
    return optimized_kv_cache.estimateBytes(
        cfg.num_hidden_layers,
        max_seq_len,
        cfg.num_key_value_heads,
        cfg.head_dim,
        kv_cache_scheme,
    );
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
};
