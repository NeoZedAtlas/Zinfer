const std = @import("std");
const GenerateOptions = @import("../../args.zig").GenerateOptions;
const cli_prompts = @import("../../prompts.zig");
const cli_runtime = @import("../../runtime.zig");
const decoder_family = @import("../../../../model/runtime/decoder_family.zig");
const kernel_registry = @import("../../../../kernel/registry.zig");
const optimized_decoder = @import("../../../../model/runtime/optimized_decoder.zig");
const optimized_kv_cache = @import("../../../../model/runtime/optimized_kv_cache.zig");
const tensor_backend = @import("../../../../tensor/backends/backend.zig");
const support = @import("support.zig");

pub fn benchPrompt(
    allocator: std.mem.Allocator,
    model_dir: []const u8,
    user_text: []const u8,
    options: GenerateOptions,
) !void {
    var runtime = try cli_runtime.GeneratorRuntime.init(allocator, model_dir, options.backend_scheme, options.thread_count);
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

    var cache = try optimized_kv_cache.ModelCache.initWithLayout(
        allocator,
        cfg.num_hidden_layers,
        prompt_ids.len + options.max_new_tokens,
        cfg.num_key_value_heads,
        cfg.head_dim,
        resolved_kv_cache_scheme,
        options.q8_layout,
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
        if (decoder_family.isEosToken(cfg.architecture, next_token)) break;

        current_logits = try runtime.model.forwardTokenId(&workspace, &cache, next_token);
        decoded_tokens += 1;
    }
    const decode_ns = decode_timer.read();

    const weights_size = runtime.model.artifactBytes();
    const kv_cache_bytes = support.estimateKvCacheBytes(
        cfg,
        prompt_ids.len + options.max_new_tokens,
        resolved_kv_cache_scheme,
        options.q8_layout,
    );
    const stdout = std.fs.File.stdout().deprecatedWriter();
    try stdout.print("Zinfer benchmark\n", .{});
    try stdout.print("model_dir: {s}\n", .{model_dir});
    try stdout.print("backend: {s}\n", .{runtime.model.backendName()});
    try stdout.print("kv_cache: {s}\n", .{resolved_kv_cache_scheme.name()});
    try stdout.print("q8_layout: {s}\n", .{options.q8_layout.name()});
    try stdout.print("kernel_isa: {s}\n", .{kernel_registry.activeIsa().name()});
    try stdout.print("threads: {d}\n", .{runtime.model.thread_count});
    try stdout.print("prompt_tokens: {d}\n", .{prompt_ids.len});
    try stdout.print("decode_tokens: {d}\n", .{decoded_tokens});
    try stdout.print("tokenize_ms: {d:.3}\n", .{support.nsToMs(tokenize_ns)});
    try stdout.print("prefill_ms: {d:.3}\n", .{support.nsToMs(prefill_ns)});
    try stdout.print("decode_ms: {d:.3}\n", .{support.nsToMs(decode_ns)});
    try stdout.print("prefill_tok_s: {d:.3}\n", .{support.tokensPerSecond(prompt_ids.len, prefill_ns)});
    try stdout.print("decode_tok_s: {d:.3}\n", .{support.tokensPerSecond(decoded_tokens, decode_ns)});
    try stdout.print("weights_bytes: {d}\n", .{weights_size});
    try stdout.print("weights_mib: {d:.3}\n", .{support.bytesToMiB(weights_size)});
    try stdout.print("kv_cache_bytes: {d}\n", .{kv_cache_bytes});
    try stdout.print("kv_cache_mib: {d:.3}\n", .{support.bytesToMiB(kv_cache_bytes)});
}

pub fn benchSuite(allocator: std.mem.Allocator, model_dir: []const u8) !void {
    const stdout = std.fs.File.stdout().deprecatedWriter();
    const profiles = [_]struct {
        name: []const u8,
        prompt: []const u8,
        decode_tokens: usize,
    }{
        .{ .name = "short", .prompt = "hello", .decode_tokens = 2 },
        .{ .name = "medium", .prompt = "Explain how KV cache layout affects decoder attention throughput in one paragraph.", .decode_tokens = 2 },
        .{ .name = "long", .prompt = "Summarize the main decode hotspots in a small transformer runtime. Focus on attention score accumulation, value accumulation, quantized GEMV, and temporary buffer traffic. Keep the answer technical and compact.", .decode_tokens = 2 },
    };
    const backends = [_]tensor_backend.Scheme{ .q6, .q8 };

    try stdout.print("Zinfer perf-gate suite\n", .{});
    try stdout.print("model_dir: {s}\n", .{model_dir});
    try stdout.print("threads: 1\n", .{});
    try stdout.print("kv_cache: auto\n", .{});
    try stdout.print("q8_layout: {s}\n", .{optimized_kv_cache.default_q8_layout.name()});
    try stdout.print("kernel_isa: {s}\n", .{kernel_registry.activeIsa().name()});

    for (profiles) |profile| {
        for (backends) |backend| {
            var options = try initBenchSuiteOptions(allocator, profile.decode_tokens, backend);
            defer options.deinit(allocator);

            try stdout.print(
                "\n[suite] profile={s} backend={s} decode_tokens={d}\n",
                .{ profile.name, backend.name(), profile.decode_tokens },
            );
            try benchPrompt(allocator, model_dir, profile.prompt, options);
        }
    }
}

pub fn benchBatchPrompt(
    allocator: std.mem.Allocator,
    model_dir: []const u8,
    user_text: []const u8,
    batch_size: usize,
    options: GenerateOptions,
) !void {
    var runtime = try cli_runtime.GeneratorRuntime.init(allocator, model_dir, options.backend_scheme, options.thread_count);
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

    var batch = try optimized_decoder.BatchRuntime.init(
        allocator,
        &runtime.model,
        batch_size,
        prompt_ids.len + options.max_new_tokens,
        resolved_kv_cache_scheme,
        options.q8_layout,
    );
    defer batch.deinit();

    var prefill_timer = try std.time.Timer.start();
    for (0..batch_size) |request_index| {
        try batch.prefillPromptIds(request_index, prompt_ids);
    }
    const prefill_ns = prefill_timer.read();

    var decode_timer = try std.time.Timer.start();
    const decode_stats = try batch.decodeRoundRobinArgMax(options.max_new_tokens);
    const decode_ns = decode_timer.read();

    const weights_size = runtime.model.artifactBytes();
    const kv_cache_bytes_per_request = support.estimateKvCacheBytes(
        cfg,
        prompt_ids.len + options.max_new_tokens,
        resolved_kv_cache_scheme,
        options.q8_layout,
    );
    const total_prompt_tokens = prompt_ids.len * batch_size;
    const total_kv_cache_bytes = kv_cache_bytes_per_request * batch_size;

    const stdout = std.fs.File.stdout().deprecatedWriter();
    try stdout.print("Zinfer batch benchmark\n", .{});
    try stdout.print("model_dir: {s}\n", .{model_dir});
    try stdout.print("backend: {s}\n", .{runtime.model.backendName()});
    try stdout.print("kv_cache: {s}\n", .{resolved_kv_cache_scheme.name()});
    try stdout.print("q8_layout: {s}\n", .{options.q8_layout.name()});
    try stdout.print("kernel_isa: {s}\n", .{kernel_registry.activeIsa().name()});
    try stdout.print("threads: {d}\n", .{runtime.model.thread_count});
    try stdout.print("scheduler: {s}\n", .{if (decode_stats.scheduler_workers > 1) "synchronous_parallel" else "round_robin"});
    try stdout.print("scheduler_workers: {d}\n", .{decode_stats.scheduler_workers});
    try stdout.print("batch_size: {d}\n", .{batch_size});
    try stdout.print("prompt_tokens_per_request: {d}\n", .{prompt_ids.len});
    try stdout.print("total_prompt_tokens: {d}\n", .{total_prompt_tokens});
    try stdout.print("target_decode_tokens_per_request: {d}\n", .{options.max_new_tokens});
    try stdout.print("decoded_tokens_total: {d}\n", .{decode_stats.total_decoded_tokens});
    try stdout.print("finished_requests: {d}\n", .{decode_stats.finished_requests});
    try stdout.print("tokenize_ms: {d:.3}\n", .{support.nsToMs(tokenize_ns)});
    try stdout.print("prefill_ms: {d:.3}\n", .{support.nsToMs(prefill_ns)});
    try stdout.print("decode_ms: {d:.3}\n", .{support.nsToMs(decode_ns)});
    try stdout.print("prefill_tok_s: {d:.3}\n", .{support.tokensPerSecond(total_prompt_tokens, prefill_ns)});
    try stdout.print("decode_tok_s: {d:.3}\n", .{support.tokensPerSecond(decode_stats.total_decoded_tokens, decode_ns)});
    try stdout.print("weights_bytes: {d}\n", .{weights_size});
    try stdout.print("weights_mib: {d:.3}\n", .{support.bytesToMiB(weights_size)});
    try stdout.print("kv_cache_bytes_per_request: {d}\n", .{kv_cache_bytes_per_request});
    try stdout.print("kv_cache_bytes_total: {d}\n", .{total_kv_cache_bytes});
    try stdout.print("kv_cache_mib_total: {d:.3}\n", .{support.bytesToMiB(total_kv_cache_bytes)});
}

fn initBenchSuiteOptions(
    allocator: std.mem.Allocator,
    max_new_tokens: usize,
    backend_scheme: tensor_backend.Scheme,
) !GenerateOptions {
    return .{
        .max_new_tokens = max_new_tokens,
        .thinking_mode = .disabled,
        .system_prompt = null,
        .sampling = .{
            .temperature = 0.0,
            .top_k = 1,
            .top_p = 1.0,
            .min_p = 0.0,
            .presence_penalty = 0.0,
            .frequency_penalty = 0.0,
            .repetition_penalty = 1.0,
        },
        .seed = 0,
        .stream_output = false,
        .stop_sequences = try allocator.alloc([]const u8, 0),
        .backend_scheme = backend_scheme,
        .kv_cache_scheme = .auto,
        .q8_layout = optimized_kv_cache.default_q8_layout,
        .thread_count = 1,
    };
}
