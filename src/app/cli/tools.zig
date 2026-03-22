const std = @import("std");
const attention = @import("../../kernel/attention/attention.zig");
const cpu = @import("../../kernel/core/cpu.zig");
const GenerateOptions = @import("args.zig").GenerateOptions;
const cli_prompts = @import("prompts.zig");
const cli_runtime = @import("runtime.zig");
const cli_token_ids = @import("token_ids.zig");
const bfloat16 = @import("../../tensor/formats/bfloat16.zig");
const optimized_kv_cache = @import("../../model/runtime/optimized_kv_cache.zig");
const decoder_family = @import("../../model/runtime/decoder_family.zig");
const tensor_backend = @import("../../tensor/backends/backend.zig");
const quantized = @import("../../tensor/formats/quantized.zig");
const tensor_store = @import("../../tensor/storage/store.zig");

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

pub fn benchSuite(allocator: std.mem.Allocator, model_dir: []const u8) !void {
    const stdout = std.fs.File.stdout().deprecatedWriter();
    const profiles = [_]struct {
        name: []const u8,
        prompt: []const u8,
        decode_tokens: usize,
    }{
        .{
            .name = "short",
            .prompt = "hello",
            .decode_tokens = 2,
        },
        .{
            .name = "medium",
            .prompt = "Explain how KV cache layout affects decoder attention throughput in one paragraph.",
            .decode_tokens = 2,
        },
        .{
            .name = "long",
            .prompt = "Summarize the main decode hotspots in a small transformer runtime. Focus on attention score accumulation, value accumulation, quantized GEMV, and temporary buffer traffic. Keep the answer technical and compact.",
            .decode_tokens = 2,
        },
    };
    const backends = [_]tensor_backend.Scheme{ .q6, .q8 };

    try stdout.print("Zinfer perf-gate suite\n", .{});
    try stdout.print("model_dir: {s}\n", .{model_dir});
    try stdout.print("threads: 1\n", .{});
    try stdout.print("kv_cache: auto\n", .{});

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

pub fn benchHandwrittenOps(
    allocator: std.mem.Allocator,
    model_dir: []const u8,
    requested_iterations: usize,
) !void {
    const config_path = try std.fs.path.join(allocator, &.{ model_dir, "config.json" });
    defer allocator.free(config_path);

    var parsed_config = try decoder_family.loadConfigFromFile(allocator, config_path);
    defer parsed_config.deinit();
    const cfg = parsed_config.value;

    const stdout = std.fs.File.stdout().deprecatedWriter();
    try stdout.print("Zinfer handwritten-op benchmark\n", .{});
    try stdout.print("model_dir: {s}\n", .{model_dir});
    try stdout.print("hidden_size: {d}\n", .{cfg.hidden_size});
    try stdout.print("intermediate_size: {d}\n", .{cfg.intermediate_size});
    try stdout.print("head_dim: {d}\n", .{cfg.head_dim});
    try stdout.print("attention_heads: {d}\n", .{cfg.num_attention_heads});
    try stdout.print("kv_heads: {d}\n", .{cfg.num_key_value_heads});
    try stdout.print("iterations: {s}\n", .{if (requested_iterations == 0) "auto" else "manual"});
    try stdout.print("\n[gemv-row]\n", .{});
    try benchGemvProfile(allocator, stdout, "hidden", cfg.hidden_size, requested_iterations);
    try benchGemvProfile(allocator, stdout, "intermediate", cfg.intermediate_size, requested_iterations);
    try stdout.print("\n[rmsnorm]\n", .{});
    try benchRmsNormProfile(allocator, stdout, "hidden", cfg.hidden_size, requested_iterations);
    try benchRmsNormProfile(allocator, stdout, "head", cfg.head_dim, requested_iterations);
    try stdout.print("\n[swiglu]\n", .{});
    try benchSwiGluProfile(allocator, stdout, cfg.intermediate_size, requested_iterations);
    try stdout.print("\n[attention-q8-cache]\n", .{});
    try benchAttentionProfile(allocator, stdout, cfg, requested_iterations);
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

fn benchGemvProfile(
    allocator: std.mem.Allocator,
    writer: anytype,
    label: []const u8,
    cols: usize,
    requested_iterations: usize,
) !void {
    const iterations = resolveBenchIterations(requested_iterations, cols, 32_000_000);
    const q6_bytes = 4 + (try std.math.divCeil(usize, cols * 6, 8));
    const q4_bytes = 4 + (try std.math.divCeil(usize, cols, 2));

    const input = try allocator.alloc(f32, cols);
    defer allocator.free(input);
    const row_values = try allocator.alloc(f32, cols);
    defer allocator.free(row_values);
    const row_f32 = try allocator.alloc(u8, cols * 4);
    defer allocator.free(row_f32);
    const row_bf16 = try allocator.alloc(u8, cols * 2);
    defer allocator.free(row_bf16);
    const row_q8 = try allocator.alloc(u8, 4 + cols);
    defer allocator.free(row_q8);
    const row_q6 = try allocator.alloc(u8, q6_bytes);
    defer allocator.free(row_q6);
    const row_q4 = try allocator.alloc(u8, q4_bytes);
    defer allocator.free(row_q4);

    fillSyntheticF32(input, 13);
    fillSyntheticF32(row_values, 29);
    encodeF32Row(row_f32, row_values);
    encodeBf16Row(row_bf16, row_values);
    quantized.encodeQ8Row(row_q8, row_values);
    quantized.encodeQ6Row(row_q6, row_values);
    quantized.encodeQ4Row(row_q4, row_values);

    try writer.print("profile: {s} cols={d} iterations={d}\n", .{ label, cols, iterations });
    try benchScalarKernel(writer, "f32", cols, iterations, struct {
        fn run(row: []const u8, vector: []const f32) f32 {
            return tensor_store.dotF32Row(row, vector);
        }
    }.run, row_f32, input);
    try benchScalarKernel(writer, "bf16", cols, iterations, struct {
        fn run(row: []const u8, vector: []const f32) f32 {
            return tensor_store.dotBf16Row(row, vector);
        }
    }.run, row_bf16, input);
    try benchScalarKernel(writer, "q8", cols, iterations, struct {
        fn run(row: []const u8, vector: []const f32) f32 {
            return quantized.dotQ8Row(row, 0, vector);
        }
    }.run, row_q8, input);
    try benchScalarKernel(writer, "q6", cols, iterations, struct {
        fn run(row: []const u8, vector: []const f32) f32 {
            return quantized.dotQ6Row(row, 0, vector);
        }
    }.run, row_q6, input);
    try benchScalarKernel(writer, "q4", cols, iterations, struct {
        fn run(row: []const u8, vector: []const f32) f32 {
            return quantized.dotQ4Row(row, 0, vector);
        }
    }.run, row_q4, input);
}

fn benchAttentionProfile(
    allocator: std.mem.Allocator,
    writer: anytype,
    cfg: decoder_family.DecoderConfig,
    requested_iterations: usize,
) !void {
    const seq_len = @min(cfg.max_position_embeddings, @as(usize, 2048));
    const scale_groups_per_head = try std.math.divCeil(usize, cfg.head_dim, attention.q8_cache_group_size);
    const dot_iterations = resolveBenchIterations(requested_iterations, seq_len * cfg.head_dim, 16_000_000);
    const full_iterations = resolveBenchIterations(
        requested_iterations,
        seq_len * cfg.head_dim * cfg.num_attention_heads,
        8_000_000,
    );

    const query = try allocator.alloc(f32, cfg.head_dim);
    defer allocator.free(query);
    const key_cache = try allocator.alloc(i8, seq_len * cfg.head_dim);
    defer allocator.free(key_cache);
    const value_cache = try allocator.alloc(i8, seq_len * cfg.head_dim);
    defer allocator.free(value_cache);
    const key_scales = try allocator.alloc(u16, seq_len * scale_groups_per_head);
    defer allocator.free(key_scales);
    const value_scales = try allocator.alloc(u16, seq_len * scale_groups_per_head);
    defer allocator.free(value_scales);

    fillSyntheticF32(query, 7);
    fillSyntheticQ8Cache(key_cache, key_scales, cfg.head_dim, 17);
    fillSyntheticQ8Cache(value_cache, value_scales, cfg.head_dim, 31);

    try writer.print("profile: single-kv-head seq_len={d} head_dim={d} iterations={d}\n", .{
        seq_len,
        cfg.head_dim,
        dot_iterations,
    });

    var dot_guard: f32 = 0.0;
    const dot_warmup = @min(dot_iterations, @as(usize, 8));
    for (0..dot_warmup) |_| {
        dot_guard += runAttentionDotSweep(query, key_cache, key_scales, cfg.head_dim, scale_groups_per_head);
    }
    var dot_timer = try std.time.Timer.start();
    for (0..dot_iterations) |_| {
        dot_guard += runAttentionDotSweep(query, key_cache, key_scales, cfg.head_dim, scale_groups_per_head);
    }
    const dot_ns = dot_timer.read();
    try writer.print("  kernel=q8_dot ns_total={d} ns_iter={d:.3} melem_s={d:.3} guard={d:.6}\n", .{
        dot_ns,
        nsPerIteration(dot_ns, dot_iterations),
        millionElementsPerSecond(seq_len * cfg.head_dim, dot_iterations, dot_ns),
        dot_guard,
    });

    const axpy_output = try allocator.alloc(f32, cfg.head_dim);
    defer allocator.free(axpy_output);
    var axpy_guard: f32 = 0.0;
    const axpy_warmup = @min(dot_iterations, @as(usize, 8));
    for (0..axpy_warmup) |_| {
        @memset(axpy_output, 0.0);
        axpy_guard += runAttentionAxpySweep(axpy_output, value_cache, value_scales, cfg.head_dim, scale_groups_per_head);
    }
    var axpy_timer = try std.time.Timer.start();
    for (0..dot_iterations) |_| {
        @memset(axpy_output, 0.0);
        axpy_guard += runAttentionAxpySweep(axpy_output, value_cache, value_scales, cfg.head_dim, scale_groups_per_head);
    }
    const axpy_ns = axpy_timer.read();
    try writer.print("  kernel=q8_axpy ns_total={d} ns_iter={d:.3} melem_s={d:.3} guard={d:.6}\n", .{
        axpy_ns,
        nsPerIteration(axpy_ns, dot_iterations),
        millionElementsPerSecond(seq_len * cfg.head_dim, dot_iterations, axpy_ns),
        axpy_guard,
    });

    try benchAttentionFullProfile(allocator, writer, cfg, seq_len, full_iterations);
}

fn benchSwiGluProfile(
    allocator: std.mem.Allocator,
    writer: anytype,
    width: usize,
    requested_iterations: usize,
) !void {
    const iterations = resolveBenchIterations(requested_iterations, width, 16_000_000);

    const gate = try allocator.alloc(f32, width);
    defer allocator.free(gate);
    const up = try allocator.alloc(f32, width);
    defer allocator.free(up);
    const output = try allocator.alloc(f32, width);
    defer allocator.free(output);

    fillSyntheticF32(gate, 149);
    fillSyntheticF32(up, 167);

    try writer.print("profile: width={d} iterations={d}\n", .{ width, iterations });

    var guard: f32 = 0.0;
    const warmup = @min(iterations, @as(usize, 8));
    for (0..warmup) |_| {
        try cpu.swiglu(output, gate, up);
        guard += output[0] + output[output.len - 1];
    }
    var timer = try std.time.Timer.start();
    for (0..iterations) |_| {
        try cpu.swiglu(output, gate, up);
        guard += output[0] + output[output.len - 1];
    }
    const elapsed_ns = timer.read();
    try writer.print("  kernel=swiglu ns_total={d} ns_iter={d:.3} melem_s={d:.3} guard={d:.6}\n", .{
        elapsed_ns,
        nsPerIteration(elapsed_ns, iterations),
        millionElementsPerSecond(width, iterations, elapsed_ns),
        guard,
    });
}

fn benchRmsNormProfile(
    allocator: std.mem.Allocator,
    writer: anytype,
    label: []const u8,
    width: usize,
    requested_iterations: usize,
) !void {
    const iterations = resolveBenchIterations(requested_iterations, width, 16_000_000);
    const repeat_count = if (std.mem.eql(u8, label, "head")) @as(usize, 16) else @as(usize, 4);

    const input = try allocator.alloc(f32, width);
    defer allocator.free(input);
    const weight = try allocator.alloc(f32, width);
    defer allocator.free(weight);
    const output = try allocator.alloc(f32, width);
    defer allocator.free(output);

    fillSyntheticF32(input, 97);
    fillSyntheticF32(weight, 113);

    try writer.print("profile: {s} width={d} iterations={d}\n", .{ label, width, iterations });

    var guard: f32 = 0.0;
    const warmup = @min(iterations, @as(usize, 8));
    for (0..warmup) |_| {
        try cpu.rmsNorm(output, input, weight, 1e-5);
        guard += output[0] + output[output.len - 1];
    }
    var timer = try std.time.Timer.start();
    for (0..iterations) |_| {
        try cpu.rmsNorm(output, input, weight, 1e-5);
        guard += output[0] + output[output.len - 1];
    }
    const elapsed_ns = timer.read();
    try writer.print("  kernel=rmsnorm ns_total={d} ns_iter={d:.3} melem_s={d:.3} guard={d:.6}\n", .{
        elapsed_ns,
        nsPerIteration(elapsed_ns, iterations),
        millionElementsPerSecond(width, iterations, elapsed_ns),
        guard,
    });

    const repeated_input = try allocator.alloc(f32, width * repeat_count);
    defer allocator.free(repeated_input);
    const repeated_output = try allocator.alloc(f32, width * repeat_count);
    defer allocator.free(repeated_output);
    fillSyntheticF32(repeated_input, 131);

    guard = 0.0;
    for (0..warmup) |_| {
        try cpu.rmsNormRepeated(repeated_output, repeated_input, repeat_count, width, weight, 1e-5);
        guard += repeated_output[0] + repeated_output[repeated_output.len - 1];
    }
    timer = try std.time.Timer.start();
    for (0..iterations) |_| {
        try cpu.rmsNormRepeated(repeated_output, repeated_input, repeat_count, width, weight, 1e-5);
        guard += repeated_output[0] + repeated_output[repeated_output.len - 1];
    }
    const repeated_ns = timer.read();
    try writer.print(
        "  kernel=rmsnorm_repeated repeats={d} ns_total={d} ns_iter={d:.3} melem_s={d:.3} guard={d:.6}\n",
        .{
            repeat_count,
            repeated_ns,
            nsPerIteration(repeated_ns, iterations),
            millionElementsPerSecond(width * repeat_count, iterations, repeated_ns),
            guard,
        },
    );
}

fn benchAttentionFullProfile(
    allocator: std.mem.Allocator,
    writer: anytype,
    cfg: decoder_family.DecoderConfig,
    seq_len: usize,
    iterations: usize,
) !void {
    const scale_groups_per_head = try std.math.divCeil(usize, cfg.head_dim, attention.q8_cache_group_size);
    const total_query = cfg.num_attention_heads * cfg.head_dim;
    const total_cache = seq_len * cfg.num_key_value_heads * cfg.head_dim;
    const total_scales = seq_len * cfg.num_key_value_heads * scale_groups_per_head;
    const head_data_stride = seq_len * cfg.head_dim;
    const head_scale_stride = seq_len * scale_groups_per_head;

    const query = try allocator.alloc(f32, total_query);
    defer allocator.free(query);
    const output = try allocator.alloc(f32, total_query);
    defer allocator.free(output);
    const scores = try allocator.alloc(f32, seq_len);
    defer allocator.free(scores);
    const key_cache_token_major = try allocator.alloc(i8, total_cache);
    defer allocator.free(key_cache_token_major);
    const value_cache_token_major = try allocator.alloc(i8, total_cache);
    defer allocator.free(value_cache_token_major);
    const key_scales_token_major = try allocator.alloc(u16, total_scales);
    defer allocator.free(key_scales_token_major);
    const value_scales_token_major = try allocator.alloc(u16, total_scales);
    defer allocator.free(value_scales_token_major);
    const key_cache_head_major = try allocator.alloc(i8, total_cache);
    defer allocator.free(key_cache_head_major);
    const value_cache_head_major = try allocator.alloc(i8, total_cache);
    defer allocator.free(value_cache_head_major);
    const key_scales_head_major = try allocator.alloc(u16, total_scales);
    defer allocator.free(key_scales_head_major);
    const value_scales_head_major = try allocator.alloc(u16, total_scales);
    defer allocator.free(value_scales_head_major);

    fillSyntheticF32(query, 43);
    fillSyntheticQ8Cache(key_cache_token_major, key_scales_token_major, cfg.head_dim, 59);
    fillSyntheticQ8Cache(value_cache_token_major, value_scales_token_major, cfg.head_dim, 71);
    transposeQ8CacheTokenToHeadMajor(
        key_cache_head_major,
        key_scales_head_major,
        key_cache_token_major,
        key_scales_token_major,
        seq_len,
        cfg.num_key_value_heads,
        cfg.head_dim,
    );
    transposeQ8CacheTokenToHeadMajor(
        value_cache_head_major,
        value_scales_head_major,
        value_cache_token_major,
        value_scales_token_major,
        seq_len,
        cfg.num_key_value_heads,
        cfg.head_dim,
    );

    var guard: f32 = 0.0;
    const warmup = @min(iterations, @as(usize, 4));
    for (0..warmup) |_| {
        switch (optimized_kv_cache.default_q8_layout) {
            .token_major_legacy => try attention.scaledDotProductAttentionSingleQueryQ8Cache(
                output,
                query,
                key_cache_token_major,
                key_scales_token_major,
                value_cache_token_major,
                value_scales_token_major,
                seq_len,
                cfg.num_attention_heads,
                cfg.num_key_value_heads,
                cfg.head_dim,
                scores,
            ),
            .head_major => try attention.scaledDotProductAttentionSingleQueryQ8CacheHeadMajor(
                output,
                query,
                key_cache_head_major,
                key_scales_head_major,
                value_cache_head_major,
                value_scales_head_major,
                head_data_stride,
                head_scale_stride,
                seq_len,
                cfg.num_attention_heads,
                cfg.num_key_value_heads,
                cfg.head_dim,
                scores,
            ),
        }
        guard += output[0] + output[output.len - 1];
    }

    var timer = try std.time.Timer.start();
    for (0..iterations) |_| {
        switch (optimized_kv_cache.default_q8_layout) {
            .token_major_legacy => try attention.scaledDotProductAttentionSingleQueryQ8Cache(
                output,
                query,
                key_cache_token_major,
                key_scales_token_major,
                value_cache_token_major,
                value_scales_token_major,
                seq_len,
                cfg.num_attention_heads,
                cfg.num_key_value_heads,
                cfg.head_dim,
                scores,
            ),
            .head_major => try attention.scaledDotProductAttentionSingleQueryQ8CacheHeadMajor(
                output,
                query,
                key_cache_head_major,
                key_scales_head_major,
                value_cache_head_major,
                value_scales_head_major,
                head_data_stride,
                head_scale_stride,
                seq_len,
                cfg.num_attention_heads,
                cfg.num_key_value_heads,
                cfg.head_dim,
                scores,
            ),
        }
        guard += output[0] + output[output.len - 1];
    }
    const elapsed_ns = timer.read();
    try writer.print(
        "  kernel=q8_full seq_len={d} q_heads={d} kv_heads={d} ns_total={d} ns_iter={d:.3} melem_s={d:.3} guard={d:.6}\n",
        .{
            seq_len,
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            elapsed_ns,
            nsPerIteration(elapsed_ns, iterations),
            millionElementsPerSecond(seq_len * cfg.head_dim * cfg.num_attention_heads, iterations, elapsed_ns),
            guard,
        },
    );
}

fn benchScalarKernel(
    writer: anytype,
    kernel_name: []const u8,
    cols: usize,
    iterations: usize,
    kernel: fn ([]const u8, []const f32) f32,
    row: []const u8,
    input: []const f32,
) !void {
    var guard: f32 = 0.0;
    const warmup = @min(iterations, @as(usize, 8));
    for (0..warmup) |_| {
        guard += kernel(row, input);
    }

    var timer = try std.time.Timer.start();
    for (0..iterations) |_| {
        guard += kernel(row, input);
    }
    const elapsed_ns = timer.read();
    try writer.print("  kernel={s} ns_total={d} ns_iter={d:.3} melem_s={d:.3} guard={d:.6}\n", .{
        kernel_name,
        elapsed_ns,
        nsPerIteration(elapsed_ns, iterations),
        millionElementsPerSecond(cols, iterations, elapsed_ns),
        guard,
    });
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
        .thread_count = 1,
    };
}

fn fillSyntheticF32(output: []f32, salt: usize) void {
    for (output, 0..) |*value, idx| {
        const bucket = @as(i32, @intCast((idx * 17 + salt) % 31)) - 15;
        value.* = @as(f32, @floatFromInt(bucket)) / 8.0;
    }
}

fn fillSyntheticQ8Cache(
    values: []i8,
    scales: []u16,
    head_dim: usize,
    salt: usize,
) void {
    const groups_per_head = std.math.divCeil(usize, head_dim, attention.q8_cache_group_size) catch unreachable;
    for (values, 0..) |*value, idx| {
        const bucket = @as(i16, @intCast((idx * 11 + salt) % 255)) - 127;
        value.* = @intCast(bucket);
    }
    for (scales, 0..) |*scale, idx| {
        const group_idx = idx % groups_per_head;
        const magnitude = @as(f32, @floatFromInt(@as(u32, @intCast((group_idx + salt) % 13 + 1))));
        scale.* = bfloat16.fromF32(magnitude / 127.0);
    }
}

fn transposeQ8CacheTokenToHeadMajor(
    dst_values: []i8,
    dst_scales: []u16,
    src_values: []const i8,
    src_scales: []const u16,
    seq_len: usize,
    num_key_value_heads: usize,
    head_dim: usize,
) void {
    const groups_per_head = std.math.divCeil(usize, head_dim, attention.q8_cache_group_size) catch unreachable;
    const head_data_stride = seq_len * head_dim;
    const head_scale_stride = seq_len * groups_per_head;

    for (0..num_key_value_heads) |head_idx| {
        for (0..seq_len) |pos| {
            const token_major_data_start = (pos * num_key_value_heads + head_idx) * head_dim;
            const token_major_scale_start = (pos * num_key_value_heads + head_idx) * groups_per_head;
            const head_major_data_start = head_idx * head_data_stride + pos * head_dim;
            const head_major_scale_start = head_idx * head_scale_stride + pos * groups_per_head;

            @memcpy(
                dst_values[head_major_data_start .. head_major_data_start + head_dim],
                src_values[token_major_data_start .. token_major_data_start + head_dim],
            );
            @memcpy(
                dst_scales[head_major_scale_start .. head_major_scale_start + groups_per_head],
                src_scales[token_major_scale_start .. token_major_scale_start + groups_per_head],
            );
        }
    }
}

fn encodeF32Row(output: []u8, values: []const f32) void {
    for (values, 0..) |value, idx| {
        const start = idx * 4;
        std.mem.writeInt(u32, output[start .. start + 4][0..4], @bitCast(value), .little);
    }
}

fn encodeBf16Row(output: []u8, values: []const f32) void {
    for (values, 0..) |value, idx| {
        const start = idx * 2;
        std.mem.writeInt(u16, output[start .. start + 2][0..2], bfloat16.fromF32(value), .little);
    }
}

fn runAttentionDotSweep(
    query: []const f32,
    key_cache: []const i8,
    key_scales: []const u16,
    head_dim: usize,
    scale_groups_per_head: usize,
) f32 {
    var sum: f32 = 0.0;
    var cache_start: usize = 0;
    var scale_start: usize = 0;
    while (cache_start < key_cache.len) : ({
        cache_start += head_dim;
        scale_start += scale_groups_per_head;
    }) {
        sum += attention.dotQ8GroupedSlice(
            query,
            key_cache[cache_start .. cache_start + head_dim],
            key_scales[scale_start .. scale_start + scale_groups_per_head],
        );
    }
    return sum;
}

fn runAttentionAxpySweep(
    output: []f32,
    value_cache: []const i8,
    value_scales: []const u16,
    head_dim: usize,
    scale_groups_per_head: usize,
) f32 {
    var cache_start: usize = 0;
    var scale_start: usize = 0;
    while (cache_start < value_cache.len) : ({
        cache_start += head_dim;
        scale_start += scale_groups_per_head;
    }) {
        attention.axpyQ8GroupedSliceInPlace(
            output,
            1.0,
            value_cache[cache_start .. cache_start + head_dim],
            value_scales[scale_start .. scale_start + scale_groups_per_head],
        );
    }
    return output[0] + output[output.len - 1];
}

fn resolveBenchIterations(requested_iterations: usize, work_per_iteration: usize, target_elements: usize) usize {
    if (requested_iterations != 0) return requested_iterations;
    const safe_work = @max(work_per_iteration, 1);
    const auto_iterations = target_elements / safe_work;
    return @max(@as(usize, 8), @min(@as(usize, 4096), auto_iterations));
}

fn nsPerIteration(elapsed_ns: u64, iterations: usize) f64 {
    if (iterations == 0) return 0.0;
    return @as(f64, @floatFromInt(elapsed_ns)) / @as(f64, @floatFromInt(iterations));
}

fn millionElementsPerSecond(work_per_iteration: usize, iterations: usize, elapsed_ns: u64) f64 {
    if (work_per_iteration == 0 or iterations == 0 or elapsed_ns == 0) return 0.0;
    const total_elements = @as(f64, @floatFromInt(work_per_iteration)) * @as(f64, @floatFromInt(iterations));
    return total_elements / @as(f64, @floatFromInt(elapsed_ns)) * 1_000.0;
}
