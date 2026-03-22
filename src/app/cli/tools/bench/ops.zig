const std = @import("std");
const attention = @import("../../../../kernel/attention/attention.zig");
const cpu = @import("../../../../kernel/core/cpu.zig");
const bfloat16 = @import("../../../../tensor/formats/bfloat16.zig");
const optimized_kv_cache = @import("../../../../model/runtime/optimized_kv_cache.zig");
const decoder_family = @import("../../../../model/runtime/decoder_family.zig");
const quantized = @import("../../../../tensor/formats/quantized.zig");
const tensor_store = @import("../../../../tensor/storage/store.zig");
const support = @import("support.zig");

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
    try stdout.print("\n[matmul-vec]\n", .{});
    try benchQuantizedMatmulProfile(allocator, stdout, "attn_proj", cfg.hidden_size, cfg.hidden_size, requested_iterations);
    try benchQuantizedMatmulProfile(allocator, stdout, "mlp_expand", cfg.intermediate_size, cfg.hidden_size, requested_iterations);
    try benchQuantizedMatmulProfile(allocator, stdout, "mlp_down", cfg.hidden_size, cfg.intermediate_size, requested_iterations);
    try stdout.print("\n[rmsnorm]\n", .{});
    try benchRmsNormProfile(allocator, stdout, "hidden", cfg.hidden_size, requested_iterations);
    try benchRmsNormProfile(allocator, stdout, "head", cfg.head_dim, requested_iterations);
    try stdout.print("\n[swiglu]\n", .{});
    try benchSwiGluProfile(allocator, stdout, cfg.intermediate_size, requested_iterations);
    try stdout.print("\n[attention-q8-cache]\n", .{});
    try benchAttentionProfile(allocator, stdout, cfg, requested_iterations);
}

fn benchGemvProfile(
    allocator: std.mem.Allocator,
    writer: anytype,
    label: []const u8,
    cols: usize,
    requested_iterations: usize,
) !void {
    const iterations = support.resolveBenchIterations(requested_iterations, cols, 32_000_000);
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

    support.fillSyntheticF32(input, 13);
    support.fillSyntheticF32(row_values, 29);
    support.encodeF32Row(row_f32, row_values);
    support.encodeBf16Row(row_bf16, row_values);
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
    const dot_iterations = support.resolveBenchIterations(requested_iterations, seq_len * cfg.head_dim, 16_000_000);
    const full_iterations = support.resolveBenchIterations(
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

    support.fillSyntheticF32(query, 7);
    support.fillSyntheticQ8Cache(key_cache, key_scales, cfg.head_dim, 17);
    support.fillSyntheticQ8Cache(value_cache, value_scales, cfg.head_dim, 31);

    try writer.print("profile: single-kv-head seq_len={d} head_dim={d} iterations={d}\n", .{
        seq_len,
        cfg.head_dim,
        dot_iterations,
    });

    var dot_guard: f32 = 0.0;
    const dot_warmup = @min(dot_iterations, @as(usize, 8));
    for (0..dot_warmup) |_| {
        dot_guard += support.runAttentionDotSweep(query, key_cache, key_scales, cfg.head_dim, scale_groups_per_head);
    }
    var dot_timer = try std.time.Timer.start();
    for (0..dot_iterations) |_| {
        dot_guard += support.runAttentionDotSweep(query, key_cache, key_scales, cfg.head_dim, scale_groups_per_head);
    }
    const dot_ns = dot_timer.read();
    try writer.print("  kernel=q8_dot ns_total={d} ns_iter={d:.3} melem_s={d:.3} guard={d:.6}\n", .{
        dot_ns,
        support.nsPerIteration(dot_ns, dot_iterations),
        support.millionElementsPerSecond(seq_len * cfg.head_dim, dot_iterations, dot_ns),
        dot_guard,
    });

    const axpy_output = try allocator.alloc(f32, cfg.head_dim);
    defer allocator.free(axpy_output);
    var axpy_guard: f32 = 0.0;
    const axpy_warmup = @min(dot_iterations, @as(usize, 8));
    for (0..axpy_warmup) |_| {
        @memset(axpy_output, 0.0);
        axpy_guard += support.runAttentionAxpySweep(axpy_output, value_cache, value_scales, cfg.head_dim, scale_groups_per_head);
    }
    var axpy_timer = try std.time.Timer.start();
    for (0..dot_iterations) |_| {
        @memset(axpy_output, 0.0);
        axpy_guard += support.runAttentionAxpySweep(axpy_output, value_cache, value_scales, cfg.head_dim, scale_groups_per_head);
    }
    const axpy_ns = axpy_timer.read();
    try writer.print("  kernel=q8_axpy ns_total={d} ns_iter={d:.3} melem_s={d:.3} guard={d:.6}\n", .{
        axpy_ns,
        support.nsPerIteration(axpy_ns, dot_iterations),
        support.millionElementsPerSecond(seq_len * cfg.head_dim, dot_iterations, axpy_ns),
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
    const iterations = support.resolveBenchIterations(requested_iterations, width, 16_000_000);

    const gate = try allocator.alloc(f32, width);
    defer allocator.free(gate);
    const up = try allocator.alloc(f32, width);
    defer allocator.free(up);
    const output = try allocator.alloc(f32, width);
    defer allocator.free(output);

    support.fillSyntheticF32(gate, 149);
    support.fillSyntheticF32(up, 167);

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
        support.nsPerIteration(elapsed_ns, iterations),
        support.millionElementsPerSecond(width, iterations, elapsed_ns),
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
    const iterations = support.resolveBenchIterations(requested_iterations, width, 16_000_000);
    const repeat_count = if (std.mem.eql(u8, label, "head")) @as(usize, 16) else @as(usize, 4);

    const input = try allocator.alloc(f32, width);
    defer allocator.free(input);
    const weight = try allocator.alloc(f32, width);
    defer allocator.free(weight);
    const output = try allocator.alloc(f32, width);
    defer allocator.free(output);

    support.fillSyntheticF32(input, 97);
    support.fillSyntheticF32(weight, 113);

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
        support.nsPerIteration(elapsed_ns, iterations),
        support.millionElementsPerSecond(width, iterations, elapsed_ns),
        guard,
    });

    const repeated_input = try allocator.alloc(f32, width * repeat_count);
    defer allocator.free(repeated_input);
    const repeated_output = try allocator.alloc(f32, width * repeat_count);
    defer allocator.free(repeated_output);
    support.fillSyntheticF32(repeated_input, 131);

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
            support.nsPerIteration(repeated_ns, iterations),
            support.millionElementsPerSecond(width * repeat_count, iterations, repeated_ns),
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

    support.fillSyntheticF32(query, 43);
    support.fillSyntheticQ8Cache(key_cache_token_major, key_scales_token_major, cfg.head_dim, 59);
    support.fillSyntheticQ8Cache(value_cache_token_major, value_scales_token_major, cfg.head_dim, 71);
    support.transposeQ8CacheTokenToHeadMajor(
        key_cache_head_major,
        key_scales_head_major,
        key_cache_token_major,
        key_scales_token_major,
        seq_len,
        cfg.num_key_value_heads,
        cfg.head_dim,
    );
    support.transposeQ8CacheTokenToHeadMajor(
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
            support.nsPerIteration(elapsed_ns, iterations),
            support.millionElementsPerSecond(seq_len * cfg.head_dim * cfg.num_attention_heads, iterations, elapsed_ns),
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
        support.nsPerIteration(elapsed_ns, iterations),
        support.millionElementsPerSecond(cols, iterations, elapsed_ns),
        guard,
    });
}

fn benchQuantizedMatmulProfile(
    allocator: std.mem.Allocator,
    writer: anytype,
    profile_name: []const u8,
    rows: usize,
    cols: usize,
    requested_iterations: usize,
) !void {
    const iterations = support.resolveBenchIterations(requested_iterations, rows * cols, 24_000_000);
    const row_bytes_q8 = 4 + cols;
    const row_bytes_q6 = 4 + (try std.math.divCeil(usize, cols * 6, 8));
    const row_bytes_q4 = 4 + (try std.math.divCeil(usize, cols, 2));

    const input = try allocator.alloc(f32, cols);
    defer allocator.free(input);
    const row_values = try allocator.alloc(f32, cols);
    defer allocator.free(row_values);
    const output = try allocator.alloc(f32, rows);
    defer allocator.free(output);
    const matrix_q8 = try allocator.alloc(u8, rows * row_bytes_q8);
    defer allocator.free(matrix_q8);
    const matrix_q6 = try allocator.alloc(u8, rows * row_bytes_q6);
    defer allocator.free(matrix_q6);
    const matrix_q4 = try allocator.alloc(u8, rows * row_bytes_q4);
    defer allocator.free(matrix_q4);

    support.fillSyntheticF32(input, 211);
    for (0..rows) |row_idx| {
        for (row_values, 0..) |*value, col_idx| {
            const bucket = @as(i32, @intCast((row_idx * 29 + col_idx * 13 + 5) % 43)) - 21;
            value.* = @as(f32, @floatFromInt(bucket)) / 9.0;
        }
        quantized.encodeQ8Row(matrix_q8[row_idx * row_bytes_q8 .. (row_idx + 1) * row_bytes_q8], row_values);
        quantized.encodeQ6Row(matrix_q6[row_idx * row_bytes_q6 .. (row_idx + 1) * row_bytes_q6], row_values);
        quantized.encodeQ4Row(matrix_q4[row_idx * row_bytes_q4 .. (row_idx + 1) * row_bytes_q4], row_values);
    }

    try writer.print("profile: {s} rows={d} cols={d} iterations={d}\n", .{
        profile_name,
        rows,
        cols,
        iterations,
    });

    try benchMatrixKernel(writer, "q8", rows, cols, iterations, struct {
        fn run(output_buf: []f32, matrix: []const u8, row_bytes: usize, vector: []const f32) void {
            quantized.matmulQ8Rows(output_buf, matrix, row_bytes, vector);
        }
    }.run, output, matrix_q8, row_bytes_q8, input);
    try benchMatrixKernel(writer, "q6", rows, cols, iterations, struct {
        fn run(output_buf: []f32, matrix: []const u8, row_bytes: usize, vector: []const f32) void {
            quantized.matmulQ6Rows(output_buf, matrix, row_bytes, vector);
        }
    }.run, output, matrix_q6, row_bytes_q6, input);
    try benchMatrixKernel(writer, "q4", rows, cols, iterations, struct {
        fn run(output_buf: []f32, matrix: []const u8, row_bytes: usize, vector: []const f32) void {
            quantized.matmulQ4Rows(output_buf, matrix, row_bytes, vector);
        }
    }.run, output, matrix_q4, row_bytes_q4, input);
}

fn benchMatrixKernel(
    writer: anytype,
    kernel_name: []const u8,
    rows: usize,
    cols: usize,
    iterations: usize,
    kernel: fn ([]f32, []const u8, usize, []const f32) void,
    output: []f32,
    matrix: []const u8,
    row_bytes: usize,
    input: []const f32,
) !void {
    var guard: f32 = 0.0;
    const warmup = @min(iterations, @as(usize, 4));
    for (0..warmup) |_| {
        kernel(output, matrix, row_bytes, input);
        guard += output[0] + output[output.len - 1];
    }

    var timer = try std.time.Timer.start();
    for (0..iterations) |_| {
        kernel(output, matrix, row_bytes, input);
        guard += output[0] + output[output.len - 1];
    }
    const elapsed_ns = timer.read();
    try writer.print("  kernel={s} ns_total={d} ns_iter={d:.3} melem_s={d:.3} guard={d:.6}\n", .{
        kernel_name,
        elapsed_ns,
        support.nsPerIteration(elapsed_ns, iterations),
        support.millionElementsPerSecond(rows * cols, iterations, elapsed_ns),
        guard,
    });
}
