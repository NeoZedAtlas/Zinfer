const std = @import("std");
const GenerateOptions = @import("../args.zig").GenerateOptions;
const optimized_kv_cache = @import("../../../model/runtime/optimized_kv_cache.zig");
const decoder_family = @import("../../../model/runtime/decoder_family.zig");
const optimized_decoder = @import("../../../model/runtime/optimized_decoder.zig");
const tensor_backend = @import("../../../tensor/backends/backend.zig");
const sampler = @import("../../../sampling/sampler.zig");
const streaming = @import("streaming.zig");

pub const GeneratorRuntime = struct {
    allocator: std.mem.Allocator,
    tokenizer: decoder_family.Tokenizer,
    model: optimized_decoder.Runtime,

    pub fn init(
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

    pub fn deinit(self: *GeneratorRuntime) void {
        self.model.deinit();
        self.tokenizer.deinit();
    }

    pub fn generateFromPrompt(
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
        const resolved_kv_cache_scheme = optimized_kv_cache.resolveScheme(options.kv_cache_scheme, self.model.backendName());
        var cache = try optimized_kv_cache.ModelCache.initWithLayout(
            self.allocator,
            cfg.num_hidden_layers,
            prompt_ids.len + options.max_new_tokens,
            cfg.num_key_value_heads,
            cfg.head_dim,
            resolved_kv_cache_scheme,
            options.q8_layout,
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
            if (try streaming.analyzeAndMaybeStream(self.allocator, &self.tokenizer, generated.items, effective_options, stdout, &streamed_len)) |trimmed| {
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
