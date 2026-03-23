const std = @import("std");
const cpu = @import("../../kernel/core/cpu.zig");
const decoder_family = @import("decoder_family.zig");
const generic_block = @import("../layers/rmsnorm_gqa_swiglu_block.zig");
const gqa_attention = @import("../layers/gqa_attention.zig");
const optimized_kv_cache = @import("optimized_kv_cache.zig");
const optimized_decoder_support = @import("optimized_decoder/support.zig");
const workspace_mod = @import("optimized_decoder/workspace.zig");
const tensor_backend = @import("../../tensor/backends/backend.zig");
const parallel_rows = @import("../../tensor/parallel/parallel_rows.zig");

pub const Workspace = workspace_mod.Workspace;

pub const Runtime = struct {
    allocator: std.mem.Allocator,
    cfg: decoder_family.DecoderConfig,
    backend: tensor_backend.Backend,
    common_weights: decoder_family.CommonWeights,
    layer_layout: generic_block.LayerLayout,
    embed_tokens_tensor: tensor_backend.Backend.TensorHandle,
    lm_head_tensor: tensor_backend.Backend.TensorHandle,
    layers: []LayerWeights,
    final_norm_weight: []f32,
    thread_count: usize,
    parallel_pool: parallel_rows.Pool,

    pub fn init(
        allocator: std.mem.Allocator,
        model_dir: []const u8,
        scheme: tensor_backend.Scheme,
        thread_count: ?usize,
    ) !Runtime {
        const config_path = try std.fs.path.join(allocator, &.{ model_dir, "config.json" });
        defer allocator.free(config_path);

        var parsed_config = try decoder_family.loadConfigFromFile(allocator, config_path);
        errdefer parsed_config.deinit();
        const cfg = parsed_config.value;

        const resolved_thread_count = thread_count orelse @max(1, std.Thread.getCpuCount() catch 1);

        var backend = try tensor_backend.Backend.openFromModelDir(allocator, model_dir, scheme);
        errdefer backend.deinit();

        var parallel_pool = try parallel_rows.Pool.init(allocator, resolved_thread_count);
        errdefer parallel_pool.deinit();

        const common_weights = decoder_family.commonWeights(cfg.architecture);
        const layer_layout = decoder_family.layerLayout(cfg.architecture);
        const io_scratch_bytes = optimized_decoder_support.maxIoScratchBytes(cfg);
        const embed_tokens_tensor = try backend.resolveTensor(common_weights.embed_tokens_weight);
        const lm_head_tensor = try backend.resolveTensor(common_weights.lm_head_weight);

        const final_norm_weight = try optimized_decoder_support.allocVector(&backend, allocator, common_weights.final_norm_weight, cfg.hidden_size, io_scratch_bytes);
        errdefer allocator.free(final_norm_weight);

        const layers = try allocator.alloc(LayerWeights, cfg.num_hidden_layers);
        errdefer allocator.free(layers);

        var initialized_layers: usize = 0;
        errdefer {
            for (layers[0..initialized_layers]) |*layer| layer.deinit(allocator);
        }

        for (layers, 0..) |*layer, layer_index| {
            layer.* = try LayerWeights.init(allocator, &backend, cfg, layer_index, layer_layout, io_scratch_bytes);
            initialized_layers += 1;
        }

        const runtime = Runtime{
            .allocator = allocator,
            .cfg = cfg,
            .backend = backend,
            .common_weights = common_weights,
            .layer_layout = layer_layout,
            .embed_tokens_tensor = embed_tokens_tensor,
            .lm_head_tensor = lm_head_tensor,
            .layers = layers,
            .final_norm_weight = final_norm_weight,
            .thread_count = resolved_thread_count,
            .parallel_pool = parallel_pool,
        };

        parsed_config.deinit();
        return runtime;
    }

    pub fn deinit(self: *Runtime) void {
        for (self.layers) |*layer| layer.deinit(self.allocator);
        self.allocator.free(self.layers);
        self.allocator.free(self.final_norm_weight);
        self.parallel_pool.deinit();
        self.backend.deinit();
    }

    pub fn initWorkspace(self: *const Runtime, max_seq_len: usize) !Workspace {
        return try Workspace.init(self.allocator, self.cfg, max_seq_len, optimized_decoder_support.maxIoScratchBytes(self.cfg));
    }

    pub fn forwardTokenId(
        self: *Runtime,
        workspace: *Workspace,
        cache: *optimized_kv_cache.ModelCache,
        token_id: usize,
    ) ![]f32 {
        if (token_id >= self.cfg.vocab_size) return error.TokenIdOutOfBounds;
        try self.backend.readRowIntoTensor(
            self.embed_tokens_tensor,
            token_id,
            workspace.hidden_a,
            workspace.io_scratch,
        );

        var hidden_in = workspace.hidden_a;
        var hidden_out = workspace.hidden_b;
        for (self.layers, 0..) |*layer, layer_index| {
            try layer.forward(
                self,
                workspace,
                &cache.layers[layer_index],
                hidden_in,
                hidden_out,
            );
            std.mem.swap([]f32, &hidden_in, &hidden_out);
        }

        try cpu.rmsNorm(
            workspace.normed,
            hidden_in,
            self.final_norm_weight,
            @floatCast(self.cfg.rms_norm_eps),
        );

        try self.backend.matmulVec(
            workspace.logits,
            self.lm_head_tensor,
            workspace.normed,
            self.thread_count,
            &self.parallel_pool,
            workspace.io_scratch,
        );
        return workspace.logits;
    }

    pub fn prefillTokenIds(
        self: *Runtime,
        workspace: *Workspace,
        cache: *optimized_kv_cache.ModelCache,
        token_ids: []const usize,
    ) ![]f32 {
        if (token_ids.len == 0) return error.EmptyPrompt;

        var last_logits: []f32 = undefined;
        for (token_ids) |token_id| {
            last_logits = try self.forwardTokenId(workspace, cache, token_id);
        }
        return last_logits;
    }

    pub fn backendName(self: Runtime) []const u8 {
        return self.backend.resolvedScheme().name();
    }

    pub fn artifactBytes(self: *const Runtime) u64 {
        return self.backend.artifactBytes();
    }
};

const LayerWeights = struct {
    spec: generic_block.Spec,
    input_ln_weight: []f32,
    post_ln_weight: []f32,
    q_norm_weight: ?[]f32,
    k_norm_weight: ?[]f32,
    q_proj_tensor: tensor_backend.Backend.TensorHandle,
    k_proj_tensor: tensor_backend.Backend.TensorHandle,
    v_proj_tensor: tensor_backend.Backend.TensorHandle,
    o_proj_tensor: tensor_backend.Backend.TensorHandle,
    gate_proj_tensor: tensor_backend.Backend.TensorHandle,
    up_proj_tensor: tensor_backend.Backend.TensorHandle,
    down_proj_tensor: tensor_backend.Backend.TensorHandle,

    fn init(
        allocator: std.mem.Allocator,
        backend: *tensor_backend.Backend,
        cfg: decoder_family.DecoderConfig,
        layer_index: usize,
        layout: generic_block.LayerLayout,
        io_scratch_bytes: usize,
    ) !LayerWeights {
        const spec = generic_block.Spec{
            .layer_index = layer_index,
            .hidden_size = cfg.hidden_size,
            .intermediate_size = cfg.intermediate_size,
            .num_attention_heads = cfg.num_attention_heads,
            .num_key_value_heads = cfg.num_key_value_heads,
            .head_dim = cfg.head_dim,
            .rope_theta = @floatCast(cfg.rope_theta),
            .rms_norm_eps = @floatCast(cfg.rms_norm_eps),
        };

        const input_ln_name = try decoder_family.layerTensorNameAlloc(allocator, cfg.architecture, layer_index, layout.input_layernorm_kind);
        defer allocator.free(input_ln_name);
        const input_ln_weight = try optimized_decoder_support.allocVector(backend, allocator, input_ln_name, cfg.hidden_size, io_scratch_bytes);
        errdefer allocator.free(input_ln_weight);

        const post_ln_name = try decoder_family.layerTensorNameAlloc(allocator, cfg.architecture, layer_index, layout.post_attention_layernorm_kind);
        defer allocator.free(post_ln_name);
        const post_ln_weight = try optimized_decoder_support.allocVector(backend, allocator, post_ln_name, cfg.hidden_size, io_scratch_bytes);
        errdefer allocator.free(post_ln_weight);

        const q_norm_weight = if (layout.q_norm_kind) |kind|
            try blk: {
                const name = try decoder_family.layerTensorNameAlloc(allocator, cfg.architecture, layer_index, kind);
                defer allocator.free(name);
                break :blk optimized_decoder_support.allocVector(backend, allocator, name, cfg.head_dim, io_scratch_bytes);
            }
        else
            null;
        errdefer if (q_norm_weight) |buffer| allocator.free(buffer);

        const k_norm_weight = if (layout.k_norm_kind) |kind|
            try blk: {
                const name = try decoder_family.layerTensorNameAlloc(allocator, cfg.architecture, layer_index, kind);
                defer allocator.free(name);
                break :blk optimized_decoder_support.allocVector(backend, allocator, name, cfg.head_dim, io_scratch_bytes);
            }
        else
            null;
        errdefer if (k_norm_weight) |buffer| allocator.free(buffer);

        return .{
            .spec = spec,
            .input_ln_weight = input_ln_weight,
            .post_ln_weight = post_ln_weight,
            .q_norm_weight = q_norm_weight,
            .k_norm_weight = k_norm_weight,
            .q_proj_tensor = try optimized_decoder_support.resolveMatrixTensor(backend, allocator, cfg, layer_index, layout.q_proj_kind),
            .k_proj_tensor = try optimized_decoder_support.resolveMatrixTensor(backend, allocator, cfg, layer_index, layout.k_proj_kind),
            .v_proj_tensor = try optimized_decoder_support.resolveMatrixTensor(backend, allocator, cfg, layer_index, layout.v_proj_kind),
            .o_proj_tensor = try optimized_decoder_support.resolveMatrixTensor(backend, allocator, cfg, layer_index, layout.o_proj_kind),
            .gate_proj_tensor = try optimized_decoder_support.resolveMatrixTensor(backend, allocator, cfg, layer_index, layout.gate_proj_kind),
            .up_proj_tensor = try optimized_decoder_support.resolveMatrixTensor(backend, allocator, cfg, layer_index, layout.up_proj_kind),
            .down_proj_tensor = try optimized_decoder_support.resolveMatrixTensor(backend, allocator, cfg, layer_index, layout.down_proj_kind),
        };
    }

    fn deinit(self: *LayerWeights, allocator: std.mem.Allocator) void {
        allocator.free(self.input_ln_weight);
        allocator.free(self.post_ln_weight);
        if (self.q_norm_weight) |buffer| allocator.free(buffer);
        if (self.k_norm_weight) |buffer| allocator.free(buffer);
    }

    fn forward(
        self: *const LayerWeights,
        runtime: *Runtime,
        workspace: *Workspace,
        cache: *optimized_kv_cache.LayerKVCache,
        hidden_in: []const f32,
        hidden_out: []f32,
    ) !void {
        try cpu.rmsNorm(workspace.normed, hidden_in, self.input_ln_weight, self.spec.rms_norm_eps);

        try runtime.backend.matmulVec(workspace.q_proj, self.q_proj_tensor, workspace.normed, runtime.thread_count, &runtime.parallel_pool, workspace.io_scratch);
        try runtime.backend.matmulVec(workspace.k_proj, self.k_proj_tensor, workspace.normed, runtime.thread_count, &runtime.parallel_pool, workspace.io_scratch);
        try runtime.backend.matmulVec(workspace.v_proj, self.v_proj_tensor, workspace.normed, runtime.thread_count, &runtime.parallel_pool, workspace.io_scratch);

        if (self.q_norm_weight) |weight| {
            try cpu.rmsNormRepeated(workspace.q_proj, workspace.q_proj, self.spec.num_attention_heads, self.spec.head_dim, weight, self.spec.rms_norm_eps);
        }

        if (self.k_norm_weight) |weight| {
            try cpu.rmsNormRepeated(workspace.k_proj, workspace.k_proj, self.spec.num_key_value_heads, self.spec.head_dim, weight, self.spec.rms_norm_eps);
        }

        const position = cache.len;
        try gqa_attention.applyRoPEToProjectedHeadsWithTableInPlace(
            self.spec.attentionSpec(),
            workspace.q_proj,
            workspace.k_proj,
            &workspace.rope_table,
            position,
        );
        try cache.append(workspace.k_proj, workspace.v_proj);

        switch (cache.scheme) {
            .auto => unreachable,
            .bf16 => try gqa_attention.forwardProjectedSingleTokenBf16Cache(
                self.spec.attentionSpec(),
                workspace.attn_flat,
                workspace.q_proj,
                cache.currentBf16Keys(),
                cache.currentBf16Values(),
                cache.len,
                workspace.scores[0..cache.len],
            ),
            .q8 => switch (cache.q8_layout) {
                .token_major_legacy => try gqa_attention.forwardProjectedSingleTokenQ8Cache(
                    self.spec.attentionSpec(),
                    workspace.attn_flat,
                    workspace.q_proj,
                    cache.currentQ8Keys(),
                    cache.currentQ8KeyScales(),
                    cache.currentQ8Values(),
                    cache.currentQ8ValueScales(),
                    cache.len,
                    workspace.scores[0..cache.len],
                ),
                .head_major => try gqa_attention.forwardProjectedSingleTokenQ8CacheHeadMajor(
                    self.spec.attentionSpec(),
                    workspace.attn_flat,
                    workspace.q_proj,
                    cache.q8KeysHeadMajor(),
                    cache.q8KeyScalesHeadMajor(),
                    cache.q8ValuesHeadMajor(),
                    cache.q8ValueScalesHeadMajor(),
                    cache.q8HeadDataStride(),
                    cache.q8HeadScaleStride(),
                    cache.len,
                    workspace.scores[0..cache.len],
                ),
            },
        }

        try runtime.backend.matmulVec(workspace.attn_out, self.o_proj_tensor, workspace.attn_flat, runtime.thread_count, &runtime.parallel_pool, workspace.io_scratch);
        for (workspace.post_attn, hidden_in, workspace.attn_out) |*out, residual, attn_value| {
            out.* = residual + attn_value;
        }

        try cpu.rmsNorm(workspace.normed, workspace.post_attn, self.post_ln_weight, self.spec.rms_norm_eps);
        try runtime.backend.matmulVec(workspace.gate, self.gate_proj_tensor, workspace.normed, runtime.thread_count, &runtime.parallel_pool, workspace.io_scratch);
        try runtime.backend.matmulVec(workspace.up, self.up_proj_tensor, workspace.normed, runtime.thread_count, &runtime.parallel_pool, workspace.io_scratch);
        try cpu.swiglu(workspace.gate, workspace.gate, workspace.up);
        try runtime.backend.matmulVec(workspace.mlp_out, self.down_proj_tensor, workspace.gate, runtime.thread_count, &runtime.parallel_pool, workspace.io_scratch);

        for (hidden_out, workspace.post_attn, workspace.mlp_out) |*out, residual, mlp_value| {
            out.* = residual + mlp_value;
        }
    }
};
