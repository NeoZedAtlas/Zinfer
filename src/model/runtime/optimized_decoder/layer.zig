const std = @import("std");
const cpu = @import("../../../kernel/core/cpu.zig");
const decoder_family = @import("../decoder_family.zig");
const generic_block = @import("../../layers/rmsnorm_gqa_swiglu_block.zig");
const gqa_attention = @import("../../layers/gqa_attention.zig");
const optimized_kv_cache = @import("../optimized_kv_cache.zig");
const optimized_decoder_support = @import("support.zig");
const tensor_backend = @import("../../../tensor/backends/backend.zig");
const parallel_rows = @import("../../../tensor/parallel/parallel_rows.zig");
const workspace_mod = @import("workspace.zig");

pub const LayerWeights = struct {
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

    pub fn init(
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

    pub fn deinit(self: *LayerWeights, allocator: std.mem.Allocator) void {
        allocator.free(self.input_ln_weight);
        allocator.free(self.post_ln_weight);
        if (self.q_norm_weight) |buffer| allocator.free(buffer);
        if (self.k_norm_weight) |buffer| allocator.free(buffer);
    }

    pub fn forward(
        self: *const LayerWeights,
        backend: *tensor_backend.Backend,
        thread_count: usize,
        parallel_pool: *parallel_rows.Pool,
        workspace: *workspace_mod.Workspace,
        cache: *optimized_kv_cache.LayerKVCache,
        hidden_in: []const f32,
        hidden_out: []f32,
    ) !void {
        try cpu.rmsNorm(workspace.normed, hidden_in, self.input_ln_weight, self.spec.rms_norm_eps);

        try backend.matmulVec(workspace.q_proj, self.q_proj_tensor, workspace.normed, thread_count, parallel_pool, workspace.io_scratch);
        try backend.matmulVec(workspace.k_proj, self.k_proj_tensor, workspace.normed, thread_count, parallel_pool, workspace.io_scratch);
        try backend.matmulVec(workspace.v_proj, self.v_proj_tensor, workspace.normed, thread_count, parallel_pool, workspace.io_scratch);

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
                .paged_head_major => try gqa_attention.forwardProjectedSingleTokenQ8CachePagedHeadMajor(
                    self.spec.attentionSpec(),
                    workspace.attn_flat,
                    workspace.q_proj,
                    cache.q8KeysPagedHeadMajor(),
                    cache.q8KeyScalesPagedHeadMajor(),
                    cache.q8ValuesPagedHeadMajor(),
                    cache.q8ValueScalesPagedHeadMajor(),
                    cache.q8PagedHeadStride(),
                    cache.q8PagedScaleHeadStride(),
                    cache.q8PageDataStride(),
                    cache.q8PageScaleStride(),
                    cache.q8PageLen(),
                    cache.q8PagesPerHead(),
                    cache.len,
                    workspace.scores[0..cache.len],
                ),
            },
        }

        try backend.matmulVec(workspace.attn_out, self.o_proj_tensor, workspace.attn_flat, thread_count, parallel_pool, workspace.io_scratch);
        for (workspace.post_attn, hidden_in, workspace.attn_out) |*out, residual, attn_value| {
            out.* = residual + attn_value;
        }

        try cpu.rmsNorm(workspace.normed, workspace.post_attn, self.post_ln_weight, self.spec.rms_norm_eps);
        try backend.matmulVec(workspace.gate, self.gate_proj_tensor, workspace.normed, thread_count, parallel_pool, workspace.io_scratch);
        try backend.matmulVec(workspace.up, self.up_proj_tensor, workspace.normed, thread_count, parallel_pool, workspace.io_scratch);
        try cpu.swiglu(workspace.gate, workspace.gate, workspace.up);
        try backend.matmulVec(workspace.mlp_out, self.down_proj_tensor, workspace.gate, thread_count, parallel_pool, workspace.io_scratch);

        for (hidden_out, workspace.post_attn, workspace.mlp_out) |*out, residual, mlp_value| {
            out.* = residual + mlp_value;
        }
    }
};
