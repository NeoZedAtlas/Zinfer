const std = @import("std");
const cpu = @import("../../../kernel/core/cpu.zig");
const decoder_family = @import("../decoder_family.zig");
const generic_block = @import("../../layers/rmsnorm_gqa_swiglu_block.zig");
const optimized_kv_cache = @import("../optimized_kv_cache.zig");
const layer_mod = @import("layer.zig");
const optimized_decoder_support = @import("support.zig");
const workspace_mod = @import("workspace.zig");
const tensor_backend = @import("../../../tensor/backends/backend.zig");
const parallel_rows = @import("../../../tensor/parallel/parallel_rows.zig");

pub const Runtime = struct {
    allocator: std.mem.Allocator,
    cfg: decoder_family.DecoderConfig,
    backend: tensor_backend.Backend,
    common_weights: decoder_family.CommonWeights,
    layer_layout: generic_block.LayerLayout,
    embed_tokens_tensor: tensor_backend.Backend.TensorHandle,
    lm_head_tensor: tensor_backend.Backend.TensorHandle,
    layers: []layer_mod.LayerWeights,
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

        const layers = try allocator.alloc(layer_mod.LayerWeights, cfg.num_hidden_layers);
        errdefer allocator.free(layers);

        var initialized_layers: usize = 0;
        errdefer {
            for (layers[0..initialized_layers]) |*layer| layer.deinit(allocator);
        }

        for (layers, 0..) |*layer, layer_index| {
            layer.* = try layer_mod.LayerWeights.init(allocator, &backend, cfg, layer_index, layer_layout, io_scratch_bytes);
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

    pub fn initWorkspace(self: *const Runtime, max_seq_len: usize) !workspace_mod.Workspace {
        return try workspace_mod.Workspace.init(
            self.allocator,
            self.cfg,
            max_seq_len,
            optimized_decoder_support.maxIoScratchBytes(self.cfg),
        );
    }

    pub fn forwardTokenId(
        self: *Runtime,
        workspace: *workspace_mod.Workspace,
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
                &self.backend,
                self.thread_count,
                &self.parallel_pool,
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
        workspace: *workspace_mod.Workspace,
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
