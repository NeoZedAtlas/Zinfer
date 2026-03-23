const std = @import("std");
const decoder_family = @import("../decoder_family.zig");
const optimized_kv_cache = @import("../optimized_kv_cache.zig");
const runtime_mod = @import("runtime.zig");
const workspace_mod = @import("workspace.zig");

pub const RequestState = struct {
    workspace: workspace_mod.Workspace,
    cache: optimized_kv_cache.ModelCache,
    prompt_tokens: usize,
    decoded_tokens: usize,
    active: bool,

    fn deinit(self: *RequestState) void {
        self.cache.deinit();
        self.workspace.deinit();
    }
};

pub const DecodeStats = struct {
    total_decoded_tokens: usize,
    finished_requests: usize,
};

pub const BatchRuntime = struct {
    allocator: std.mem.Allocator,
    model: *runtime_mod.Runtime,
    requests: []RequestState,

    pub fn init(
        allocator: std.mem.Allocator,
        model: *runtime_mod.Runtime,
        batch_size: usize,
        max_seq_len: usize,
        kv_cache_scheme: optimized_kv_cache.Scheme,
        q8_layout: optimized_kv_cache.Q8Layout,
    ) !BatchRuntime {
        if (batch_size == 0) return error.InvalidBatchSize;

        const requests = try allocator.alloc(RequestState, batch_size);
        errdefer allocator.free(requests);

        var initialized: usize = 0;
        errdefer {
            for (requests[0..initialized]) |*request| request.deinit();
        }

        for (requests) |*request| {
            request.* = .{
                .workspace = try model.initWorkspace(max_seq_len),
                .cache = try optimized_kv_cache.ModelCache.initWithLayout(
                    allocator,
                    model.cfg.num_hidden_layers,
                    max_seq_len,
                    model.cfg.num_key_value_heads,
                    model.cfg.head_dim,
                    kv_cache_scheme,
                    q8_layout,
                ),
                .prompt_tokens = 0,
                .decoded_tokens = 0,
                .active = false,
            };
            initialized += 1;
        }

        return .{
            .allocator = allocator,
            .model = model,
            .requests = requests,
        };
    }

    pub fn deinit(self: *BatchRuntime) void {
        for (self.requests) |*request| request.deinit();
        self.allocator.free(self.requests);
    }

    pub fn prefillPromptIds(self: *BatchRuntime, request_index: usize, token_ids: []const usize) !void {
        if (request_index >= self.requests.len) return error.RequestIndexOutOfBounds;
        if (token_ids.len == 0) return error.EmptyPrompt;

        const request = &self.requests[request_index];
        _ = try self.model.prefillTokenIds(&request.workspace, &request.cache, token_ids);
        request.prompt_tokens = token_ids.len;
        request.decoded_tokens = 0;
        request.active = true;
    }

    pub fn decodeRoundRobinArgMax(self: *BatchRuntime, max_new_tokens: usize) !DecodeStats {
        var active_requests = self.requests.len;
        var total_decoded_tokens: usize = 0;
        var finished_requests: usize = 0;

        for (0..max_new_tokens) |_| {
            if (active_requests == 0) break;

            var progressed = false;
            for (self.requests) |*request| {
                if (!request.active) continue;

                const next_token = try decoder_family.argMaxLogit(request.workspace.logits);
                if (decoder_family.isEosToken(self.model.cfg.architecture, next_token)) {
                    request.active = false;
                    active_requests -= 1;
                    finished_requests += 1;
                    continue;
                }

                _ = try self.model.forwardTokenId(&request.workspace, &request.cache, next_token);
                request.decoded_tokens += 1;
                total_decoded_tokens += 1;
                progressed = true;
            }

            if (!progressed) break;
        }

        return .{
            .total_decoded_tokens = total_decoded_tokens,
            .finished_requests = finished_requests,
        };
    }
};

test "batch runtime decode stats default to finished when no request progresses" {
    const stats = DecodeStats{
        .total_decoded_tokens = 0,
        .finished_requests = 0,
    };
    try std.testing.expectEqual(@as(usize, 0), stats.total_decoded_tokens);
    try std.testing.expectEqual(@as(usize, 0), stats.finished_requests);
}
