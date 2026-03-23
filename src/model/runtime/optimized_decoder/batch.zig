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
    scheduler_workers: usize,
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
        request.cache.reset();
        _ = try self.model.prefillTokenIds(&request.workspace, &request.cache, token_ids);
        request.prompt_tokens = token_ids.len;
        request.decoded_tokens = 0;
        request.active = true;
    }

    pub fn processQueuedPromptIdsArgMax(
        self: *BatchRuntime,
        prompt_ids: []const usize,
        total_requests: usize,
        max_new_tokens: usize,
    ) !DecodeStats {
        if (prompt_ids.len == 0) return error.EmptyPrompt;
        if (total_requests == 0) return error.InvalidBatchSize;

        var next_request_to_submit: usize = 0;
        const initial_requests = @min(total_requests, self.requests.len);
        for (0..initial_requests) |request_index| {
            try self.prefillPromptIds(request_index, prompt_ids);
            next_request_to_submit += 1;
        }

        var active_requests = initial_requests;
        var total_decoded_tokens: usize = 0;
        var finished_requests: usize = 0;
        const scheduler_workers = self.schedulerWorkerCount();

        while (active_requests != 0) {
            var progressed = false;
            if (scheduler_workers <= 1) {
                for (self.requests) |*request| {
                    if (!request.active) continue;

                    const next_token = try decoder_family.argMaxLogit(request.workspace.logits);
                    if (decoder_family.isEosToken(self.model.cfg.architecture, next_token)) {
                        finishRequest(request);
                        active_requests -= 1;
                        finished_requests += 1;
                        continue;
                    }

                    _ = try self.model.forwardTokenId(&request.workspace, &request.cache, next_token);
                    request.decoded_tokens += 1;
                    total_decoded_tokens += 1;
                    progressed = true;
                    if (request.decoded_tokens >= max_new_tokens) {
                        finishRequest(request);
                        active_requests -= 1;
                        finished_requests += 1;
                    }
                }
            } else {
                var launches = std.ArrayListUnmanaged(Launch).empty;
                defer launches.deinit(self.allocator);

                for (self.requests) |*request| {
                    if (!request.active) continue;

                    const next_token = try decoder_family.argMaxLogit(request.workspace.logits);
                    if (decoder_family.isEosToken(self.model.cfg.architecture, next_token)) {
                        finishRequest(request);
                        active_requests -= 1;
                        finished_requests += 1;
                        continue;
                    }

                    try launches.append(self.allocator, .{
                        .request = request,
                        .token_id = next_token,
                    });
                }

                if (launches.items.len != 0) {
                    const worker_count = @min(scheduler_workers, launches.items.len);
                    const threads = try self.allocator.alloc(std.Thread, worker_count - 1);
                    defer self.allocator.free(threads);
                    const contexts = try self.allocator.alloc(WorkerContext, worker_count - 1);
                    defer self.allocator.free(contexts);
                    const chunk = std.math.divCeil(usize, launches.items.len, worker_count) catch unreachable;

                    var start_index: usize = 0;
                    for (0..worker_count - 1) |idx| {
                        const end_index = @min(launches.items.len, start_index + chunk);
                        contexts[idx] = .{
                            .runtime = self.model,
                            .launches = launches.items[start_index..end_index],
                        };
                        threads[idx] = try std.Thread.spawn(.{}, WorkerContext.run, .{&contexts[idx]});
                        start_index = end_index;
                    }

                    runLaunchRange(self.model, launches.items[start_index..]);
                    for (threads) |thread| thread.join();

                    for (launches.items) |*launch| {
                        _ = try launch.result;
                        launch.request.decoded_tokens += 1;
                        total_decoded_tokens += 1;
                        progressed = true;
                        if (launch.request.decoded_tokens >= max_new_tokens) {
                            finishRequest(launch.request);
                            active_requests -= 1;
                            finished_requests += 1;
                        }
                    }
                }
            }

            for (self.requests, 0..) |*request, request_index| {
                if (request.active or next_request_to_submit >= total_requests) continue;
                try self.prefillPromptIds(request_index, prompt_ids);
                next_request_to_submit += 1;
                active_requests += 1;
            }

            if (!progressed and next_request_to_submit >= total_requests) break;
        }

        return .{
            .total_decoded_tokens = total_decoded_tokens,
            .finished_requests = finished_requests,
            .scheduler_workers = scheduler_workers,
        };
    }

    pub fn decodeRoundRobinArgMax(self: *BatchRuntime, max_new_tokens: usize) !DecodeStats {
        var active_requests = self.requests.len;
        var total_decoded_tokens: usize = 0;
        var finished_requests: usize = 0;
        const scheduler_workers = self.schedulerWorkerCount();

        for (0..max_new_tokens) |_| {
            if (active_requests == 0) break;

            var progressed = false;
            if (scheduler_workers <= 1) {
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
            } else {
                var launches = std.ArrayListUnmanaged(Launch).empty;
                defer launches.deinit(self.allocator);

                for (self.requests) |*request| {
                    if (!request.active) continue;

                    const next_token = try decoder_family.argMaxLogit(request.workspace.logits);
                    if (decoder_family.isEosToken(self.model.cfg.architecture, next_token)) {
                        request.active = false;
                        active_requests -= 1;
                        finished_requests += 1;
                        continue;
                    }

                    try launches.append(self.allocator, .{
                        .request = request,
                        .token_id = next_token,
                    });
                }

                if (launches.items.len != 0) {
                    const worker_count = @min(scheduler_workers, launches.items.len);
                    const threads = try self.allocator.alloc(std.Thread, worker_count - 1);
                    defer self.allocator.free(threads);
                    const contexts = try self.allocator.alloc(WorkerContext, worker_count - 1);
                    defer self.allocator.free(contexts);
                    const chunk = std.math.divCeil(usize, launches.items.len, worker_count) catch unreachable;

                    var start_index: usize = 0;
                    for (0..worker_count - 1) |idx| {
                        const end_index = @min(launches.items.len, start_index + chunk);
                        contexts[idx] = .{
                            .runtime = self.model,
                            .launches = launches.items[start_index..end_index],
                        };
                        threads[idx] = try std.Thread.spawn(.{}, WorkerContext.run, .{&contexts[idx]});
                        start_index = end_index;
                    }

                    runLaunchRange(self.model, launches.items[start_index..]);
                    for (threads) |thread| thread.join();

                    for (launches.items) |*launch| {
                        _ = try launch.result;
                        launch.request.decoded_tokens += 1;
                        total_decoded_tokens += 1;
                        progressed = true;
                    }
                }
            }

            if (!progressed) break;
        }

        return .{
            .total_decoded_tokens = total_decoded_tokens,
            .finished_requests = finished_requests,
            .scheduler_workers = scheduler_workers,
        };
    }

    fn schedulerWorkerCount(self: *const BatchRuntime) usize {
        if (self.model.thread_count != 1) return 1;
        return @min(self.requests.len, std.Thread.getCpuCount() catch 1);
    }
};

fn finishRequest(request: *RequestState) void {
    request.active = false;
    request.prompt_tokens = 0;
    request.decoded_tokens = 0;
}

const Launch = struct {
    request: *RequestState,
    token_id: usize,
    result: anyerror![]f32 = &.{},
};

const WorkerContext = struct {
    runtime: *runtime_mod.Runtime,
    launches: []Launch,

    fn run(self: *WorkerContext) void {
        runLaunchRange(self.runtime, self.launches);
    }
};

fn runLaunchRange(runtime: *runtime_mod.Runtime, launches: []Launch) void {
    for (launches) |*launch| {
        launch.result = runtime.forwardTokenId(&launch.request.workspace, &launch.request.cache, launch.token_id);
    }
}

test "batch runtime decode stats default to finished when no request progresses" {
    const stats = DecodeStats{
        .total_decoded_tokens = 0,
        .finished_requests = 0,
        .scheduler_workers = 0,
    };
    try std.testing.expectEqual(@as(usize, 0), stats.total_decoded_tokens);
    try std.testing.expectEqual(@as(usize, 0), stats.finished_requests);
    try std.testing.expectEqual(@as(usize, 0), stats.scheduler_workers);
}
