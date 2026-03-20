const std = @import("std");

pub const RangeFn = *const fn (ctx: *anyopaque, start_row: usize, end_row: usize) void;

pub const Pool = struct {
    allocator: std.mem.Allocator,
    threads: []std.Thread,
    workers: []Worker,
    shared: *Shared,

    pub fn init(allocator: std.mem.Allocator, thread_count: usize) !Pool {
        if (thread_count <= 1) {
            const shared = try allocator.create(Shared);
            shared.* = .{};
            return .{
                .allocator = allocator,
                .threads = &.{},
                .workers = &.{},
                .shared = shared,
            };
        }

        const worker_count = thread_count - 1;
        const shared = try allocator.create(Shared);
        errdefer allocator.destroy(shared);
        const threads = try allocator.alloc(std.Thread, worker_count);
        errdefer allocator.free(threads);
        const workers = try allocator.alloc(Worker, worker_count);
        errdefer allocator.free(workers);
        shared.* = .{
            .total_workers = thread_count,
        };

        var pool = Pool{
            .allocator = allocator,
            .threads = threads,
            .workers = workers,
            .shared = shared,
        };

        errdefer pool.deinit();
        for (pool.workers, 0..) |*worker, idx| {
            worker.* = .{
                .worker_index = idx + 1,
                .observed_generation = 0,
            };
            worker.shared = shared;
            pool.threads[idx] = try std.Thread.spawn(.{}, Worker.run, .{worker});
        }

        return pool;
    }

    pub fn deinit(self: *Pool) void {
        if (self.threads.len == 0) return;

        self.shared.mutex.lock();
        self.shared.stop = true;
        self.shared.work_ready.broadcast();
        self.shared.mutex.unlock();

        for (self.threads) |thread| thread.join();
        self.allocator.free(self.workers);
        self.allocator.free(self.threads);
        self.allocator.destroy(self.shared);
        self.workers = &.{};
        self.threads = &.{};
    }

    pub fn workerCount(self: *const Pool) usize {
        return self.shared.total_workers;
    }

    pub fn run(self: *Pool, row_count: usize, ctx: *anyopaque, range_fn: RangeFn) void {
        if (self.threads.len == 0 or row_count == 0) {
            range_fn(ctx, 0, row_count);
            return;
        }

        self.shared.mutex.lock();
        self.shared.row_count = row_count;
        self.shared.ctx = ctx;
        self.shared.range_fn = range_fn;
        self.shared.pending_workers = self.threads.len;
        self.shared.generation += 1;
        self.shared.work_ready.broadcast();
        self.shared.mutex.unlock();

        const main_range = computeRange(row_count, self.shared.total_workers, 0);
        if (main_range.start < main_range.end) {
            range_fn(ctx, main_range.start, main_range.end);
        }

        self.shared.mutex.lock();
        defer self.shared.mutex.unlock();
        while (self.shared.pending_workers != 0) {
            self.shared.work_done.wait(&self.shared.mutex);
        }
    }

    const Range = struct {
        start: usize,
        end: usize,
    };

    fn computeRange(row_count: usize, total_workers: usize, worker_index: usize) Range {
        const start = row_count * worker_index / total_workers;
        const end = row_count * (worker_index + 1) / total_workers;
        return .{ .start = start, .end = end };
    }

    const Shared = struct {
        mutex: std.Thread.Mutex = .{},
        work_ready: std.Thread.Condition = .{},
        work_done: std.Thread.Condition = .{},
        stop: bool = false,
        generation: u64 = 0,
        row_count: usize = 0,
        total_workers: usize = 1,
        pending_workers: usize = 0,
        ctx: *anyopaque = undefined,
        range_fn: RangeFn = undefined,
    };

    const Worker = struct {
        shared: *Shared = undefined,
        worker_index: usize,
        observed_generation: u64,

        fn run(self: *Worker) void {
            while (true) {
                self.shared.mutex.lock();
                while (!self.shared.stop and self.shared.generation == self.observed_generation) {
                    self.shared.work_ready.wait(&self.shared.mutex);
                }
                if (self.shared.stop) {
                    self.shared.mutex.unlock();
                    return;
                }

                self.observed_generation = self.shared.generation;
                const row_count = self.shared.row_count;
                const total_workers = self.shared.total_workers;
                const ctx = self.shared.ctx;
                const range_fn = self.shared.range_fn;
                self.shared.mutex.unlock();

                const range = computeRange(row_count, total_workers, self.worker_index);
                if (range.start < range.end) {
                    range_fn(ctx, range.start, range.end);
                }

                self.shared.mutex.lock();
                self.shared.pending_workers -= 1;
                if (self.shared.pending_workers == 0) {
                    self.shared.work_done.signal();
                }
                self.shared.mutex.unlock();
            }
        }
    };
};
