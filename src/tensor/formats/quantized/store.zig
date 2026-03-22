const std = @import("std");
const file_impl = @import("file.zig");
const kernels = @import("kernels.zig");
const types = @import("types.zig");
const parallel_rows = @import("../../parallel/parallel_rows.zig");
const tensor_store = @import("../../storage/store.zig");

pub const Store = struct {
    allocator: std.mem.Allocator,
    bytes: []u8,
    parsed: types.ParsedFile,

    pub fn open(allocator: std.mem.Allocator, path: []const u8) !Store {
        const bytes = if (std.fs.path.isAbsolute(path)) blk: {
            const file = try std.fs.openFileAbsolute(path, .{});
            defer file.close();
            break :blk try file.readToEndAlloc(allocator, std.math.maxInt(usize));
        } else try std.fs.cwd().readFileAlloc(allocator, path, std.math.maxInt(usize));
        errdefer allocator.free(bytes);

        var parsed = try file_impl.parseFromBytes(allocator, bytes);
        errdefer parsed.deinit();

        return .{
            .allocator = allocator,
            .bytes = bytes,
            .parsed = parsed,
        };
    }

    pub fn deinit(self: *Store) void {
        self.parsed.deinit();
        self.allocator.free(self.bytes);
    }

    pub fn getTensor(self: *const Store, name: []const u8) ?types.TensorInfo {
        return self.parsed.getTensor(name);
    }

    pub fn readElementsAsF32Into(
        self: *const Store,
        name: []const u8,
        start_element: u64,
        output: []f32,
    ) !void {
        const tensor = self.getTensor(name) orelse return error.TensorNotFound;
        try self.readTensorElementsAsF32Into(tensor, start_element, output);
    }

    pub fn readTensorElementsAsF32Into(
        self: *const Store,
        tensor: types.TensorInfo,
        start_element: u64,
        output: []f32,
    ) !void {
        if (tensor.encoding != .f32) return error.UnsupportedEncoding;
        const total_elements = file_impl.elementCount(tensor.shape);
        if (start_element + output.len > total_elements) return error.ElementRangeOutOfBounds;
        const start = std.math.cast(usize, tensor.absolute_offset + start_element * 4) orelse return error.BufferTooLarge;
        for (output, 0..) |*value, idx| {
            const offset = start + idx * 4;
            const raw = std.mem.readInt(u32, self.bytes[offset .. offset + 4][0..4], .little);
            value.* = @bitCast(raw);
        }
    }

    pub fn readRowAsF32Into(
        self: *const Store,
        name: []const u8,
        row_index: usize,
        output: []f32,
    ) !void {
        const tensor = self.getTensor(name) orelse return error.TensorNotFound;
        try self.readTensorRowAsF32Into(tensor, row_index, output);
    }

    pub fn readTensorRowAsF32Into(
        self: *const Store,
        tensor: types.TensorInfo,
        row_index: usize,
        output: []f32,
    ) !void {
        if (tensor.rank() != 2) return error.InvalidTensorRank;
        const rows = std.math.cast(usize, tensor.shape[0]) orelse return error.DimensionTooLarge;
        const cols = std.math.cast(usize, tensor.shape[1]) orelse return error.DimensionTooLarge;
        if (row_index >= rows) return error.RowIndexOutOfBounds;
        if (output.len != cols) return error.SizeMismatch;

        const row_offset = tensor.absolute_offset + @as(u64, row_index) * tensor.row_bytes;
        switch (tensor.encoding) {
            .f32 => {
                const start = std.math.cast(usize, row_offset) orelse return error.BufferTooLarge;
                for (output, 0..) |*value, idx| {
                    const offset = start + idx * 4;
                    const raw = std.mem.readInt(u32, self.bytes[offset .. offset + 4][0..4], .little);
                    value.* = @bitCast(raw);
                }
            },
            .q6_0 => kernels.decodeQ6Row(self.bytes, row_offset, output),
            .q8_0 => kernels.decodeQ8Row(self.bytes, row_offset, output),
            .q4_0 => kernels.decodeQ4Row(self.bytes, row_offset, output),
        }
    }

    pub fn matmulVecByName(
        self: *const Store,
        output: []f32,
        name: []const u8,
        input: []const f32,
        thread_count: usize,
        pool: ?*parallel_rows.Pool,
    ) !void {
        const tensor = self.getTensor(name) orelse return error.TensorNotFound;
        try self.matmulVec(output, tensor, input, thread_count, pool);
    }

    pub fn matmulVec(
        self: *const Store,
        output: []f32,
        tensor: types.TensorInfo,
        input: []const f32,
        thread_count: usize,
        pool: ?*parallel_rows.Pool,
    ) !void {
        if (tensor.rank() != 2) return error.InvalidTensorRank;

        const rows = std.math.cast(usize, tensor.shape[0]) orelse return error.DimensionTooLarge;
        const cols = std.math.cast(usize, tensor.shape[1]) orelse return error.DimensionTooLarge;
        if (output.len != rows or input.len != cols) return error.SizeMismatch;

        if (shouldParallelize(rows, cols, thread_count, pool != null)) {
            if (pool) |available_pool| {
                self.matmulWithPool(output, tensor, input, available_pool);
                return;
            }
        }
        if (shouldParallelize(rows, cols, thread_count, true)) {
            try self.matmulThreaded(output, tensor, input, thread_count);
            return;
        }
        self.matmulRange(output, tensor, input, 0, rows);
    }

    fn matmulThreaded(
        self: *const Store,
        output: []f32,
        tensor: types.TensorInfo,
        input: []const f32,
        thread_count: usize,
    ) !void {
        const actual_threads = @min(thread_count, output.len);
        if (actual_threads <= 1) {
            self.matmulRange(output, tensor, input, 0, output.len);
            return;
        }

        const rows_per_job = std.math.divCeil(usize, output.len, actual_threads) catch return error.InvalidThreadCount;

        const Worker = struct {
            store: *const Store,
            output: []f32,
            tensor: types.TensorInfo,
            input: []const f32,
            start_row: usize,
            end_row: usize,

            fn run(ctx: *@This()) void {
                ctx.store.matmulRange(ctx.output, ctx.tensor, ctx.input, ctx.start_row, ctx.end_row);
            }
        };

        const worker_count = actual_threads - 1;
        const threads = try self.allocator.alloc(std.Thread, worker_count);
        defer self.allocator.free(threads);
        const contexts = try self.allocator.alloc(Worker, worker_count);
        defer self.allocator.free(contexts);

        var start_row: usize = 0;
        for (0..worker_count) |idx| {
            const end_row = @min(output.len, start_row + rows_per_job);
            contexts[idx] = .{
                .store = self,
                .output = output,
                .tensor = tensor,
                .input = input,
                .start_row = start_row,
                .end_row = end_row,
            };
            threads[idx] = try std.Thread.spawn(.{}, Worker.run, .{&contexts[idx]});
            start_row = end_row;
        }

        self.matmulRange(output, tensor, input, start_row, output.len);
        for (threads) |thread| thread.join();
    }

    fn matmulRange(
        self: *const Store,
        output: []f32,
        tensor: types.TensorInfo,
        input: []const f32,
        start_row: usize,
        end_row: usize,
    ) void {
        var row_offset = tensor.absolute_offset + @as(u64, start_row) * tensor.row_bytes;
        switch (tensor.encoding) {
            .f32 => {
                for (start_row..end_row) |row_idx| {
                    output[row_idx] = tensor_store.dotF32Row(self.bytes[@intCast(row_offset)..][0 .. input.len * 4], input);
                    row_offset += tensor.row_bytes;
                }
            },
            .q6_0 => {
                for (start_row..end_row) |row_idx| {
                    output[row_idx] = kernels.dotQ6Row(self.bytes, row_offset, input);
                    row_offset += tensor.row_bytes;
                }
            },
            .q8_0 => {
                for (start_row..end_row) |row_idx| {
                    output[row_idx] = kernels.dotQ8Row(self.bytes, row_offset, input);
                    row_offset += tensor.row_bytes;
                }
            },
            .q4_0 => {
                for (start_row..end_row) |row_idx| {
                    output[row_idx] = kernels.dotQ4Row(self.bytes, row_offset, input);
                    row_offset += tensor.row_bytes;
                }
            },
        }
    }

    fn matmulWithPool(
        self: *const Store,
        output: []f32,
        tensor: types.TensorInfo,
        input: []const f32,
        pool: *parallel_rows.Pool,
    ) void {
        const Context = struct {
            store: *const Store,
            output: []f32,
            tensor: types.TensorInfo,
            input: []const f32,

            fn runRange(ctx_ptr: *anyopaque, start_row: usize, end_row: usize) void {
                const ctx: *@This() = @ptrCast(@alignCast(ctx_ptr));
                ctx.store.matmulRange(ctx.output, ctx.tensor, ctx.input, start_row, end_row);
            }
        };

        var context = Context{
            .store = self,
            .output = output,
            .tensor = tensor,
            .input = input,
        };
        pool.run(output.len, &context, Context.runRange);
    }
};

fn shouldParallelize(rows: usize, cols: usize, thread_count: usize, has_parallel_backend: bool) bool {
    if (!has_parallel_backend or thread_count <= 1) return false;
    const work = std.math.mul(u64, rows, cols) catch return true;
    return work >= 1_000_000;
}

test "quantized store matmulVecByName matches q8 rows for hot width" {
    const testing = std.testing;
    const rows: usize = 3;
    const cols = tensor_store.handwritten_hidden_width;
    const row_bytes = 4 + cols;
    const payload_len = rows * row_bytes;
    const header = try std.fmt.allocPrint(
        testing.allocator,
        "{{\"weight\":{{\"encoding\":\"Q8_0\",\"shape\":[{d},{d}],\"row_bytes\":{d},\"data_offsets\":[0,{d}]}}}}",
        .{ rows, cols, row_bytes, payload_len },
    );
    defer testing.allocator.free(header);

    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    const file = try tmp.dir.createFile("matmul_q8.zinfer", .{});
    defer file.close();

    var length_prefix: [8]u8 = undefined;
    std.mem.writeInt(u64, &length_prefix, header.len, .little);
    try file.writeAll(&length_prefix);
    try file.writeAll(header);

    const input = try testing.allocator.alloc(f32, cols);
    defer testing.allocator.free(input);
    const row_values = try testing.allocator.alloc(f32, cols);
    defer testing.allocator.free(row_values);
    const encoded_row = try testing.allocator.alloc(u8, row_bytes);
    defer testing.allocator.free(encoded_row);

    for (input, 0..) |*value, idx| {
        value.* = @as(f32, @floatFromInt(@as(i32, @intCast((idx * 11 + 7) % 37)) - 18)) / 9.0;
    }

    for (0..rows) |row_idx| {
        for (row_values, 0..) |*value, col_idx| {
            const bucket = @as(i32, @intCast((row_idx * 23 + col_idx * 7 + 5) % 41)) - 20;
            value.* = @as(f32, @floatFromInt(bucket)) / 8.0;
        }
        kernels.encodeQ8Row(encoded_row, row_values);
        try file.writeAll(encoded_row);
    }

    var path_buffer: [std.fs.max_path_bytes]u8 = undefined;
    const path = try tmp.dir.realpath("matmul_q8.zinfer", &path_buffer);

    var store = try Store.open(testing.allocator, path);
    defer store.deinit();

    const tensor = store.getTensor("weight") orelse return error.TensorNotFound;
    var output = [_]f32{ 0.0, 0.0, 0.0 };
    try store.matmulVecByName(&output, "weight", input, 1, null);

    for (0..rows) |row_idx| {
        const row_offset = tensor.absolute_offset + @as(u64, row_idx) * tensor.row_bytes;
        const expected = kernels.dotQ8Row(store.bytes, row_offset, input);
        try testing.expectApproxEqAbs(expected, output[row_idx], 1e-4);
    }
}

test "quantized store matmulVecByName matches q6 rows for hot width" {
    const testing = std.testing;
    const rows: usize = 3;
    const cols = tensor_store.handwritten_hidden_width;
    const row_bytes = 4 + (try std.math.divCeil(usize, cols * 6, 8));
    const payload_len = rows * row_bytes;
    const header = try std.fmt.allocPrint(
        testing.allocator,
        "{{\"weight\":{{\"encoding\":\"Q6_0\",\"shape\":[{d},{d}],\"row_bytes\":{d},\"data_offsets\":[0,{d}]}}}}",
        .{ rows, cols, row_bytes, payload_len },
    );
    defer testing.allocator.free(header);

    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    const file = try tmp.dir.createFile("matmul_q6.zinfer", .{});
    defer file.close();

    var length_prefix: [8]u8 = undefined;
    std.mem.writeInt(u64, &length_prefix, header.len, .little);
    try file.writeAll(&length_prefix);
    try file.writeAll(header);

    const input = try testing.allocator.alloc(f32, cols);
    defer testing.allocator.free(input);
    const row_values = try testing.allocator.alloc(f32, cols);
    defer testing.allocator.free(row_values);
    const encoded_row = try testing.allocator.alloc(u8, row_bytes);
    defer testing.allocator.free(encoded_row);

    for (input, 0..) |*value, idx| {
        value.* = @as(f32, @floatFromInt(@as(i32, @intCast((idx * 13 + 9) % 37)) - 18)) / 9.0;
    }

    for (0..rows) |row_idx| {
        for (row_values, 0..) |*value, col_idx| {
            const bucket = @as(i32, @intCast((row_idx * 19 + col_idx * 5 + 7) % 41)) - 20;
            value.* = @as(f32, @floatFromInt(bucket)) / 8.0;
        }
        kernels.encodeQ6Row(encoded_row, row_values);
        try file.writeAll(encoded_row);
    }

    var path_buffer: [std.fs.max_path_bytes]u8 = undefined;
    const path = try tmp.dir.realpath("matmul_q6.zinfer", &path_buffer);

    var store = try Store.open(testing.allocator, path);
    defer store.deinit();

    const tensor = store.getTensor("weight") orelse return error.TensorNotFound;
    var output = [_]f32{ 0.0, 0.0, 0.0 };
    try store.matmulVecByName(&output, "weight", input, 1, null);

    for (0..rows) |row_idx| {
        const row_offset = tensor.absolute_offset + @as(u64, row_idx) * tensor.row_bytes;
        const expected = kernels.dotQ6Row(store.bytes, row_offset, input);
        try testing.expectApproxEqAbs(expected, output[row_idx], 1e-4);
    }
}
