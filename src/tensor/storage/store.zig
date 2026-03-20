const std = @import("std");
const bfloat16 = @import("../formats/bfloat16.zig");
const parallel_rows = @import("../parallel/parallel_rows.zig");
const safetensors = @import("../../format/safetensors.zig");

pub const TensorStore = struct {
    allocator: std.mem.Allocator,
    bytes: []u8,
    parsed: safetensors.ParsedFile,

    pub fn open(allocator: std.mem.Allocator, weights_path: []const u8) !TensorStore {
        const bytes = if (std.fs.path.isAbsolute(weights_path)) blk: {
            const file = try std.fs.openFileAbsolute(weights_path, .{});
            defer file.close();
            break :blk try file.readToEndAlloc(allocator, std.math.maxInt(usize));
        } else try std.fs.cwd().readFileAlloc(allocator, weights_path, std.math.maxInt(usize));
        errdefer allocator.free(bytes);

        var parsed = try safetensors.parseFromBytes(allocator, bytes);
        errdefer parsed.deinit();

        return .{
            .allocator = allocator,
            .bytes = bytes,
            .parsed = parsed,
        };
    }

    pub fn deinit(self: *TensorStore) void {
        self.parsed.deinit();
        self.allocator.free(self.bytes);
    }

    pub fn getTensor(self: *const TensorStore, name: []const u8) ?safetensors.TensorInfo {
        return self.parsed.getTensor(name);
    }

    pub fn readElementsAsF32Alloc(
        self: *const TensorStore,
        name: []const u8,
        start_element: u64,
        count: usize,
    ) ![]f32 {
        const tensor = self.getTensor(name) orelse return error.TensorNotFound;
        const total_elements = try tensor.elementCount();
        if (start_element > total_elements) return error.ElementRangeOutOfBounds;

        const available = total_elements - start_element;
        const requested = @min(@as(u64, count), available);
        const requested_usize = std.math.cast(usize, requested) orelse return error.ElementCountTooLarge;

        const output = try self.allocator.alloc(f32, requested_usize);
        errdefer self.allocator.free(output);
        try self.readElementsAsF32Into(name, start_element, output, &.{});
        return output;
    }

    pub fn readElementsAsF32Into(
        self: *const TensorStore,
        name: []const u8,
        start_element: u64,
        output: []f32,
        scratch: []u8,
    ) !void {
        _ = scratch;
        const tensor = self.getTensor(name) orelse return error.TensorNotFound;
        try self.readTensorElementsAsF32Into(tensor, start_element, output);
    }

    pub fn readTensorElementsAsF32Into(
        self: *const TensorStore,
        tensor: safetensors.TensorInfo,
        start_element: u64,
        output: []f32,
    ) !void {
        const total_elements = try tensor.elementCount();
        if (start_element > total_elements) return error.ElementRangeOutOfBounds;
        const requested = @as(u64, output.len);
        if (requested > total_elements - start_element) return error.ElementRangeOutOfBounds;

        switch (tensor.dtype) {
            .bf16 => try self.readBf16AsF32Into(tensor, start_element, output),
            .f32 => try self.readF32Into(tensor, start_element, output),
            else => return error.UnsupportedTensorDType,
        }
    }

    pub fn readRowAsF32Alloc(
        self: *const TensorStore,
        name: []const u8,
        row_index: usize,
    ) ![]f32 {
        const tensor = self.getTensor(name) orelse return error.TensorNotFound;
        if (tensor.rank() != 2) return error.InvalidTensorRank;

        const rows = std.math.cast(usize, tensor.shape[0]) orelse return error.DimensionTooLarge;
        const cols = std.math.cast(usize, tensor.shape[1]) orelse return error.DimensionTooLarge;
        if (row_index >= rows) return error.RowIndexOutOfBounds;

        const output = try self.allocator.alloc(f32, cols);
        errdefer self.allocator.free(output);
        try self.readRowAsF32Into(name, row_index, output, &.{});
        return output;
    }

    pub fn readRowAsF32Into(
        self: *const TensorStore,
        name: []const u8,
        row_index: usize,
        output: []f32,
        scratch: []u8,
    ) !void {
        _ = scratch;
        const tensor = self.getTensor(name) orelse return error.TensorNotFound;
        try self.readTensorRowAsF32Into(tensor, row_index, output);
    }

    pub fn readTensorRowAsF32Into(
        self: *const TensorStore,
        tensor: safetensors.TensorInfo,
        row_index: usize,
        output: []f32,
    ) !void {
        if (tensor.rank() != 2) return error.InvalidTensorRank;

        const rows = std.math.cast(usize, tensor.shape[0]) orelse return error.DimensionTooLarge;
        const cols = std.math.cast(usize, tensor.shape[1]) orelse return error.DimensionTooLarge;
        if (row_index >= rows) return error.RowIndexOutOfBounds;
        if (output.len != cols) return error.SizeMismatch;

        const start = try std.math.mul(u64, row_index, cols);
        try self.readTensorElementsAsF32Into(tensor, start, output);
    }

    pub fn matmulVecByName(
        self: *const TensorStore,
        output: []f32,
        name: []const u8,
        input: []const f32,
    ) !void {
        try self.matmulVecByNameThreaded(output, name, input, 1, null, &.{});
    }

    pub fn matmulVecByNameWithScratch(
        self: *const TensorStore,
        output: []f32,
        name: []const u8,
        input: []const f32,
        scratch: []u8,
    ) !void {
        try self.matmulVecByNameThreaded(output, name, input, 1, null, scratch);
    }

    pub fn matmulVecByNameThreaded(
        self: *const TensorStore,
        output: []f32,
        name: []const u8,
        input: []const f32,
        thread_count: usize,
        pool: ?*parallel_rows.Pool,
        scratch: []u8,
    ) !void {
        _ = scratch;
        const tensor = self.getTensor(name) orelse return error.TensorNotFound;
        try self.matmulVecThreaded(output, tensor, input, thread_count, pool);
    }

    pub fn matmulVecThreaded(
        self: *const TensorStore,
        output: []f32,
        tensor: safetensors.TensorInfo,
        input: []const f32,
        thread_count: usize,
        pool: ?*parallel_rows.Pool,
    ) !void {
        if (tensor.rank() != 2) return error.InvalidTensorRank;

        const rows = std.math.cast(usize, tensor.shape[0]) orelse return error.DimensionTooLarge;
        const cols = std.math.cast(usize, tensor.shape[1]) orelse return error.DimensionTooLarge;
        if (output.len != rows or input.len != cols) return error.SizeMismatch;
        if (tensor.dtype != .bf16 and tensor.dtype != .f32) return error.UnsupportedTensorDType;

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

    fn readBf16AsF32Into(
        self: *const TensorStore,
        tensor: safetensors.TensorInfo,
        start_element: u64,
        output: []f32,
    ) !void {
        const byte_offset = try std.math.mul(u64, start_element, 2);
        const read_offset = try std.math.add(u64, tensor.absolute_offset, byte_offset);
        const byte_count = try std.math.mul(u64, output.len, 2);
        const slice = try self.byteRange(read_offset, byte_count);

        for (output, 0..) |*value, idx| {
            const start = idx * 2;
            const bits = std.mem.readInt(u16, slice[start .. start + 2][0..2], .little);
            value.* = bfloat16.toF32(bits);
        }
    }

    fn readF32Into(
        self: *const TensorStore,
        tensor: safetensors.TensorInfo,
        start_element: u64,
        output: []f32,
    ) !void {
        const byte_offset = try std.math.mul(u64, start_element, 4);
        const read_offset = try std.math.add(u64, tensor.absolute_offset, byte_offset);
        const byte_count = try std.math.mul(u64, output.len, 4);
        const slice = try self.byteRange(read_offset, byte_count);

        for (output, 0..) |*value, idx| {
            const start = idx * 4;
            const raw = std.mem.readInt(u32, slice[start .. start + 4][0..4], .little);
            value.* = @bitCast(raw);
        }
    }

    fn matmulThreaded(
        self: *const TensorStore,
        output: []f32,
        tensor: safetensors.TensorInfo,
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
            store: *const TensorStore,
            output: []f32,
            tensor: safetensors.TensorInfo,
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
        self: *const TensorStore,
        output: []f32,
        tensor: safetensors.TensorInfo,
        input: []const f32,
        start_row: usize,
        end_row: usize,
    ) void {
        const cols = input.len;
        const row_bytes = std.math.cast(usize, @as(u64, cols) * tensor.dtype.byteSize()) orelse unreachable;
        var row_offset = std.math.add(
            u64,
            tensor.absolute_offset,
            std.math.mul(u64, start_row, row_bytes) catch unreachable,
        ) catch unreachable;

        switch (tensor.dtype) {
            .bf16 => {
                for (start_row..end_row) |row_idx| {
                    const row = self.byteRange(row_offset, row_bytes) catch unreachable;
                    output[row_idx] = dotBf16Row(row, input);
                    row_offset += row_bytes;
                }
            },
            .f32 => {
                for (start_row..end_row) |row_idx| {
                    const row = self.byteRange(row_offset, row_bytes) catch unreachable;
                    output[row_idx] = dotF32Row(row, input);
                    row_offset += row_bytes;
                }
            },
            else => unreachable,
        }
    }

    fn matmulWithPool(
        self: *const TensorStore,
        output: []f32,
        tensor: safetensors.TensorInfo,
        input: []const f32,
        pool: *parallel_rows.Pool,
    ) void {
        const Context = struct {
            store: *const TensorStore,
            output: []f32,
            tensor: safetensors.TensorInfo,
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

    fn byteRange(self: *const TensorStore, absolute_offset: u64, byte_count: u64) ![]const u8 {
        const start = std.math.cast(usize, absolute_offset) orelse return error.BufferTooLarge;
        const len = std.math.cast(usize, byte_count) orelse return error.BufferTooLarge;
        const end = std.math.add(usize, start, len) catch return error.BufferTooLarge;
        if (end > self.bytes.len) return error.UnexpectedEndOfFile;
        return self.bytes[start..end];
    }
};

fn shouldParallelize(rows: usize, cols: usize, thread_count: usize, has_parallel_backend: bool) bool {
    if (!has_parallel_backend or thread_count <= 1) return false;
    const work = std.math.mul(u64, rows, cols) catch return true;
    return work >= 1_000_000;
}

pub const handwritten_hidden_width: usize = 1024;
pub const handwritten_intermediate_width: usize = 3072;

pub fn dotF32Row(row: []const u8, input: []const f32) f32 {
    return switch (input.len) {
        handwritten_hidden_width => dotF32RowFixed(handwritten_hidden_width, row, input),
        handwritten_intermediate_width => dotF32RowFixed(handwritten_intermediate_width, row, input),
        else => dotF32RowGeneric(row, input),
    };
}

fn dotF32RowGeneric(row: []const u8, input: []const f32) f32 {
    var sum: f32 = 0.0;
    var index: usize = 0;
    while (index + 16 <= input.len) : (index += 16) {
        var lhs_arr: [16]f32 = undefined;
        inline for (0..16) |lane| {
            const offset = (index + lane) * 4;
            const raw = std.mem.readInt(u32, row[offset .. offset + 4][0..4], .little);
            lhs_arr[lane] = @bitCast(raw);
        }
        const lhs: @Vector(16, f32) = lhs_arr;
        const rhs: @Vector(16, f32) = input[index..][0..16].*;
        sum += @reduce(.Add, lhs * rhs);
    }
    while (index < input.len) : (index += 1) {
        const offset = index * 4;
        const raw = std.mem.readInt(u32, row[offset .. offset + 4][0..4], .little);
        sum += @as(f32, @bitCast(raw)) * input[index];
    }
    return sum;
}

fn dotF32RowFixed(comptime cols: usize, row: []const u8, input: []const f32) f32 {
    std.debug.assert(input.len == cols);
    std.debug.assert(row.len == cols * 4);

    var acc0: @Vector(16, f32) = @splat(0.0);
    var acc1: @Vector(16, f32) = @splat(0.0);
    var index: usize = 0;
    while (index < cols) : (index += 32) {
        const lhs0 = loadF32Vector16(row, index * 4);
        const lhs1 = loadF32Vector16(row, (index + 16) * 4);
        const rhs0: @Vector(16, f32) = input[index..][0..16].*;
        const rhs1: @Vector(16, f32) = input[index + 16 ..][0..16].*;
        acc0 += lhs0 * rhs0;
        acc1 += lhs1 * rhs1;
    }
    return @reduce(.Add, acc0 + acc1);
}

pub fn dotBf16Row(row: []const u8, input: []const f32) f32 {
    return switch (input.len) {
        handwritten_hidden_width => dotBf16RowFixed(handwritten_hidden_width, row, input),
        handwritten_intermediate_width => dotBf16RowFixed(handwritten_intermediate_width, row, input),
        else => dotBf16RowGeneric(row, input),
    };
}

fn dotBf16RowGeneric(row: []const u8, input: []const f32) f32 {
    var sum: f32 = 0.0;
    var index: usize = 0;
    while (index + 16 <= input.len) : (index += 16) {
        var lhs_arr: [16]f32 = undefined;
        inline for (0..16) |lane| {
            const offset = (index + lane) * 2;
            const bits = std.mem.readInt(u16, row[offset .. offset + 2][0..2], .little);
            lhs_arr[lane] = bfloat16.toF32(bits);
        }
        const lhs: @Vector(16, f32) = lhs_arr;
        const rhs: @Vector(16, f32) = input[index..][0..16].*;
        sum += @reduce(.Add, lhs * rhs);
    }
    while (index < input.len) : (index += 1) {
        const offset = index * 2;
        const bits = std.mem.readInt(u16, row[offset .. offset + 2][0..2], .little);
        sum += bfloat16.toF32(bits) * input[index];
    }
    return sum;
}

fn dotBf16RowFixed(comptime cols: usize, row: []const u8, input: []const f32) f32 {
    std.debug.assert(input.len == cols);
    std.debug.assert(row.len == cols * 2);

    var acc0: @Vector(16, f32) = @splat(0.0);
    var acc1: @Vector(16, f32) = @splat(0.0);
    var index: usize = 0;
    while (index < cols) : (index += 32) {
        const lhs0 = loadBf16Vector16(row, index * 2);
        const lhs1 = loadBf16Vector16(row, (index + 16) * 2);
        const rhs0: @Vector(16, f32) = input[index..][0..16].*;
        const rhs1: @Vector(16, f32) = input[index + 16 ..][0..16].*;
        acc0 += lhs0 * rhs0;
        acc1 += lhs1 * rhs1;
    }
    return @reduce(.Add, acc0 + acc1);
}

fn loadF32Vector16(bytes: []const u8, byte_offset: usize) @Vector(16, f32) {
    var values: [16]f32 = undefined;
    inline for (0..16) |lane| {
        const offset = byte_offset + lane * 4;
        const raw = std.mem.readInt(u32, bytes[offset .. offset + 4][0..4], .little);
        values[lane] = @bitCast(raw);
    }
    return values;
}

fn loadBf16Vector16(bytes: []const u8, byte_offset: usize) @Vector(16, f32) {
    var values: [16]f32 = undefined;
    inline for (0..16) |lane| {
        const offset = byte_offset + lane * 2;
        const bits = std.mem.readInt(u16, bytes[offset .. offset + 2][0..2], .little);
        values[lane] = bfloat16.toF32(bits);
    }
    return values;
}

test "wide handwritten f32 row kernel matches generic path" {
    const testing = std.testing;

    inline for (.{ handwritten_hidden_width, handwritten_intermediate_width }) |cols| {
        const row = try testing.allocator.alloc(u8, cols * 4);
        defer testing.allocator.free(row);
        const input = try testing.allocator.alloc(f32, cols);
        defer testing.allocator.free(input);

        for (input, 0..) |*value, idx| {
            value.* = @as(f32, @floatFromInt(@as(i32, @intCast((idx * 7) % 23)) - 11)) / 5.0;
            const weight = @as(f32, @floatFromInt(@as(i32, @intCast((idx * 13 + 3) % 29)) - 14)) / 7.0;
            std.mem.writeInt(u32, row[idx * 4 .. idx * 4 + 4][0..4], @bitCast(weight), .little);
        }

        const generic = dotF32RowGeneric(row, input);
        const handwritten = dotF32Row(row, input);
        try testing.expectApproxEqAbs(generic, handwritten, 3e-5);
    }
}

test "wide handwritten bf16 row kernel matches generic path" {
    const testing = std.testing;

    inline for (.{ handwritten_hidden_width, handwritten_intermediate_width }) |cols| {
        const row = try testing.allocator.alloc(u8, cols * 2);
        defer testing.allocator.free(row);
        const input = try testing.allocator.alloc(f32, cols);
        defer testing.allocator.free(input);

        for (input, 0..) |*value, idx| {
            value.* = @as(f32, @floatFromInt(@as(i32, @intCast((idx * 5) % 19)) - 9)) / 4.0;
            const weight = @as(f32, @floatFromInt(@as(i32, @intCast((idx * 11 + 1) % 31)) - 15)) / 6.0;
            std.mem.writeInt(u16, row[idx * 2 .. idx * 2 + 2][0..2], bfloat16.fromF32(weight), .little);
        }

        const generic = dotBf16RowGeneric(row, input);
        const handwritten = dotBf16Row(row, input);
        try testing.expectApproxEqAbs(generic, handwritten, 1e-6);
    }
}

test "tensor store reads a synthetic bf16 tensor" {
    const testing = std.testing;

    const header =
        \\{"tensor":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}}
    ;

    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    const file = try tmp.dir.createFile("tiny.safetensors", .{});
    defer file.close();

    var length_prefix: [8]u8 = undefined;
    std.mem.writeInt(u64, &length_prefix, header.len, .little);
    try file.writeAll(&length_prefix);
    try file.writeAll(header);

    var payload: [4]u8 = undefined;
    std.mem.writeInt(u16, payload[0..2], bfloat16.fromF32(1.0), .little);
    std.mem.writeInt(u16, payload[2..4], bfloat16.fromF32(-2.5), .little);
    try file.writeAll(&payload);

    var path_buffer: [std.fs.max_path_bytes]u8 = undefined;
    const path = try tmp.dir.realpath("tiny.safetensors", &path_buffer);

    var store = try TensorStore.open(testing.allocator, path);
    defer store.deinit();

    const values = try store.readElementsAsF32Alloc("tensor", 0, 2);
    defer testing.allocator.free(values);

    try testing.expectEqual(@as(usize, 2), values.len);
    try testing.expectEqual(@as(f32, 1.0), values[0]);
    try testing.expectEqual(@as(f32, -2.5), values[1]);
}

test "tensor store matmulVecByName multiplies a synthetic bf16 matrix" {
    const testing = std.testing;

    const header =
        \\{"weight":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]}}
    ;

    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    const file = try tmp.dir.createFile("matmul_bf16.safetensors", .{});
    defer file.close();

    var length_prefix: [8]u8 = undefined;
    std.mem.writeInt(u64, &length_prefix, header.len, .little);
    try file.writeAll(&length_prefix);
    try file.writeAll(header);

    var payload: [12]u8 = undefined;
    const matrix = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    for (matrix, 0..) |value, idx| {
        const start = idx * 2;
        const bits = bfloat16.fromF32(value);
        payload[start] = @truncate(bits & 0xff);
        payload[start + 1] = @truncate(bits >> 8);
    }
    try file.writeAll(&payload);

    var path_buffer: [std.fs.max_path_bytes]u8 = undefined;
    const path = try tmp.dir.realpath("matmul_bf16.safetensors", &path_buffer);

    var store = try TensorStore.open(testing.allocator, path);
    defer store.deinit();

    const input = [_]f32{ 1.0, 0.5, -1.0 };
    var output = [_]f32{ 0.0, 0.0 };

    try store.matmulVecByName(&output, "weight", &input);

    try testing.expectApproxEqAbs(@as(f32, -1.0), output[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.5), output[1], 1e-6);
}
