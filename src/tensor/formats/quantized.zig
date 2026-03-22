const std = @import("std");
const codec = @import("quantized/codec.zig");
const file_impl = @import("quantized/file.zig");
const types = @import("quantized/types.zig");
const kernel_registry = @import("../../kernel/registry.zig");
const safetensors = @import("../../format/safetensors.zig");
const parallel_rows = @import("../parallel/parallel_rows.zig");
const tensor_store = @import("../storage/store.zig");

pub const Scheme = types.Scheme;
pub const Encoding = types.Encoding;
pub const TensorInfo = types.TensorInfo;
pub const ParsedFile = types.ParsedFile;

pub const Store = struct {
    allocator: std.mem.Allocator,
    bytes: []u8,
    parsed: ParsedFile,

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

    pub fn getTensor(self: *const Store, name: []const u8) ?TensorInfo {
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
        tensor: TensorInfo,
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
        tensor: TensorInfo,
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
            .q6_0 => decodeQ6Row(self.bytes, row_offset, output),
            .q8_0 => decodeQ8Row(self.bytes, row_offset, output),
            .q4_0 => decodeQ4Row(self.bytes, row_offset, output),
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
        tensor: TensorInfo,
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
        tensor: TensorInfo,
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
            tensor: TensorInfo,
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
        tensor: TensorInfo,
        input: []const f32,
        start_row: usize,
        end_row: usize,
    ) void {
        var row_offset = tensor.absolute_offset + @as(u64, start_row) * tensor.row_bytes;
        switch (tensor.encoding) {
            .f32 => {
                for (start_row..end_row) |row_idx| {
                    output[row_idx] = dotF32Row(self.bytes, row_offset, input);
                    row_offset += tensor.row_bytes;
                }
            },
            .q6_0 => {
                for (start_row..end_row) |row_idx| {
                    output[row_idx] = dotQ6Row(self.bytes, row_offset, input);
                    row_offset += tensor.row_bytes;
                }
            },
            .q8_0 => {
                for (start_row..end_row) |row_idx| {
                    output[row_idx] = dotQ8Row(self.bytes, row_offset, input);
                    row_offset += tensor.row_bytes;
                }
            },
            .q4_0 => {
                for (start_row..end_row) |row_idx| {
                    output[row_idx] = dotQ4Row(self.bytes, row_offset, input);
                    row_offset += tensor.row_bytes;
                }
            },
        }
    }

    fn matmulWithPool(
        self: *const Store,
        output: []f32,
        tensor: TensorInfo,
        input: []const f32,
        pool: *parallel_rows.Pool,
    ) void {
        const Context = struct {
            store: *const Store,
            output: []f32,
            tensor: TensorInfo,
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

const QuantizedEntry = struct {
    name: []const u8,
    encoding: Encoding,
    shape: []const u64,
    row_bytes: u64,
    data_offsets: [2]u64,
};

const SourceTensor = struct {
    name: []const u8,
    info: safetensors.TensorInfo,
};

pub fn quantizeModel(
    allocator: std.mem.Allocator,
    input_path: []const u8,
    output_path: []const u8,
    scheme: Scheme,
) !void {
    try file_impl.quantizeModel(allocator, input_path, output_path, scheme);
}

fn selectEncoding(name: []const u8, info: safetensors.TensorInfo, scheme: Scheme) Encoding {
    if (info.rank() != 2) return .f32;
    return switch (scheme) {
        .q6 => .q6_0,
        .q8 => .q8_0,
        .q4 => if (shouldKeepQ8InQ4(name)) .q8_0 else .q4_0,
    };
}

fn shouldKeepQ8InQ4(name: []const u8) bool {
    return std.mem.eql(u8, name, "model.embed_tokens.weight") or
        std.mem.eql(u8, name, "lm_head.weight") or
        std.mem.endsWith(u8, name, ".self_attn.q_proj.weight") or
        std.mem.endsWith(u8, name, ".self_attn.k_proj.weight") or
        std.mem.endsWith(u8, name, ".self_attn.v_proj.weight") or
        std.mem.endsWith(u8, name, ".self_attn.o_proj.weight");
}

fn sortedSourceTensors(allocator: std.mem.Allocator, parsed: *const safetensors.ParsedFile) ![]SourceTensor {
    const items = try allocator.alloc(SourceTensor, parsed.tensors.count());
    var index: usize = 0;
    var it = parsed.tensors.iterator();
    while (it.next()) |entry| {
        items[index] = .{
            .name = entry.key_ptr.*,
            .info = entry.value_ptr.*,
        };
        index += 1;
    }
    std.sort.block(SourceTensor, items, {}, struct {
        fn lessThan(_: void, lhs: SourceTensor, rhs: SourceTensor) bool {
            return std.mem.lessThan(u8, lhs.name, rhs.name);
        }
    }.lessThan);
    return items;
}

fn buildHeader(allocator: std.mem.Allocator, entries: []const QuantizedEntry, scheme: Scheme) ![]u8 {
    var output = std.ArrayListUnmanaged(u8).empty;
    defer output.deinit(allocator);
    const writer = output.writer(allocator);

    try writer.writeAll("{\"__metadata__\":{");
    try writer.print("\"format\":{f},\"scheme\":{f}", .{
        std.json.fmt("zinfer_quantized", .{}),
        std.json.fmt(scheme.name(), .{}),
    });
    try writer.writeAll("}");

    for (entries) |entry| {
        try writer.writeByte(',');
        try writer.print("{f}:{{\"encoding\":{f},\"shape\":[", .{
            std.json.fmt(entry.name, .{}),
            std.json.fmt(entry.encoding.name(), .{}),
        });
        for (entry.shape, 0..) |dim, idx| {
            if (idx != 0) try writer.writeByte(',');
            try writer.print("{d}", .{dim});
        }
        try writer.print("],\"row_bytes\":{d},\"data_offsets\":[{d},{d}]}}", .{
            entry.row_bytes,
            entry.data_offsets[0],
            entry.data_offsets[1],
        });
    }

    try writer.writeAll("}");
    return output.toOwnedSlice(allocator);
}

fn parseFromBytes(backing_allocator: std.mem.Allocator, bytes: []const u8) !ParsedFile {
    var arena = std.heap.ArenaAllocator.init(backing_allocator);
    errdefer arena.deinit();
    const allocator = arena.allocator();

    if (bytes.len < 8) return error.InvalidQuantizedFile;
    const header_len = std.mem.readInt(u64, bytes[0..8], .little);
    const data_start = 8 + header_len;
    if (data_start > bytes.len) return error.InvalidQuantizedFile;

    const header = bytes[8 .. 8 + header_len];
    const root = try std.json.parseFromSliceLeaky(std.json.Value, allocator, header, .{});
    if (root != .object) return error.InvalidHeaderJson;

    var metadata: std.StringArrayHashMapUnmanaged([]const u8) = .empty;
    var tensors: std.StringArrayHashMapUnmanaged(TensorInfo) = .empty;

    var it = root.object.iterator();
    while (it.next()) |entry| {
        if (std.mem.eql(u8, entry.key_ptr.*, "__metadata__")) {
            if (entry.value_ptr.* != .object) return error.InvalidMetadataEntry;
            var meta_it = entry.value_ptr.*.object.iterator();
            while (meta_it.next()) |meta| {
                if (meta.value_ptr.* != .string) return error.InvalidMetadataEntry;
                try metadata.put(allocator, meta.key_ptr.*, meta.value_ptr.*.string);
            }
            continue;
        }

        if (entry.value_ptr.* != .object) return error.InvalidTensorEntry;
        const obj = entry.value_ptr.*.object;
        const encoding_value = obj.get("encoding") orelse return error.MissingTensorField;
        const shape_value = obj.get("shape") orelse return error.MissingTensorField;
        const row_bytes_value = obj.get("row_bytes") orelse return error.MissingTensorField;
        const offsets_value = obj.get("data_offsets") orelse return error.MissingTensorField;
        if (encoding_value != .string or shape_value != .array or row_bytes_value != .integer or offsets_value != .array) {
            return error.InvalidTensorField;
        }

        const shape = try allocator.alloc(u64, shape_value.array.items.len);
        for (shape_value.array.items, 0..) |dim, idx| {
            shape[idx] = try jsonNonNegativeInt(dim);
        }
        const begin = try jsonNonNegativeInt(offsets_value.array.items[0]);
        const end = try jsonNonNegativeInt(offsets_value.array.items[1]);
        const row_bytes = try jsonNonNegativeInt(row_bytes_value);
        try tensors.put(allocator, entry.key_ptr.*, .{
            .encoding = try Encoding.fromString(encoding_value.string),
            .shape = shape,
            .row_bytes = row_bytes,
            .data_offsets = .{ begin, end },
            .absolute_offset = data_start + begin,
        });
    }

    return .{
        .arena = arena,
        .file_size = bytes.len,
        .header_len = header_len,
        .data_start = data_start,
        .metadata = metadata,
        .tensors = tensors,
    };
}

fn elementCount(shape: []const u64) usize {
    var total: usize = 1;
    for (shape) |dim| total *= @intCast(dim);
    return total;
}

fn jsonNonNegativeInt(value: std.json.Value) !u64 {
    return switch (value) {
        .integer => |n| {
            if (n < 0) return error.InvalidIntegerValue;
            return @intCast(n);
        },
        else => error.InvalidIntegerValue,
    };
}

pub fn encodeQ8Row(output: []u8, values: []const f32) void {
    codec.encodeQ8Row(output, values);
}

pub fn encodeQ6Row(output: []u8, values: []const f32) void {
    codec.encodeQ6Row(output, values);
}

pub fn encodeQ4Row(output: []u8, values: []const f32) void {
    codec.encodeQ4Row(output, values);
}

fn decodeQ6Row(bytes: []const u8, row_offset: u64, output: []f32) void {
    const start = @as(usize, @intCast(row_offset));
    const scale_bits = std.mem.readInt(u32, bytes[start .. start + 4][0..4], .little);
    const scale: f32 = @bitCast(scale_bits);
    const payload = bytes[start + 4 ..];

    var out_index: usize = 0;
    var payload_index: usize = 0;
    while (out_index + 8 <= output.len) : ({
        out_index += 8;
        payload_index += 6;
    }) {
        const packed24_a = @as(u32, payload[payload_index]) |
            (@as(u32, payload[payload_index + 1]) << 8) |
            (@as(u32, payload[payload_index + 2]) << 16);
        const packed24_b = @as(u32, payload[payload_index + 3]) |
            (@as(u32, payload[payload_index + 4]) << 8) |
            (@as(u32, payload[payload_index + 5]) << 16);

        inline for (0..4) |lane| {
            const encoded: u8 = @intCast((packed24_a >> (lane * 6)) & 0x3F);
            const q: i32 = @as(i32, encoded) - 32;
            output[out_index + lane] = @as(f32, @floatFromInt(q)) * scale;
        }
        inline for (0..4) |lane| {
            const encoded: u8 = @intCast((packed24_b >> (lane * 6)) & 0x3F);
            const q: i32 = @as(i32, encoded) - 32;
            output[out_index + 4 + lane] = @as(f32, @floatFromInt(q)) * scale;
        }
    }
    while (out_index + 4 <= output.len) : ({
        out_index += 4;
        payload_index += 3;
    }) {
        const packed24 = @as(u32, payload[payload_index]) |
            (@as(u32, payload[payload_index + 1]) << 8) |
            (@as(u32, payload[payload_index + 2]) << 16);
        inline for (0..4) |lane| {
            const encoded: u8 = @intCast((packed24 >> (lane * 6)) & 0x3F);
            const q: i32 = @as(i32, encoded) - 32;
            output[out_index + lane] = @as(f32, @floatFromInt(q)) * scale;
        }
    }
    while (out_index < output.len) : (out_index += 1) {
        const bit_index = out_index * 6;
        const encoded = readPackedBits(payload, bit_index, 6);
        const q: i32 = @as(i32, encoded) - 32;
        output[out_index] = @as(f32, @floatFromInt(q)) * scale;
    }
}

fn decodeQ8Row(bytes: []const u8, row_offset: u64, output: []f32) void {
    const start = @as(usize, @intCast(row_offset));
    const scale_bits = std.mem.readInt(u32, bytes[start .. start + 4][0..4], .little);
    const scale: f32 = @bitCast(scale_bits);
    for (output, 0..) |*value, idx| {
        const q: i8 = @bitCast(bytes[start + 4 + idx]);
        value.* = @as(f32, @floatFromInt(q)) * scale;
    }
}

fn decodeQ4Row(bytes: []const u8, row_offset: u64, output: []f32) void {
    const start = @as(usize, @intCast(row_offset));
    const scale_bits = std.mem.readInt(u32, bytes[start .. start + 4][0..4], .little);
    const scale: f32 = @bitCast(scale_bits);
    for (output, 0..) |*value, idx| {
        const packed_byte = bytes[start + 4 + idx / 2];
        const nibble = if (idx % 2 == 0) packed_byte & 0x0F else packed_byte >> 4;
        const q: i8 = @intCast(@as(i16, nibble) - 8);
        value.* = @as(f32, @floatFromInt(q)) * scale;
    }
}

pub fn dotQ6Row(bytes: []const u8, row_offset: u64, input: []const f32) f32 {
    return switch (kernel_registry.resolve(.{ .gemv_row = .{ .op = .q6_row, .cols = input.len } }).shape) {
        .qwen3_hidden_1024 => dotQ6RowFixed(tensor_store.handwritten_hidden_width, bytes, row_offset, input),
        .qwen3_intermediate_3072 => dotQ6RowFixed(tensor_store.handwritten_intermediate_width, bytes, row_offset, input),
        else => dotQ6RowGeneric(bytes, row_offset, input),
    };
}

fn dotQ6RowGeneric(bytes: []const u8, row_offset: u64, input: []const f32) f32 {
    const start = @as(usize, @intCast(row_offset));
    const scale_bits = std.mem.readInt(u32, bytes[start .. start + 4][0..4], .little);
    const scale: f32 = @bitCast(scale_bits);
    const payload = bytes[start + 4 ..];

    var sum: f32 = 0.0;
    var index: usize = 0;
    var payload_index: usize = 0;
    while (index + 8 <= input.len) : ({
        index += 8;
        payload_index += 6;
    }) {
        const packed24_a = @as(u32, payload[payload_index]) |
            (@as(u32, payload[payload_index + 1]) << 8) |
            (@as(u32, payload[payload_index + 2]) << 16);
        const packed24_b = @as(u32, payload[payload_index + 3]) |
            (@as(u32, payload[payload_index + 4]) << 8) |
            (@as(u32, payload[payload_index + 5]) << 16);

        var q_arr: [8]f32 = undefined;
        inline for (0..4) |lane| {
            const encoded: u8 = @intCast((packed24_a >> (lane * 6)) & 0x3F);
            const q: i32 = @as(i32, encoded) - 32;
            q_arr[lane] = @floatFromInt(q);
        }
        inline for (0..4) |lane| {
            const encoded: u8 = @intCast((packed24_b >> (lane * 6)) & 0x3F);
            const q: i32 = @as(i32, encoded) - 32;
            q_arr[4 + lane] = @floatFromInt(q);
        }
        const qv: @Vector(8, f32) = q_arr;
        const rhs: @Vector(8, f32) = input[index..][0..8].*;
        sum += @reduce(.Add, qv * rhs);
    }
    while (index + 4 <= input.len) : ({
        index += 4;
        payload_index += 3;
    }) {
        const packed24 = @as(u32, payload[payload_index]) |
            (@as(u32, payload[payload_index + 1]) << 8) |
            (@as(u32, payload[payload_index + 2]) << 16);

        var q_arr: [4]f32 = undefined;
        inline for (0..4) |lane| {
            const encoded: u8 = @intCast((packed24 >> (lane * 6)) & 0x3F);
            const q: i32 = @as(i32, encoded) - 32;
            q_arr[lane] = @floatFromInt(q);
        }
        const qv: @Vector(4, f32) = q_arr;
        const rhs: @Vector(4, f32) = input[index..][0..4].*;
        sum += @reduce(.Add, qv * rhs);
    }
    while (index < input.len) : (index += 1) {
        const bit_index = index * 6;
        const encoded = readPackedBits(payload, bit_index, 6);
        const q: i32 = @as(i32, encoded) - 32;
        sum += @as(f32, @floatFromInt(q)) * input[index];
    }
    return sum * scale;
}

fn dotQ6RowFixed(comptime cols: usize, bytes: []const u8, row_offset: u64, input: []const f32) f32 {
    std.debug.assert(input.len == cols);

    const start = @as(usize, @intCast(row_offset));
    const scale_bits = std.mem.readInt(u32, bytes[start .. start + 4][0..4], .little);
    const scale: f32 = @bitCast(scale_bits);
    const payload = bytes[start + 4 ..];

    var acc0: @Vector(8, f32) = @splat(0.0);
    var acc1: @Vector(8, f32) = @splat(0.0);
    var index: usize = 0;
    var payload_index: usize = 0;
    while (index < cols) : ({
        index += 16;
        payload_index += 12;
    }) {
        const q0 = loadQ6Vector8(payload, payload_index);
        const q1 = loadQ6Vector8(payload, payload_index + 6);
        const rhs0: @Vector(8, f32) = input[index..][0..8].*;
        const rhs1: @Vector(8, f32) = input[index + 8 ..][0..8].*;
        acc0 += q0 * rhs0;
        acc1 += q1 * rhs1;
    }
    return @reduce(.Add, acc0 + acc1) * scale;
}

pub fn matmulQ6Rows(output: []f32, bytes: []const u8, row_bytes: usize, input: []const f32) void {
    std.debug.assert(row_bytes == 4 + (std.math.divCeil(usize, input.len * 6, 8) catch unreachable));
    std.debug.assert(bytes.len == output.len * row_bytes);

    var row_offset: u64 = 0;
    for (output) |*value| {
        value.* = dotQ6Row(bytes, row_offset, input);
        row_offset += row_bytes;
    }
}

fn writePackedBits(buffer: []u8, bit_index: usize, bit_width: u8, value: u8) void {
    var remaining = bit_width;
    var source: u16 = value;
    var dst_bit_index = bit_index;
    while (remaining > 0) {
        const byte_index = dst_bit_index / 8;
        const bit_offset: u3 = @intCast(dst_bit_index % 8);
        const available: u8 = 8 - @as(u8, bit_offset);
        const chunk_bits: u8 = @min(remaining, available);
        const mask: u16 = (@as(u16, 1) << @intCast(chunk_bits)) - 1;
        const chunk: u8 = @intCast(source & mask);
        buffer[byte_index] |= chunk << bit_offset;
        source >>= @intCast(chunk_bits);
        dst_bit_index += chunk_bits;
        remaining -= chunk_bits;
    }
}

fn readPackedBits(buffer: []const u8, bit_index: usize, bit_width: u8) u8 {
    var remaining = bit_width;
    var src_bit_index = bit_index;
    var result: u16 = 0;
    var result_shift: u8 = 0;
    while (remaining > 0) {
        const byte_index = src_bit_index / 8;
        const bit_offset: u3 = @intCast(src_bit_index % 8);
        const available: u8 = 8 - @as(u8, bit_offset);
        const chunk_bits: u8 = @min(remaining, available);
        const mask: u8 = (@as(u8, 1) << @intCast(chunk_bits)) - 1;
        const chunk = (buffer[byte_index] >> bit_offset) & mask;
        result |= @as(u16, chunk) << @intCast(result_shift);
        src_bit_index += chunk_bits;
        result_shift += chunk_bits;
        remaining -= chunk_bits;
    }
    return @intCast(result);
}

fn dotF32Row(bytes: []const u8, row_offset: u64, input: []const f32) f32 {
    const start = @as(usize, @intCast(row_offset));
    return tensor_store.dotF32Row(bytes[start .. start + input.len * 4], input);
}

pub fn dotQ8Row(bytes: []const u8, row_offset: u64, input: []const f32) f32 {
    return switch (kernel_registry.resolve(.{ .gemv_row = .{ .op = .q8_row, .cols = input.len } }).shape) {
        .qwen3_hidden_1024 => dotQ8RowFixed(tensor_store.handwritten_hidden_width, bytes, row_offset, input),
        .qwen3_intermediate_3072 => dotQ8RowFixed(tensor_store.handwritten_intermediate_width, bytes, row_offset, input),
        else => dotQ8RowGeneric(bytes, row_offset, input),
    };
}

fn dotQ8RowGeneric(bytes: []const u8, row_offset: u64, input: []const f32) f32 {
    const start = @as(usize, @intCast(row_offset));
    const scale_bits = std.mem.readInt(u32, bytes[start .. start + 4][0..4], .little);
    const scale: f32 = @bitCast(scale_bits);
    var sum: f32 = 0.0;
    var index: usize = 0;
    while (index + 16 <= input.len) : (index += 16) {
        var q_arr: [16]f32 = undefined;
        inline for (0..16) |lane| {
            const q: i8 = @bitCast(bytes[start + 4 + index + lane]);
            q_arr[lane] = @floatFromInt(q);
        }
        const qv: @Vector(16, f32) = q_arr;
        const rhs: @Vector(16, f32) = input[index..][0..16].*;
        sum += @reduce(.Add, qv * rhs);
    }
    while (index < input.len) : (index += 1) {
        const q: i8 = @bitCast(bytes[start + 4 + index]);
        sum += @as(f32, @floatFromInt(q)) * input[index];
    }
    return sum * scale;
}

fn dotQ8RowFixed(comptime cols: usize, bytes: []const u8, row_offset: u64, input: []const f32) f32 {
    std.debug.assert(input.len == cols);

    const start = @as(usize, @intCast(row_offset));
    const scale_bits = std.mem.readInt(u32, bytes[start .. start + 4][0..4], .little);
    const scale: f32 = @bitCast(scale_bits);
    const payload = bytes[start + 4 ..];

    var acc0: @Vector(16, f32) = @splat(0.0);
    var acc1: @Vector(16, f32) = @splat(0.0);
    var index: usize = 0;
    while (index < cols) : (index += 32) {
        const q0 = loadQ8Vector16(payload, index);
        const q1 = loadQ8Vector16(payload, index + 16);
        const rhs0: @Vector(16, f32) = input[index..][0..16].*;
        const rhs1: @Vector(16, f32) = input[index + 16 ..][0..16].*;
        acc0 += q0 * rhs0;
        acc1 += q1 * rhs1;
    }
    return @reduce(.Add, acc0 + acc1) * scale;
}

pub fn matmulQ8Rows(output: []f32, bytes: []const u8, row_bytes: usize, input: []const f32) void {
    std.debug.assert(row_bytes == 4 + input.len);
    std.debug.assert(bytes.len == output.len * row_bytes);

    var row_offset: u64 = 0;
    for (output) |*value| {
        value.* = dotQ8Row(bytes, row_offset, input);
        row_offset += row_bytes;
    }
}

pub fn dotQ4Row(bytes: []const u8, row_offset: u64, input: []const f32) f32 {
    return switch (kernel_registry.resolve(.{ .gemv_row = .{ .op = .q4_row, .cols = input.len } }).shape) {
        .qwen3_hidden_1024 => dotQ4RowFixed(tensor_store.handwritten_hidden_width, bytes, row_offset, input),
        .qwen3_intermediate_3072 => dotQ4RowFixed(tensor_store.handwritten_intermediate_width, bytes, row_offset, input),
        else => dotQ4RowGeneric(bytes, row_offset, input),
    };
}

fn dotQ4RowGeneric(bytes: []const u8, row_offset: u64, input: []const f32) f32 {
    const start = @as(usize, @intCast(row_offset));
    const scale_bits = std.mem.readInt(u32, bytes[start .. start + 4][0..4], .little);
    const scale: f32 = @bitCast(scale_bits);
    var sum: f32 = 0.0;
    var index: usize = 0;
    while (index + 16 <= input.len) : (index += 16) {
        var low_arr: [8]f32 = undefined;
        var high_arr: [8]f32 = undefined;
        var low_rhs_arr: [8]f32 = undefined;
        var high_rhs_arr: [8]f32 = undefined;
        inline for (0..8) |lane| {
            const packed_byte = bytes[start + 4 + index / 2 + lane];
            const low_nibble = packed_byte & 0x0F;
            const high_nibble = packed_byte >> 4;
            low_arr[lane] = @floatFromInt(@as(i8, @intCast(@as(i16, low_nibble) - 8)));
            high_arr[lane] = @floatFromInt(@as(i8, @intCast(@as(i16, high_nibble) - 8)));
            low_rhs_arr[lane] = input[index + lane * 2];
            high_rhs_arr[lane] = input[index + lane * 2 + 1];
        }
        const low: @Vector(8, f32) = low_arr;
        const high: @Vector(8, f32) = high_arr;
        const low_rhs: @Vector(8, f32) = low_rhs_arr;
        const high_rhs: @Vector(8, f32) = high_rhs_arr;
        sum += @reduce(.Add, low * low_rhs) + @reduce(.Add, high * high_rhs);
    }
    while (index + 8 <= input.len) : (index += 8) {
        var q_arr: [8]f32 = undefined;
        var rhs_arr: [8]f32 = undefined;
        inline for (0..8) |lane| {
            const packed_byte = bytes[start + 4 + (index + lane) / 2];
            const nibble = if ((index + lane) % 2 == 0) packed_byte & 0x0F else packed_byte >> 4;
            const q: i8 = @intCast(@as(i16, nibble) - 8);
            q_arr[lane] = @floatFromInt(q);
            rhs_arr[lane] = input[index + lane];
        }
        const qv: @Vector(8, f32) = q_arr;
        const rhs: @Vector(8, f32) = rhs_arr;
        sum += @reduce(.Add, qv * rhs);
    }
    while (index < input.len) : (index += 1) {
        const packed_byte = bytes[start + 4 + index / 2];
        const nibble = if (index % 2 == 0) packed_byte & 0x0F else packed_byte >> 4;
        const q: i8 = @intCast(@as(i16, nibble) - 8);
        sum += @as(f32, @floatFromInt(q)) * input[index];
    }
    return sum * scale;
}

pub fn matmulQ4Rows(output: []f32, bytes: []const u8, row_bytes: usize, input: []const f32) void {
    std.debug.assert(row_bytes == 4 + (std.math.divCeil(usize, input.len, 2) catch unreachable));
    std.debug.assert(bytes.len == output.len * row_bytes);

    var row_offset: u64 = 0;
    for (output) |*value| {
        value.* = dotQ4Row(bytes, row_offset, input);
        row_offset += row_bytes;
    }
}

fn dotQ4RowFixed(comptime cols: usize, bytes: []const u8, row_offset: u64, input: []const f32) f32 {
    std.debug.assert(input.len == cols);

    const start = @as(usize, @intCast(row_offset));
    const scale_bits = std.mem.readInt(u32, bytes[start .. start + 4][0..4], .little);
    const scale: f32 = @bitCast(scale_bits);
    const payload = bytes[start + 4 ..];

    var sum: f32 = 0.0;
    var index: usize = 0;
    while (index < cols) : (index += 16) {
        var low_arr: [8]f32 = undefined;
        var high_arr: [8]f32 = undefined;
        var low_rhs_arr: [8]f32 = undefined;
        var high_rhs_arr: [8]f32 = undefined;
        inline for (0..8) |lane| {
            const packed_byte = payload[index / 2 + lane];
            const low_nibble = packed_byte & 0x0F;
            const high_nibble = packed_byte >> 4;
            low_arr[lane] = @floatFromInt(@as(i8, @intCast(@as(i16, low_nibble) - 8)));
            high_arr[lane] = @floatFromInt(@as(i8, @intCast(@as(i16, high_nibble) - 8)));
            low_rhs_arr[lane] = input[index + lane * 2];
            high_rhs_arr[lane] = input[index + lane * 2 + 1];
        }
        const low: @Vector(8, f32) = low_arr;
        const high: @Vector(8, f32) = high_arr;
        const low_rhs: @Vector(8, f32) = low_rhs_arr;
        const high_rhs: @Vector(8, f32) = high_rhs_arr;
        sum += @reduce(.Add, low * low_rhs) + @reduce(.Add, high * high_rhs);
    }
    return sum * scale;
}

fn loadQ8Vector16(bytes: []const u8, start: usize) @Vector(16, f32) {
    var values: [16]f32 = undefined;
    inline for (0..16) |lane| {
        const q: i8 = @bitCast(bytes[start + lane]);
        values[lane] = @floatFromInt(q);
    }
    return values;
}

fn loadQ6Vector8(bytes: []const u8, start: usize) @Vector(8, f32) {
    const packed24_a = @as(u32, bytes[start]) |
        (@as(u32, bytes[start + 1]) << 8) |
        (@as(u32, bytes[start + 2]) << 16);
    const packed24_b = @as(u32, bytes[start + 3]) |
        (@as(u32, bytes[start + 4]) << 8) |
        (@as(u32, bytes[start + 5]) << 16);

    var values: [8]f32 = undefined;
    inline for (0..4) |lane| {
        const encoded: u8 = @intCast((packed24_a >> (lane * 6)) & 0x3F);
        values[lane] = @floatFromInt(@as(i32, encoded) - 32);
    }
    inline for (0..4) |lane| {
        const encoded: u8 = @intCast((packed24_b >> (lane * 6)) & 0x3F);
        values[4 + lane] = @floatFromInt(@as(i32, encoded) - 32);
    }
    return values;
}

test "wide quantized handwritten kernels match generic path" {
    const testing = std.testing;

    inline for (.{ tensor_store.handwritten_hidden_width, tensor_store.handwritten_intermediate_width }) |cols| {
        const values = try testing.allocator.alloc(f32, cols);
        defer testing.allocator.free(values);
        const input = try testing.allocator.alloc(f32, cols);
        defer testing.allocator.free(input);
        const row_q8 = try testing.allocator.alloc(u8, 4 + cols);
        defer testing.allocator.free(row_q8);
        const row_q6 = try testing.allocator.alloc(u8, 4 + (try std.math.divCeil(usize, cols * 6, 8)));
        defer testing.allocator.free(row_q6);
        const row_q4 = try testing.allocator.alloc(u8, 4 + (try std.math.divCeil(usize, cols, 2)));
        defer testing.allocator.free(row_q4);

        for (values, 0..) |*value, idx| {
            value.* = @as(f32, @floatFromInt(@as(i32, @intCast((idx * 17 + 5) % 41)) - 20)) / 9.0;
            input[idx] = @as(f32, @floatFromInt(@as(i32, @intCast((idx * 9 + 7) % 37)) - 18)) / 8.0;
        }

        encodeQ8Row(row_q8, values);
        encodeQ6Row(row_q6, values);
        encodeQ4Row(row_q4, values);

        try testing.expectApproxEqAbs(
            dotQ8RowGeneric(row_q8, 0, input),
            dotQ8Row(row_q8, 0, input),
            1e-6,
        );
        try testing.expectApproxEqAbs(
            dotQ6RowGeneric(row_q6, 0, input),
            dotQ6Row(row_q6, 0, input),
            1e-6,
        );
        try testing.expectApproxEqAbs(
            dotQ4RowGeneric(row_q4, 0, input),
            dotQ4Row(row_q4, 0, input),
            1e-6,
        );
    }
}

test "quantized q8 row roundtrip and dot" {
    const testing = std.testing;
    const values = [_]f32{ 1.0, -2.0, 0.5, 3.0 };
    var row: [8]u8 = undefined;
    encodeQ8Row(&row, &values);

    var decoded = [_]f32{ 0.0, 0.0, 0.0, 0.0 };
    decodeQ8Row(&row, 0, &decoded);
    try testing.expectApproxEqAbs(@as(f32, 1.0), decoded[0], 0.05);
    try testing.expectApproxEqAbs(@as(f32, -2.0), decoded[1], 0.05);

    const input = [_]f32{ 1.0, 2.0, -1.0, 0.5 };
    const approx = dotQ8Row(&row, 0, &input);
    try testing.expectApproxEqAbs(@as(f32, -2.0), approx, 0.2);
}

test "quantized store matmulVecByName matches generic q8 rows for hot width" {
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
        encodeQ8Row(encoded_row, row_values);
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
        const expected = dotQ8RowGeneric(store.bytes, row_offset, input);
        try testing.expectApproxEqAbs(expected, output[row_idx], 1e-4);
    }
}

test "quantized store matmulVecByName matches generic q6 rows for hot width" {
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
        encodeQ6Row(encoded_row, row_values);
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
        const expected = dotQ6RowGeneric(store.bytes, row_offset, input);
        try testing.expectApproxEqAbs(expected, output[row_idx], 1e-4);
    }
}

test "quantized q6 row roundtrip and dot" {
    const testing = std.testing;
    const values = [_]f32{ 1.0, -2.0, 0.5, 3.0 };
    var row: [7]u8 = undefined;
    encodeQ6Row(&row, &values);

    var decoded = [_]f32{ 0.0, 0.0, 0.0, 0.0 };
    decodeQ6Row(&row, 0, &decoded);
    try testing.expectApproxEqAbs(@as(f32, 1.0), decoded[0], 0.2);
    try testing.expectApproxEqAbs(@as(f32, -2.0), decoded[1], 0.2);

    const input = [_]f32{ 1.0, 2.0, -1.0, 0.5 };
    const approx = dotQ6Row(&row, 0, &input);
    try testing.expectApproxEqAbs(@as(f32, -2.0), approx, 0.4);
}

test "quantized q4 row roundtrip and dot" {
    const testing = std.testing;
    const values = [_]f32{ 1.0, -2.0, 0.5, 3.0 };
    var row: [6]u8 = undefined;
    encodeQ4Row(&row, &values);

    var decoded = [_]f32{ 0.0, 0.0, 0.0, 0.0 };
    decodeQ4Row(&row, 0, &decoded);
    try testing.expectApproxEqAbs(@as(f32, 1.0), decoded[0], 0.5);
    try testing.expectApproxEqAbs(@as(f32, -2.0), decoded[1], 0.5);

    const input = [_]f32{ 1.0, 2.0, -1.0, 0.5 };
    const approx = dotQ4Row(&row, 0, &input);
    try testing.expectApproxEqAbs(@as(f32, -2.0), approx, 0.8);
}

test "q4 mixed policy keeps sensitive tensors at q8" {
    const testing = std.testing;

    const matrix_info = safetensors.TensorInfo{
        .dtype = .bf16,
        .shape = &.{ 8, 8 },
        .data_offsets = .{ 0, 128 },
        .absolute_offset = 0,
    };

    try testing.expectEqual(
        Encoding.q8_0,
        selectEncoding("model.embed_tokens.weight", matrix_info, .q4),
    );
    try testing.expectEqual(
        Encoding.q8_0,
        selectEncoding("model.layers.0.self_attn.q_proj.weight", matrix_info, .q4),
    );
    try testing.expectEqual(
        Encoding.q4_0,
        selectEncoding("model.layers.0.mlp.down_proj.weight", matrix_info, .q4),
    );
}
