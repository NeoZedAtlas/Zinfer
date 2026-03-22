const std = @import("std");
const safetensors = @import("../../../format/safetensors.zig");
const tensor_store = @import("../../storage/store.zig");
const codec = @import("codec.zig");
const types = @import("types.zig");

const QuantizedEntry = struct {
    name: []const u8,
    encoding: types.Encoding,
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
    scheme: types.Scheme,
) !void {
    var store = try tensor_store.TensorStore.open(allocator, input_path);
    defer store.deinit();

    const temp_path = try std.fmt.allocPrint(allocator, "{s}.tmpdata", .{output_path});
    defer allocator.free(temp_path);

    const temp_file = if (std.fs.path.isAbsolute(temp_path))
        try std.fs.createFileAbsolute(temp_path, .{ .truncate = true, .read = true })
    else
        try std.fs.cwd().createFile(temp_path, .{ .truncate = true, .read = true });
    defer temp_file.close();

    var entries = std.ArrayListUnmanaged(QuantizedEntry).empty;
    defer entries.deinit(allocator);

    const sorted = try sortedSourceTensors(allocator, &store.parsed);
    defer allocator.free(sorted);

    var current_offset: u64 = 0;
    for (sorted) |item| {
        const begin = current_offset;
        const encoding = selectEncoding(item.name, item.info, scheme);

        const row_bytes: u64 = switch (encoding) {
            .f32 => 0,
            .q6_0 => 4 + (std.math.divCeil(u64, item.info.shape[1] * 6, 8) catch return error.InvalidShape),
            .q8_0 => 4 + item.info.shape[1],
            .q4_0 => 4 + (std.math.divCeil(u64, item.info.shape[1], 2) catch return error.InvalidShape),
        };

        if (encoding == .f32) {
            const count = std.math.cast(usize, try item.info.elementCount()) orelse return error.BufferTooLarge;
            const values = try store.readElementsAsF32Alloc(item.name, 0, count);
            defer allocator.free(values);
            var raw = try allocator.alloc(u8, values.len * 4);
            defer allocator.free(raw);
            for (values, 0..) |value, idx| {
                const offset = idx * 4;
                std.mem.writeInt(u32, raw[offset .. offset + 4][0..4], @bitCast(value), .little);
            }
            try temp_file.writeAll(raw);
            current_offset += raw.len;
        } else {
            const rows = std.math.cast(usize, item.info.shape[0]) orelse return error.DimensionTooLarge;
            const cols = std.math.cast(usize, item.info.shape[1]) orelse return error.DimensionTooLarge;
            const input_row_bytes = std.math.cast(usize, item.info.byteLen() / item.info.shape[0]) orelse return error.BufferTooLarge;
            const row_values = try allocator.alloc(f32, cols);
            defer allocator.free(row_values);
            const row_scratch = try allocator.alloc(u8, input_row_bytes);
            defer allocator.free(row_scratch);
            const encoded_row = try allocator.alloc(u8, std.math.cast(usize, row_bytes) orelse return error.BufferTooLarge);
            defer allocator.free(encoded_row);

            for (0..rows) |row_index| {
                try store.readRowAsF32Into(item.name, row_index, row_values, row_scratch);
                switch (encoding) {
                    .q6_0 => codec.encodeQ6Row(encoded_row, row_values),
                    .q8_0 => codec.encodeQ8Row(encoded_row, row_values),
                    .q4_0 => codec.encodeQ4Row(encoded_row, row_values),
                    .f32 => unreachable,
                }
                try temp_file.writeAll(encoded_row);
                current_offset += encoded_row.len;
            }
        }

        try entries.append(allocator, .{
            .name = item.name,
            .encoding = encoding,
            .shape = item.info.shape,
            .row_bytes = row_bytes,
            .data_offsets = .{ begin, current_offset },
        });
    }

    try temp_file.seekTo(0);
    const header = try buildHeader(allocator, entries.items, scheme);
    defer allocator.free(header);

    const output_file = if (std.fs.path.isAbsolute(output_path))
        try std.fs.createFileAbsolute(output_path, .{ .truncate = true })
    else
        try std.fs.cwd().createFile(output_path, .{ .truncate = true });
    defer output_file.close();

    var header_len_bytes: [8]u8 = undefined;
    std.mem.writeInt(u64, &header_len_bytes, header.len, .little);
    try output_file.writeAll(&header_len_bytes);
    try output_file.writeAll(header);

    var copy_buffer: [64 * 1024]u8 = undefined;
    while (true) {
        const read = try temp_file.read(&copy_buffer);
        if (read == 0) break;
        try output_file.writeAll(copy_buffer[0..read]);
    }

    if (std.fs.path.isAbsolute(temp_path)) {
        try std.fs.deleteFileAbsolute(temp_path);
    } else {
        try std.fs.cwd().deleteFile(temp_path);
    }
}

pub fn parseFromBytes(backing_allocator: std.mem.Allocator, bytes: []const u8) !types.ParsedFile {
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
    var tensors: std.StringArrayHashMapUnmanaged(types.TensorInfo) = .empty;

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
            .encoding = try types.Encoding.fromString(encoding_value.string),
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

pub fn elementCount(shape: []const u64) usize {
    var total: usize = 1;
    for (shape) |dim| total *= @intCast(dim);
    return total;
}

pub fn selectEncoding(name: []const u8, info: safetensors.TensorInfo, scheme: types.Scheme) types.Encoding {
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

fn buildHeader(allocator: std.mem.Allocator, entries: []const QuantizedEntry, scheme: types.Scheme) ![]u8 {
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

fn jsonNonNegativeInt(value: std.json.Value) !u64 {
    return switch (value) {
        .integer => |n| {
            if (n < 0) return error.InvalidIntegerValue;
            return @intCast(n);
        },
        else => error.InvalidIntegerValue,
    };
}

test "q4 mixed policy keeps sensitive tensors at q8" {
    const testing = std.testing;

    const matrix_info = safetensors.TensorInfo{
        .dtype = .bf16,
        .shape = &.{ 8, 8 },
        .data_offsets = .{ 0, 128 },
        .absolute_offset = 0,
    };

    try testing.expectEqual(types.Encoding.q8_0, selectEncoding("model.embed_tokens.weight", matrix_info, .q4));
    try testing.expectEqual(types.Encoding.q8_0, selectEncoding("model.layers.0.self_attn.q_proj.weight", matrix_info, .q4));
    try testing.expectEqual(types.Encoding.q4_0, selectEncoding("model.layers.0.mlp.down_proj.weight", matrix_info, .q4));
}
