const std = @import("std");

pub const DType = enum {
    bool,
    u8,
    i8,
    u16,
    i16,
    u32,
    i32,
    u64,
    i64,
    f16,
    bf16,
    f32,
    f64,
    c64,
    f8_e4m3,
    f8_e5m2,

    pub fn fromString(dtype_name: []const u8) !DType {
        if (std.mem.eql(u8, dtype_name, "BOOL")) return .bool;
        if (std.mem.eql(u8, dtype_name, "U8")) return .u8;
        if (std.mem.eql(u8, dtype_name, "I8")) return .i8;
        if (std.mem.eql(u8, dtype_name, "U16")) return .u16;
        if (std.mem.eql(u8, dtype_name, "I16")) return .i16;
        if (std.mem.eql(u8, dtype_name, "U32")) return .u32;
        if (std.mem.eql(u8, dtype_name, "I32")) return .i32;
        if (std.mem.eql(u8, dtype_name, "U64")) return .u64;
        if (std.mem.eql(u8, dtype_name, "I64")) return .i64;
        if (std.mem.eql(u8, dtype_name, "F16")) return .f16;
        if (std.mem.eql(u8, dtype_name, "BF16")) return .bf16;
        if (std.mem.eql(u8, dtype_name, "F32")) return .f32;
        if (std.mem.eql(u8, dtype_name, "F64")) return .f64;
        if (std.mem.eql(u8, dtype_name, "C64")) return .c64;
        if (std.mem.eql(u8, dtype_name, "F8_E4M3")) return .f8_e4m3;
        if (std.mem.eql(u8, dtype_name, "F8_E5M2")) return .f8_e5m2;
        return error.UnsupportedDType;
    }

    pub fn byteSize(self: DType) u64 {
        return switch (self) {
            .bool, .u8, .i8, .f8_e4m3, .f8_e5m2 => 1,
            .u16, .i16, .f16, .bf16 => 2,
            .u32, .i32, .f32 => 4,
            .u64, .i64, .f64, .c64 => 8,
        };
    }

    pub fn name(self: DType) []const u8 {
        return switch (self) {
            .bool => "BOOL",
            .u8 => "U8",
            .i8 => "I8",
            .u16 => "U16",
            .i16 => "I16",
            .u32 => "U32",
            .i32 => "I32",
            .u64 => "U64",
            .i64 => "I64",
            .f16 => "F16",
            .bf16 => "BF16",
            .f32 => "F32",
            .f64 => "F64",
            .c64 => "C64",
            .f8_e4m3 => "F8_E4M3",
            .f8_e5m2 => "F8_E5M2",
        };
    }
};

pub const TensorInfo = struct {
    dtype: DType,
    shape: []const u64,
    data_offsets: [2]u64,
    absolute_offset: u64,

    pub fn rank(self: TensorInfo) usize {
        return self.shape.len;
    }

    pub fn elementCount(self: TensorInfo) !u64 {
        var count: u64 = 1;
        for (self.shape) |dim| {
            count = try std.math.mul(u64, count, dim);
        }
        return count;
    }

    pub fn byteLen(self: TensorInfo) u64 {
        return self.data_offsets[1] - self.data_offsets[0];
    }

    pub fn expectedByteLen(self: TensorInfo) !u64 {
        return std.math.mul(u64, try self.elementCount(), self.dtype.byteSize());
    }
};

pub const ParsedFile = struct {
    arena: std.heap.ArenaAllocator,
    file_size: u64,
    header_len: u64,
    data_start: u64,
    metadata: std.StringArrayHashMapUnmanaged([]const u8),
    tensors: std.StringArrayHashMapUnmanaged(TensorInfo),

    pub fn deinit(self: *ParsedFile) void {
        self.arena.deinit();
    }

    pub fn getTensor(self: *const ParsedFile, name: []const u8) ?TensorInfo {
        return self.tensors.get(name);
    }

    pub fn tensorCount(self: *const ParsedFile) usize {
        return self.tensors.count();
    }
};

pub fn loadFromFile(backing_allocator: std.mem.Allocator, path: []const u8) !ParsedFile {
    const file = try std.fs.cwd().openFile(path, .{});
    return loadFromOpenFile(backing_allocator, file);
}

pub fn loadFromOpenFile(backing_allocator: std.mem.Allocator, file_handle: std.fs.File) !ParsedFile {
    const parsed = try parseFromFileHandle(backing_allocator, file_handle);
    file_handle.close();
    return parsed;
}

pub fn parseFromFileHandle(backing_allocator: std.mem.Allocator, file: std.fs.File) !ParsedFile {
    var arena = std.heap.ArenaAllocator.init(backing_allocator);
    errdefer arena.deinit();

    const allocator = arena.allocator();

    const stat = try file.stat();
    const file_size = stat.size;
    if (file_size < 8) return error.InvalidSafetensorsFile;

    var header_len_bytes: [8]u8 = undefined;
    const prefix_read = try file.preadAll(&header_len_bytes, 0);
    if (prefix_read != header_len_bytes.len) return error.InvalidSafetensorsFile;

    const header_len = std.mem.readInt(u64, &header_len_bytes, .little);
    const data_start = try std.math.add(u64, 8, header_len);
    if (data_start > file_size) return error.InvalidHeaderLength;

    const header_len_usize = std.math.cast(usize, header_len) orelse return error.HeaderTooLarge;
    const header_buffer = try allocator.alloc(u8, header_len_usize);
    const header_read = try file.preadAll(header_buffer, 8);
    if (header_read != header_buffer.len) return error.InvalidSafetensorsFile;

    const header_json = std.mem.trimRight(u8, header_buffer, " ");
    const root = try std.json.parseFromSliceLeaky(std.json.Value, allocator, header_json, .{});
    if (root != .object) return error.InvalidHeaderJson;

    var metadata: std.StringArrayHashMapUnmanaged([]const u8) = .empty;
    var tensors: std.StringArrayHashMapUnmanaged(TensorInfo) = .empty;

    var it = root.object.iterator();
    while (it.next()) |entry| {
        const key = entry.key_ptr.*;
        const value = entry.value_ptr.*;

        if (std.mem.eql(u8, key, "__metadata__")) {
            try parseMetadataObject(allocator, &metadata, value);
            continue;
        }

        const tensor = try parseTensorInfo(allocator, value, data_start);
        try tensors.put(allocator, key, tensor);
    }

    const data_section_len = std.math.sub(u64, file_size, data_start) catch return error.InvalidSafetensorsFile;
    try validateDataLayout(allocator, tensors, data_section_len);

    return .{
        .arena = arena,
        .file_size = file_size,
        .header_len = header_len,
        .data_start = data_start,
        .metadata = metadata,
        .tensors = tensors,
    };
}

pub fn parseFromBytes(backing_allocator: std.mem.Allocator, bytes: []const u8) !ParsedFile {
    var arena = std.heap.ArenaAllocator.init(backing_allocator);
    errdefer arena.deinit();

    const allocator = arena.allocator();
    if (bytes.len < 8) return error.InvalidSafetensorsFile;

    const file_size = bytes.len;
    const header_len = std.mem.readInt(u64, bytes[0..8], .little);
    const data_start = try std.math.add(u64, 8, header_len);
    if (data_start > file_size) return error.InvalidHeaderLength;

    const header_len_usize = std.math.cast(usize, header_len) orelse return error.HeaderTooLarge;
    if (8 + header_len_usize > bytes.len) return error.InvalidSafetensorsFile;
    const header_json = std.mem.trimRight(u8, bytes[8 .. 8 + header_len_usize], " ");
    const root = try std.json.parseFromSliceLeaky(std.json.Value, allocator, header_json, .{});
    if (root != .object) return error.InvalidHeaderJson;

    var metadata: std.StringArrayHashMapUnmanaged([]const u8) = .empty;
    var tensors: std.StringArrayHashMapUnmanaged(TensorInfo) = .empty;

    var it = root.object.iterator();
    while (it.next()) |entry| {
        const key = entry.key_ptr.*;
        const value = entry.value_ptr.*;

        if (std.mem.eql(u8, key, "__metadata__")) {
            try parseMetadataObject(allocator, &metadata, value);
            continue;
        }

        const tensor = try parseTensorInfo(allocator, value, data_start);
        try tensors.put(allocator, key, tensor);
    }

    const data_section_len = std.math.sub(u64, file_size, data_start) catch return error.InvalidSafetensorsFile;
    try validateDataLayout(allocator, tensors, data_section_len);

    return .{
        .arena = arena,
        .file_size = file_size,
        .header_len = header_len,
        .data_start = data_start,
        .metadata = metadata,
        .tensors = tensors,
    };
}

fn parseMetadataObject(
    allocator: std.mem.Allocator,
    metadata: *std.StringArrayHashMapUnmanaged([]const u8),
    value: std.json.Value,
) !void {
    if (value != .object) return error.InvalidMetadataEntry;

    var it = value.object.iterator();
    while (it.next()) |entry| {
        if (entry.value_ptr.* != .string) return error.InvalidMetadataEntry;
        try metadata.put(allocator, entry.key_ptr.*, entry.value_ptr.*.string);
    }
}

fn parseTensorInfo(
    allocator: std.mem.Allocator,
    value: std.json.Value,
    data_start: u64,
) !TensorInfo {
    if (value != .object) return error.InvalidTensorEntry;

    const object = value.object;
    const dtype_value = object.get("dtype") orelse return error.MissingTensorField;
    const shape_value = object.get("shape") orelse return error.MissingTensorField;
    const offsets_value = object.get("data_offsets") orelse return error.MissingTensorField;

    if (dtype_value != .string) return error.InvalidTensorField;
    if (shape_value != .array) return error.InvalidTensorField;
    if (offsets_value != .array) return error.InvalidTensorField;
    if (offsets_value.array.items.len != 2) return error.InvalidTensorField;

    const dtype = try DType.fromString(dtype_value.string);
    const shape = try allocator.alloc(u64, shape_value.array.items.len);
    for (shape_value.array.items, 0..) |dimension, idx| {
        shape[idx] = try jsonNonNegativeInt(dimension);
    }

    const begin = try jsonNonNegativeInt(offsets_value.array.items[0]);
    const end = try jsonNonNegativeInt(offsets_value.array.items[1]);
    if (end < begin) return error.InvalidDataOffsets;

    return .{
        .dtype = dtype,
        .shape = shape,
        .data_offsets = .{ begin, end },
        .absolute_offset = try std.math.add(u64, data_start, begin),
    };
}

fn validateDataLayout(
    allocator: std.mem.Allocator,
    tensors: std.StringArrayHashMapUnmanaged(TensorInfo),
    data_section_len: u64,
) !void {
    const Range = struct {
        name: []const u8,
        begin: u64,
        end: u64,
    };

    const ranges = try allocator.alloc(Range, tensors.count());
    var count: usize = 0;

    var it = tensors.iterator();
    while (it.next()) |entry| {
        const tensor = entry.value_ptr.*;
        if (tensor.data_offsets[1] > data_section_len) return error.TensorOutOfBounds;
        if (try tensor.expectedByteLen() != tensor.byteLen()) return error.TensorByteSizeMismatch;

        ranges[count] = .{
            .name = entry.key_ptr.*,
            .begin = tensor.data_offsets[0],
            .end = tensor.data_offsets[1],
        };
        count += 1;
    }

    std.sort.block(Range, ranges, {}, struct {
        fn lessThan(_: void, lhs: Range, rhs: Range) bool {
            return lhs.begin < rhs.begin;
        }
    }.lessThan);

    var cursor: u64 = 0;
    for (ranges) |range| {
        if (range.begin < cursor) return error.OverlappingTensorData;
        if (range.begin > cursor) return error.NonContiguousTensorData;
        cursor = range.end;
    }

    if (cursor != data_section_len) return error.NonContiguousTensorData;
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

test "parse synthetic safetensors file" {
    const testing = std.testing;

    const header =
        \\{"__metadata__":{"format":"pt"},"tensor_a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"tensor_b":{"dtype":"I16","shape":[2,2],"data_offsets":[8,16]}}
    ;

    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    const file = try tmp.dir.createFile("tiny.safetensors", .{});
    defer file.close();

    var length_prefix: [8]u8 = undefined;
    std.mem.writeInt(u64, &length_prefix, header.len, .little);
    try file.writeAll(&length_prefix);
    try file.writeAll(header);

    const payload = [_]u8{0} ** 16;
    try file.writeAll(&payload);

    const read_file = try tmp.dir.openFile("tiny.safetensors", .{});
    var parsed = try loadFromOpenFile(testing.allocator, read_file);
    defer parsed.deinit();

    try testing.expectEqual(@as(usize, 2), parsed.tensorCount());
    try testing.expectEqualStrings("pt", parsed.metadata.get("format").?);

    const tensor_a = parsed.getTensor("tensor_a").?;
    try testing.expectEqual(DType.f32, tensor_a.dtype);
    try testing.expectEqual(@as(u64, 8), tensor_a.byteLen());
    try testing.expectEqual(@as(usize, 1), tensor_a.rank());

    const tensor_b = parsed.getTensor("tensor_b").?;
    try testing.expectEqual(DType.i16, tensor_b.dtype);
    try testing.expectEqual(@as(u64, 4), try tensor_b.elementCount());
    try testing.expectEqual(@as(u64, 16), tensor_b.data_offsets[1]);
}
