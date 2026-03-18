const std = @import("std");
const bfloat16 = @import("bfloat16.zig");
const safetensors = @import("../format/safetensors.zig");

pub const TensorStore = struct {
    allocator: std.mem.Allocator,
    file: std.fs.File,
    parsed: safetensors.ParsedFile,

    pub fn open(allocator: std.mem.Allocator, weights_path: []const u8) !TensorStore {
        const file = try std.fs.cwd().openFile(weights_path, .{});
        errdefer file.close();

        var parsed = try safetensors.parseFromFileHandle(allocator, file);
        errdefer parsed.deinit();

        return .{
            .allocator = allocator,
            .file = file,
            .parsed = parsed,
        };
    }

    pub fn deinit(self: *TensorStore) void {
        self.file.close();
        self.parsed.deinit();
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

        switch (tensor.dtype) {
            .bf16 => try self.readBf16AsF32(tensor, start_element, output),
            .f32 => try self.readF32(tensor, start_element, output),
            else => return error.UnsupportedTensorDType,
        }

        return output;
    }

    fn readBf16AsF32(
        self: *const TensorStore,
        tensor: safetensors.TensorInfo,
        start_element: u64,
        output: []f32,
    ) !void {
        const bytes_per_item: u64 = 2;
        const byte_offset = try std.math.mul(u64, start_element, bytes_per_item);
        const read_offset = try std.math.add(u64, tensor.absolute_offset, byte_offset);
        const byte_count = try std.math.mul(u64, output.len, bytes_per_item);
        const byte_count_usize = std.math.cast(usize, byte_count) orelse return error.BufferTooLarge;

        const buffer = try self.allocator.alloc(u8, byte_count_usize);
        defer self.allocator.free(buffer);

        const bytes_read = try self.file.preadAll(buffer, read_offset);
        if (bytes_read != buffer.len) return error.UnexpectedEndOfFile;

        for (output, 0..) |*value, idx| {
            const start = idx * 2;
            const bits = std.mem.readInt(u16, buffer[start .. start + 2][0..2], .little);
            value.* = bfloat16.toF32(bits);
        }
    }

    fn readF32(
        self: *const TensorStore,
        tensor: safetensors.TensorInfo,
        start_element: u64,
        output: []f32,
    ) !void {
        const bytes_per_item: u64 = 4;
        const byte_offset = try std.math.mul(u64, start_element, bytes_per_item);
        const read_offset = try std.math.add(u64, tensor.absolute_offset, byte_offset);
        const byte_count = try std.math.mul(u64, output.len, bytes_per_item);
        const byte_count_usize = std.math.cast(usize, byte_count) orelse return error.BufferTooLarge;

        const buffer = try self.allocator.alloc(u8, byte_count_usize);
        defer self.allocator.free(buffer);

        const bytes_read = try self.file.preadAll(buffer, read_offset);
        if (bytes_read != buffer.len) return error.UnexpectedEndOfFile;

        for (output, 0..) |*value, idx| {
            const start = idx * 4;
            const raw = std.mem.readInt(u32, buffer[start .. start + 4][0..4], .little);
            value.* = @bitCast(raw);
        }
    }
};

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

    const read_file = try tmp.dir.openFile("tiny.safetensors", .{});
    defer read_file.close();

    var parsed = try safetensors.parseFromFileHandle(testing.allocator, read_file);
    defer parsed.deinit();

    const reopened = try tmp.dir.openFile("tiny.safetensors", .{});
    var store = TensorStore{
        .allocator = testing.allocator,
        .file = reopened,
        .parsed = parsed,
    };
    defer store.file.close();

    const values = try store.readElementsAsF32Alloc("tensor", 0, 2);
    defer testing.allocator.free(values);

    try testing.expectEqual(@as(usize, 2), values.len);
    try testing.expectEqual(@as(f32, 1.0), values[0]);
    try testing.expectEqual(@as(f32, -2.5), values[1]);
}
