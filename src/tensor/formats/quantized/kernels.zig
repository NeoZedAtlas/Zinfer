const std = @import("std");
const kernel_registry = @import("../../../kernel/registry.zig");
const tensor_store = @import("../../storage/store.zig");
const codec = @import("codec.zig");

pub fn encodeQ8Row(output: []u8, values: []const f32) void {
    codec.encodeQ8Row(output, values);
}

pub fn encodeQ6Row(output: []u8, values: []const f32) void {
    codec.encodeQ6Row(output, values);
}

pub fn encodeQ4Row(output: []u8, values: []const f32) void {
    codec.encodeQ4Row(output, values);
}

pub fn decodeQ6Row(bytes: []const u8, row_offset: u64, output: []f32) void {
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

pub fn decodeQ8Row(bytes: []const u8, row_offset: u64, output: []f32) void {
    const start = @as(usize, @intCast(row_offset));
    const scale_bits = std.mem.readInt(u32, bytes[start .. start + 4][0..4], .little);
    const scale: f32 = @bitCast(scale_bits);
    for (output, 0..) |*value, idx| {
        const q: i8 = @bitCast(bytes[start + 4 + idx]);
        value.* = @as(f32, @floatFromInt(q)) * scale;
    }
}

pub fn decodeQ4Row(bytes: []const u8, row_offset: u64, output: []f32) void {
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
    var acc2: @Vector(8, f32) = @splat(0.0);
    var acc3: @Vector(8, f32) = @splat(0.0);
    var index: usize = 0;
    var payload_index: usize = 0;
    while (index < cols) : ({
        index += 32;
        payload_index += 24;
    }) {
        const q0 = loadQ6Vector8(payload, payload_index);
        const q1 = loadQ6Vector8(payload, payload_index + 6);
        const q2 = loadQ6Vector8(payload, payload_index + 12);
        const q3 = loadQ6Vector8(payload, payload_index + 18);
        const rhs0: @Vector(8, f32) = input[index..][0..8].*;
        const rhs1: @Vector(8, f32) = input[index + 8 ..][0..8].*;
        const rhs2: @Vector(8, f32) = input[index + 16 ..][0..8].*;
        const rhs3: @Vector(8, f32) = input[index + 24 ..][0..8].*;
        acc0 += q0 * rhs0;
        acc1 += q1 * rhs1;
        acc2 += q2 * rhs2;
        acc3 += q3 * rhs3;
    }
    return @reduce(.Add, acc0 + acc1 + acc2 + acc3) * scale;
}

pub fn matmulQ6Rows(output: []f32, bytes: []const u8, row_bytes: usize, input: []const f32) void {
    std.debug.assert(row_bytes == 4 + (std.math.divCeil(usize, input.len * 6, 8) catch unreachable));
    std.debug.assert(bytes.len == output.len * row_bytes);

    switch (kernel_registry.resolve(.{ .gemv_row = .{ .op = .q6_row, .cols = input.len } }).shape) {
        .qwen3_hidden_1024 => {
            matmulQ6RowsFixedBlocks(tensor_store.handwritten_hidden_width, output, bytes, row_bytes, input);
            return;
        },
        .qwen3_intermediate_3072 => {
            matmulQ6RowsFixedBlocks(tensor_store.handwritten_intermediate_width, output, bytes, row_bytes, input);
            return;
        },
        else => {},
    }

    var row_offset: u64 = 0;
    for (output) |*value| {
        value.* = dotQ6Row(bytes, row_offset, input);
        row_offset += row_bytes;
    }
}

fn matmulQ6RowsFixedBlocks(
    comptime cols: usize,
    output: []f32,
    bytes: []const u8,
    row_bytes: usize,
    input: []const f32,
) void {
    std.debug.assert(input.len == cols);

    var row_idx: usize = 0;
    while (row_idx + 3 < output.len) : (row_idx += 4) {
        const row0_start = row_idx * row_bytes;
        const row1_start = row0_start + row_bytes;
        const row2_start = row1_start + row_bytes;
        const row3_start = row2_start + row_bytes;
        const row0_payload = bytes[row0_start + 4 .. row0_start + row_bytes];
        const row1_payload = bytes[row1_start + 4 .. row1_start + row_bytes];
        const row2_payload = bytes[row2_start + 4 .. row2_start + row_bytes];
        const row3_payload = bytes[row3_start + 4 .. row3_start + row_bytes];
        const row0_scale: f32 = @bitCast(std.mem.readInt(u32, bytes[row0_start .. row0_start + 4][0..4], .little));
        const row1_scale: f32 = @bitCast(std.mem.readInt(u32, bytes[row1_start .. row1_start + 4][0..4], .little));
        const row2_scale: f32 = @bitCast(std.mem.readInt(u32, bytes[row2_start .. row2_start + 4][0..4], .little));
        const row3_scale: f32 = @bitCast(std.mem.readInt(u32, bytes[row3_start .. row3_start + 4][0..4], .little));

        var row0_acc0: @Vector(8, f32) = @splat(0.0);
        var row0_acc1: @Vector(8, f32) = @splat(0.0);
        var row1_acc0: @Vector(8, f32) = @splat(0.0);
        var row1_acc1: @Vector(8, f32) = @splat(0.0);
        var row2_acc0: @Vector(8, f32) = @splat(0.0);
        var row2_acc1: @Vector(8, f32) = @splat(0.0);
        var row3_acc0: @Vector(8, f32) = @splat(0.0);
        var row3_acc1: @Vector(8, f32) = @splat(0.0);
        var index: usize = 0;
        var payload_index: usize = 0;
        while (index < cols) : ({
            index += 16;
            payload_index += 12;
        }) {
            const rhs0: @Vector(8, f32) = input[index..][0..8].*;
            const rhs1: @Vector(8, f32) = input[index + 8 ..][0..8].*;

            row0_acc0 += loadQ6Vector8(row0_payload, payload_index) * rhs0;
            row0_acc1 += loadQ6Vector8(row0_payload, payload_index + 6) * rhs1;
            row1_acc0 += loadQ6Vector8(row1_payload, payload_index) * rhs0;
            row1_acc1 += loadQ6Vector8(row1_payload, payload_index + 6) * rhs1;
            row2_acc0 += loadQ6Vector8(row2_payload, payload_index) * rhs0;
            row2_acc1 += loadQ6Vector8(row2_payload, payload_index + 6) * rhs1;
            row3_acc0 += loadQ6Vector8(row3_payload, payload_index) * rhs0;
            row3_acc1 += loadQ6Vector8(row3_payload, payload_index + 6) * rhs1;
        }

        output[row_idx] = @reduce(.Add, row0_acc0 + row0_acc1) * row0_scale;
        output[row_idx + 1] = @reduce(.Add, row1_acc0 + row1_acc1) * row1_scale;
        output[row_idx + 2] = @reduce(.Add, row2_acc0 + row2_acc1) * row2_scale;
        output[row_idx + 3] = @reduce(.Add, row3_acc0 + row3_acc1) * row3_scale;
    }

    while (row_idx + 1 < output.len) : (row_idx += 2) {
        const row0_start = row_idx * row_bytes;
        const row1_start = row0_start + row_bytes;
        const row0_payload = bytes[row0_start + 4 .. row0_start + row_bytes];
        const row1_payload = bytes[row1_start + 4 .. row1_start + row_bytes];
        const row0_scale: f32 = @bitCast(std.mem.readInt(u32, bytes[row0_start .. row0_start + 4][0..4], .little));
        const row1_scale: f32 = @bitCast(std.mem.readInt(u32, bytes[row1_start .. row1_start + 4][0..4], .little));

        var row0_acc0: @Vector(8, f32) = @splat(0.0);
        var row0_acc1: @Vector(8, f32) = @splat(0.0);
        var row1_acc0: @Vector(8, f32) = @splat(0.0);
        var row1_acc1: @Vector(8, f32) = @splat(0.0);
        var index: usize = 0;
        var payload_index: usize = 0;
        while (index < cols) : ({
            index += 16;
            payload_index += 12;
        }) {
            const rhs0: @Vector(8, f32) = input[index..][0..8].*;
            const rhs1: @Vector(8, f32) = input[index + 8 ..][0..8].*;

            row0_acc0 += loadQ6Vector8(row0_payload, payload_index) * rhs0;
            row0_acc1 += loadQ6Vector8(row0_payload, payload_index + 6) * rhs1;
            row1_acc0 += loadQ6Vector8(row1_payload, payload_index) * rhs0;
            row1_acc1 += loadQ6Vector8(row1_payload, payload_index + 6) * rhs1;
        }

        output[row_idx] = @reduce(.Add, row0_acc0 + row0_acc1) * row0_scale;
        output[row_idx + 1] = @reduce(.Add, row1_acc0 + row1_acc1) * row1_scale;
    }

    if (row_idx < output.len) {
        output[row_idx] = dotQ6RowFixed(cols, bytes, @as(u64, row_idx * row_bytes), input);
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

    switch (kernel_registry.resolve(.{ .gemv_row = .{ .op = .q8_row, .cols = input.len } }).shape) {
        .qwen3_hidden_1024 => {
            matmulQ8RowsFixedBlocks(tensor_store.handwritten_hidden_width, output, bytes, row_bytes, input);
            return;
        },
        .qwen3_intermediate_3072 => {
            matmulQ8RowsFixedBlocks(tensor_store.handwritten_intermediate_width, output, bytes, row_bytes, input);
            return;
        },
        else => {},
    }

    var row_offset: u64 = 0;
    for (output) |*value| {
        value.* = dotQ8Row(bytes, row_offset, input);
        row_offset += row_bytes;
    }
}

fn matmulQ8RowsFixedBlocks(
    comptime cols: usize,
    output: []f32,
    bytes: []const u8,
    row_bytes: usize,
    input: []const f32,
) void {
    std.debug.assert(input.len == cols);

    var row_idx: usize = 0;
    while (row_idx + 3 < output.len) : (row_idx += 4) {
        const row0_start = row_idx * row_bytes;
        const row1_start = row0_start + row_bytes;
        const row2_start = row1_start + row_bytes;
        const row3_start = row2_start + row_bytes;
        const row0_payload = bytes[row0_start + 4 .. row0_start + row_bytes];
        const row1_payload = bytes[row1_start + 4 .. row1_start + row_bytes];
        const row2_payload = bytes[row2_start + 4 .. row2_start + row_bytes];
        const row3_payload = bytes[row3_start + 4 .. row3_start + row_bytes];
        const row0_scale: f32 = @bitCast(std.mem.readInt(u32, bytes[row0_start .. row0_start + 4][0..4], .little));
        const row1_scale: f32 = @bitCast(std.mem.readInt(u32, bytes[row1_start .. row1_start + 4][0..4], .little));
        const row2_scale: f32 = @bitCast(std.mem.readInt(u32, bytes[row2_start .. row2_start + 4][0..4], .little));
        const row3_scale: f32 = @bitCast(std.mem.readInt(u32, bytes[row3_start .. row3_start + 4][0..4], .little));

        var row0_acc0: @Vector(16, f32) = @splat(0.0);
        var row0_acc1: @Vector(16, f32) = @splat(0.0);
        var row1_acc0: @Vector(16, f32) = @splat(0.0);
        var row1_acc1: @Vector(16, f32) = @splat(0.0);
        var row2_acc0: @Vector(16, f32) = @splat(0.0);
        var row2_acc1: @Vector(16, f32) = @splat(0.0);
        var row3_acc0: @Vector(16, f32) = @splat(0.0);
        var row3_acc1: @Vector(16, f32) = @splat(0.0);
        var index: usize = 0;
        while (index < cols) : (index += 32) {
            const rhs0: @Vector(16, f32) = input[index..][0..16].*;
            const rhs1: @Vector(16, f32) = input[index + 16 ..][0..16].*;

            row0_acc0 += loadQ8Vector16(row0_payload, index) * rhs0;
            row0_acc1 += loadQ8Vector16(row0_payload, index + 16) * rhs1;
            row1_acc0 += loadQ8Vector16(row1_payload, index) * rhs0;
            row1_acc1 += loadQ8Vector16(row1_payload, index + 16) * rhs1;
            row2_acc0 += loadQ8Vector16(row2_payload, index) * rhs0;
            row2_acc1 += loadQ8Vector16(row2_payload, index + 16) * rhs1;
            row3_acc0 += loadQ8Vector16(row3_payload, index) * rhs0;
            row3_acc1 += loadQ8Vector16(row3_payload, index + 16) * rhs1;
        }

        output[row_idx] = @reduce(.Add, row0_acc0 + row0_acc1) * row0_scale;
        output[row_idx + 1] = @reduce(.Add, row1_acc0 + row1_acc1) * row1_scale;
        output[row_idx + 2] = @reduce(.Add, row2_acc0 + row2_acc1) * row2_scale;
        output[row_idx + 3] = @reduce(.Add, row3_acc0 + row3_acc1) * row3_scale;
    }

    while (row_idx < output.len) : (row_idx += 1) {
        output[row_idx] = dotQ8RowFixed(cols, bytes, @as(u64, row_idx * row_bytes), input);
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

        try testing.expectApproxEqAbs(dotQ8RowGeneric(row_q8, 0, input), dotQ8Row(row_q8, 0, input), 1e-6);
        try testing.expectApproxEqAbs(dotQ6RowGeneric(row_q6, 0, input), dotQ6Row(row_q6, 0, input), 1e-6);
        try testing.expectApproxEqAbs(dotQ4RowGeneric(row_q4, 0, input), dotQ4Row(row_q4, 0, input), 1e-6);
    }
}

test "wide q6 matmul rows match row-dot path" {
    const testing = std.testing;

    inline for (.{ tensor_store.handwritten_hidden_width, tensor_store.handwritten_intermediate_width }) |cols| {
        const rows: usize = 5;
        const row_bytes = 4 + (try std.math.divCeil(usize, cols * 6, 8));
        const matrix = try testing.allocator.alloc(u8, rows * row_bytes);
        defer testing.allocator.free(matrix);
        const values = try testing.allocator.alloc(f32, cols);
        defer testing.allocator.free(values);
        const input = try testing.allocator.alloc(f32, cols);
        defer testing.allocator.free(input);
        const output = try testing.allocator.alloc(f32, rows);
        defer testing.allocator.free(output);

        for (input, 0..) |*value, idx| {
            value.* = @as(f32, @floatFromInt(@as(i32, @intCast((idx * 5 + 11) % 37)) - 18)) / 8.0;
        }
        for (0..rows) |row_idx| {
            for (values, 0..) |*value, col_idx| {
                value.* = @as(f32, @floatFromInt(@as(i32, @intCast((row_idx * 19 + col_idx * 7 + 3) % 41)) - 20)) / 9.0;
            }
            encodeQ6Row(matrix[row_idx * row_bytes .. (row_idx + 1) * row_bytes], values);
        }

        matmulQ6Rows(output, matrix, row_bytes, input);

        for (0..rows) |row_idx| {
            try testing.expectApproxEqAbs(
                dotQ6Row(matrix, @as(u64, row_idx * row_bytes), input),
                output[row_idx],
                1e-6,
            );
        }
    }
}

test "wide q8 matmul rows match row-dot path" {
    const testing = std.testing;

    inline for (.{ tensor_store.handwritten_hidden_width, tensor_store.handwritten_intermediate_width }) |cols| {
        const rows: usize = 5;
        const row_bytes = 4 + cols;
        const matrix = try testing.allocator.alloc(u8, rows * row_bytes);
        defer testing.allocator.free(matrix);
        const values = try testing.allocator.alloc(f32, cols);
        defer testing.allocator.free(values);
        const input = try testing.allocator.alloc(f32, cols);
        defer testing.allocator.free(input);
        const output = try testing.allocator.alloc(f32, rows);
        defer testing.allocator.free(output);

        for (input, 0..) |*value, idx| {
            value.* = @as(f32, @floatFromInt(@as(i32, @intCast((idx * 7 + 13) % 37)) - 18)) / 8.0;
        }
        for (0..rows) |row_idx| {
            for (values, 0..) |*value, col_idx| {
                value.* = @as(f32, @floatFromInt(@as(i32, @intCast((row_idx * 17 + col_idx * 11 + 5) % 41)) - 20)) / 9.0;
            }
            encodeQ8Row(matrix[row_idx * row_bytes .. (row_idx + 1) * row_bytes], values);
        }

        matmulQ8Rows(output, matrix, row_bytes, input);

        for (0..rows) |row_idx| {
            try testing.expectApproxEqAbs(
                dotQ8Row(matrix, @as(u64, row_idx * row_bytes), input),
                output[row_idx],
                1e-6,
            );
        }
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
