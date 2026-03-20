const std = @import("std");
const parallel_rows = @import("parallel_rows.zig");
const quantized = @import("quantized.zig");
const safetensors = @import("../format/safetensors.zig");
const tensor_store = @import("store.zig");

pub const Scheme = enum {
    auto,
    bf16,
    q8,
    q4,

    pub fn name(self: Scheme) []const u8 {
        return switch (self) {
            .auto => "auto",
            .bf16 => "bf16",
            .q8 => "q8",
            .q4 => "q4",
        };
    }
};

pub const Backend = union(Scheme) {
    auto: void,
    bf16: tensor_store.TensorStore,
    q8: quantized.Store,
    q4: quantized.Store,

    pub fn openFromModelDir(allocator: std.mem.Allocator, model_dir: []const u8, scheme: Scheme) !Backend {
        const q4_path = try std.fs.path.join(allocator, &.{ model_dir, quantized.Scheme.q4.fileName() });
        defer allocator.free(q4_path);
        const q8_path = try std.fs.path.join(allocator, &.{ model_dir, quantized.Scheme.q8.fileName() });
        defer allocator.free(q8_path);
        const bf16_path = try std.fs.path.join(allocator, &.{ model_dir, "model.safetensors" });
        defer allocator.free(bf16_path);

        return switch (scheme) {
            .q4 => .{ .q4 = try quantized.Store.open(allocator, q4_path) },
            .q8 => .{ .q8 = try quantized.Store.open(allocator, q8_path) },
            .bf16 => .{ .bf16 = try tensor_store.TensorStore.open(allocator, bf16_path) },
            .auto => blk: {
                if (pathExists(q4_path)) break :blk .{ .q4 = try quantized.Store.open(allocator, q4_path) };
                if (pathExists(q8_path)) break :blk .{ .q8 = try quantized.Store.open(allocator, q8_path) };
                break :blk .{ .bf16 = try tensor_store.TensorStore.open(allocator, bf16_path) };
            },
        };
    }

    pub fn deinit(self: *Backend) void {
        switch (self.*) {
            .auto => {},
            .bf16 => |*store| store.deinit(),
            .q8 => |*store| store.deinit(),
            .q4 => |*store| store.deinit(),
        }
    }

    pub fn resolvedScheme(self: Backend) Scheme {
        return switch (self) {
            .auto => .auto,
            .bf16 => .bf16,
            .q8 => .q8,
            .q4 => .q4,
        };
    }

    pub fn artifactBytes(self: *const Backend) u64 {
        return switch (self.*) {
            .auto => 0,
            .bf16 => |*store| @intCast(store.bytes.len),
            .q8 => |*store| @intCast(store.bytes.len),
            .q4 => |*store| @intCast(store.bytes.len),
        };
    }

    pub const TensorHandle = union(enum) {
        bf16: safetensors.TensorInfo,
        q8: quantized.TensorInfo,
        q4: quantized.TensorInfo,
    };

    pub fn resolveTensor(self: *const Backend, name: []const u8) !TensorHandle {
        return switch (self.*) {
            .bf16 => |*store| .{ .bf16 = store.getTensor(name) orelse return error.TensorNotFound },
            .q8 => |*store| .{ .q8 = store.getTensor(name) orelse return error.TensorNotFound },
            .q4 => |*store| .{ .q4 = store.getTensor(name) orelse return error.TensorNotFound },
            .auto => unreachable,
        };
    }

    pub fn readVectorInto(self: *const Backend, name: []const u8, output: []f32, scratch: []u8) !void {
        switch (self.*) {
            .bf16 => |*store| try store.readElementsAsF32Into(name, 0, output, scratch),
            .q8 => |*store| try store.readElementsAsF32Into(name, 0, output),
            .q4 => |*store| try store.readElementsAsF32Into(name, 0, output),
            .auto => unreachable,
        }
    }

    pub fn readRowInto(self: *const Backend, name: []const u8, row_index: usize, output: []f32, scratch: []u8) !void {
        switch (self.*) {
            .bf16 => |*store| try store.readRowAsF32Into(name, row_index, output, scratch),
            .q8 => |*store| try store.readRowAsF32Into(name, row_index, output),
            .q4 => |*store| try store.readRowAsF32Into(name, row_index, output),
            .auto => unreachable,
        }
    }

    pub fn readRowIntoTensor(self: *const Backend, tensor: TensorHandle, row_index: usize, output: []f32, scratch: []u8) !void {
        switch (tensor) {
            .bf16 => |info| switch (self.*) {
                .bf16 => |*store| try store.readTensorRowAsF32Into(info, row_index, output),
                else => return error.BackendTensorMismatch,
            },
            .q8 => |info| switch (self.*) {
                .q8 => |*store| try store.readTensorRowAsF32Into(info, row_index, output),
                else => return error.BackendTensorMismatch,
            },
            .q4 => |info| switch (self.*) {
                .q4 => |*store| try store.readTensorRowAsF32Into(info, row_index, output),
                else => return error.BackendTensorMismatch,
            },
        }
        _ = scratch;
    }

    pub fn matmulVecByName(
        self: *const Backend,
        output: []f32,
        name: []const u8,
        input: []const f32,
        thread_count: usize,
        pool: ?*parallel_rows.Pool,
        scratch: []u8,
    ) !void {
        switch (self.*) {
            .bf16 => |*store| try store.matmulVecByNameThreaded(output, name, input, thread_count, pool, scratch),
            .q8 => |*store| try store.matmulVecByName(output, name, input, thread_count, pool),
            .q4 => |*store| try store.matmulVecByName(output, name, input, thread_count, pool),
            .auto => unreachable,
        }
    }

    pub fn matmulVec(
        self: *const Backend,
        output: []f32,
        tensor: TensorHandle,
        input: []const f32,
        thread_count: usize,
        pool: ?*parallel_rows.Pool,
        scratch: []u8,
    ) !void {
        switch (tensor) {
            .bf16 => |info| switch (self.*) {
                .bf16 => |*store| try store.matmulVecThreaded(output, info, input, thread_count, pool),
                else => return error.BackendTensorMismatch,
            },
            .q8 => |info| switch (self.*) {
                .q8 => |*store| try store.matmulVec(output, info, input, thread_count, pool),
                else => return error.BackendTensorMismatch,
            },
            .q4 => |info| switch (self.*) {
                .q4 => |*store| try store.matmulVec(output, info, input, thread_count, pool),
                else => return error.BackendTensorMismatch,
            },
        }
        _ = scratch;
    }
};

fn pathExists(path: []const u8) bool {
    const file = if (std.fs.path.isAbsolute(path))
        std.fs.openFileAbsolute(path, .{})
    else
        std.fs.cwd().openFile(path, .{});
    if (file) |handle| {
        handle.close();
        return true;
    } else |_| {
        return false;
    }
}
