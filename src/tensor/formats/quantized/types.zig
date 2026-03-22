const std = @import("std");

pub const Scheme = enum {
    q6,
    q8,
    q4,

    pub fn name(self: Scheme) []const u8 {
        return switch (self) {
            .q6 => "Q6_0",
            .q8 => "Q8_0",
            .q4 => "Q4_0",
        };
    }

    pub fn fileName(self: Scheme) []const u8 {
        return switch (self) {
            .q6 => "model.q6.zinfer",
            .q8 => "model.q8.zinfer",
            .q4 => "model.q4.zinfer",
        };
    }
};

pub const Encoding = enum {
    f32,
    q6_0,
    q8_0,
    q4_0,

    pub fn fromString(text: []const u8) !Encoding {
        if (std.mem.eql(u8, text, "F32")) return .f32;
        if (std.mem.eql(u8, text, "Q6_0")) return .q6_0;
        if (std.mem.eql(u8, text, "Q8_0")) return .q8_0;
        if (std.mem.eql(u8, text, "Q4_0")) return .q4_0;
        return error.UnsupportedEncoding;
    }

    pub fn name(self: Encoding) []const u8 {
        return switch (self) {
            .f32 => "F32",
            .q6_0 => "Q6_0",
            .q8_0 => "Q8_0",
            .q4_0 => "Q4_0",
        };
    }
};

pub const TensorInfo = struct {
    encoding: Encoding,
    shape: []const u64,
    row_bytes: u64,
    data_offsets: [2]u64,
    absolute_offset: u64,

    pub fn rank(self: TensorInfo) usize {
        return self.shape.len;
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
};
