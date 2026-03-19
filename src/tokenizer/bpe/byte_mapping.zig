const std = @import("std");

pub fn init(
    allocator: std.mem.Allocator,
    byte_encoder: *[256][]const u8,
    byte_decoder: *std.AutoHashMapUnmanaged(u21, u8),
) !void {
    var bs = std.ArrayListUnmanaged(u32).empty;
    defer bs.deinit(allocator);
    var cs = std.ArrayListUnmanaged(u32).empty;
    defer cs.deinit(allocator);

    var b: u32 = '!';
    while (b <= '~') : (b += 1) {
        try bs.append(allocator, b);
        try cs.append(allocator, b);
    }
    b = 0xA1;
    while (b <= 0xAC) : (b += 1) {
        try bs.append(allocator, b);
        try cs.append(allocator, b);
    }
    b = 0xAE;
    while (b <= 0xFF) : (b += 1) {
        try bs.append(allocator, b);
        try cs.append(allocator, b);
    }

    var n: u32 = 0;
    for (0..256) |byte_value| {
        if (!containsCode(bs.items, @intCast(byte_value))) {
            try bs.append(allocator, @intCast(byte_value));
            try cs.append(allocator, 256 + n);
            n += 1;
        }
    }

    for (bs.items, cs.items) |byte_value, codepoint| {
        var buffer: [4]u8 = undefined;
        const len = try std.unicode.utf8Encode(@intCast(codepoint), &buffer);
        const string = try allocator.dupe(u8, buffer[0..len]);
        byte_encoder[@intCast(byte_value)] = string;
        try byte_decoder.put(allocator, @intCast(codepoint), @intCast(byte_value));
    }
}

fn containsCode(values: []const u32, needle: u32) bool {
    for (values) |value| {
        if (value == needle) return true;
    }
    return false;
}
