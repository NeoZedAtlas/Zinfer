const std = @import("std");

pub fn parseTokenIdsAlloc(
    allocator: std.mem.Allocator,
    csv: []const u8,
) ![]usize {
    var list: std.ArrayListUnmanaged(usize) = .empty;
    defer list.deinit(allocator);

    var it = std.mem.splitScalar(u8, csv, ',');
    while (it.next()) |part| {
        const trimmed = std.mem.trim(u8, part, " ");
        if (trimmed.len == 0) continue;
        try list.append(allocator, try std.fmt.parseInt(usize, trimmed, 10));
    }

    return list.toOwnedSlice(allocator);
}
