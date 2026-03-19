const std = @import("std");

pub fn apply(
    allocator: std.mem.Allocator,
    merges: std.StringHashMapUnmanaged(u32),
    encoded: []const u8,
) ![]const []const u8 {
    var pieces = std.ArrayListUnmanaged([]const u8).empty;
    defer pieces.deinit(allocator);

    var view = try std.unicode.Utf8View.init(encoded);
    var it = view.iterator();
    while (it.nextCodepointSlice()) |cp_slice| {
        try pieces.append(allocator, cp_slice);
    }

    while (pieces.items.len >= 2) {
        var best_rank: ?u32 = null;
        var best_index: usize = 0;

        for (0..pieces.items.len - 1) |i| {
            const rank = mergeRank(allocator, merges, pieces.items[i], pieces.items[i + 1]) orelse continue;
            if (best_rank == null or rank < best_rank.?) {
                best_rank = rank;
                best_index = i;
            }
        }

        if (best_rank == null) break;

        const merged = try std.mem.concat(allocator, u8, &.{ pieces.items[best_index], pieces.items[best_index + 1] });
        pieces.items[best_index] = merged;
        _ = pieces.orderedRemove(best_index + 1);
    }

    return pieces.toOwnedSlice(allocator);
}

fn mergeRank(
    allocator: std.mem.Allocator,
    merges: std.StringHashMapUnmanaged(u32),
    left: []const u8,
    right: []const u8,
) ?u32 {
    const key = std.fmt.allocPrint(allocator, "{s} {s}", .{ left, right }) catch return null;
    defer allocator.free(key);
    return merges.get(key);
}
