const std = @import("std");

pub const TopLogit = struct {
    token_id: usize,
    logit: f32,
};

pub fn topKLogitsAlloc(
    allocator: std.mem.Allocator,
    logits: []const f32,
    k: usize,
) ![]TopLogit {
    const actual_k = @min(k, logits.len);
    const top = try allocator.alloc(TopLogit, actual_k);
    errdefer allocator.free(top);

    for (top, 0..) |*entry, idx| {
        entry.* = .{
            .token_id = idx,
            .logit = logits[idx],
        };
    }

    var cursor = actual_k;
    while (cursor < logits.len) : (cursor += 1) {
        var min_index: usize = 0;
        for (top[1..], 1..) |entry, idx| {
            if (entry.logit < top[min_index].logit) min_index = idx;
        }
        if (logits[cursor] > top[min_index].logit) {
            top[min_index] = .{
                .token_id = cursor,
                .logit = logits[cursor],
            };
        }
    }

    std.sort.block(TopLogit, top, {}, struct {
        fn lessThan(_: void, lhs: TopLogit, rhs: TopLogit) bool {
            return lhs.logit > rhs.logit;
        }
    }.lessThan);

    return top;
}

pub fn argMaxLogit(logits: []const f32) !usize {
    if (logits.len == 0) return error.EmptyLogits;

    var best_index: usize = 0;
    var best_value = logits[0];
    for (logits[1..], 1..) |value, idx| {
        if (value > best_value) {
            best_value = value;
            best_index = idx;
        }
    }
    return best_index;
}

test "topKLogitsAlloc returns logits in descending order" {
    const testing = std.testing;

    const logits = [_]f32{ 0.5, -1.0, 2.0, 1.5 };
    const top = try topKLogitsAlloc(testing.allocator, &logits, 3);
    defer testing.allocator.free(top);

    try testing.expectEqual(@as(usize, 2), top[0].token_id);
    try testing.expectEqual(@as(usize, 3), top[1].token_id);
    try testing.expectEqual(@as(usize, 0), top[2].token_id);
}

test "argMaxLogit returns highest index" {
    const testing = std.testing;

    try testing.expectEqual(@as(usize, 2), try argMaxLogit(&[_]f32{ -1.0, 0.0, 3.0, 2.0 }));
}
