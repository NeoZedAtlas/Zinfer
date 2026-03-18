const std = @import("std");

pub const Candidate = struct {
    token_id: usize,
    logit: f32,
};

pub const SamplingConfig = struct {
    temperature: f32 = 1.0,
    top_k: usize = 0,
    top_p: f32 = 1.0,
};

pub fn sampleToken(
    allocator: std.mem.Allocator,
    random: std.Random,
    logits: []const f32,
    cfg: SamplingConfig,
) !usize {
    if (logits.len == 0) return error.EmptyLogits;
    if (cfg.temperature <= 0.0 or cfg.top_k == 1) return argMax(logits);

    const actual_k = if (cfg.top_k == 0) logits.len else @min(cfg.top_k, logits.len);
    const candidates = try topKCandidatesAlloc(allocator, logits, actual_k);
    defer allocator.free(candidates);

    var probs = try allocator.alloc(f32, candidates.len);
    defer allocator.free(probs);

    var max_scaled = -std.math.inf(f32);
    for (candidates, 0..) |candidate, idx| {
        const scaled = candidate.logit / cfg.temperature;
        probs[idx] = scaled;
        if (scaled > max_scaled) max_scaled = scaled;
    }

    var sum: f64 = 0.0;
    for (probs) |*value| {
        value.* = @exp(value.* - max_scaled);
        sum += value.*;
    }
    if (sum <= 0.0) return candidates[0].token_id;

    const clamped_top_p = std.math.clamp(cfg.top_p, @as(f32, 0.0), @as(f32, 1.0));
    var keep_len = probs.len;
    if (clamped_top_p < 1.0) {
        var cumulative: f64 = 0.0;
        keep_len = 0;
        while (keep_len < probs.len) : (keep_len += 1) {
            cumulative += probs[keep_len] / sum;
            if (cumulative >= clamped_top_p and keep_len + 1 >= 1) {
                keep_len += 1;
                break;
            }
        }
        keep_len = @min(@max(keep_len, 1), probs.len);
    }

    var kept_sum: f64 = 0.0;
    for (probs[0..keep_len]) |value| kept_sum += value;
    if (kept_sum <= 0.0) return candidates[0].token_id;

    var threshold = random.float(f64) * kept_sum;
    for (candidates[0..keep_len], probs[0..keep_len]) |candidate, value| {
        threshold -= value;
        if (threshold <= 0.0) return candidate.token_id;
    }
    return candidates[keep_len - 1].token_id;
}

pub fn argMax(logits: []const f32) !usize {
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

fn topKCandidatesAlloc(
    allocator: std.mem.Allocator,
    logits: []const f32,
    k: usize,
) ![]Candidate {
    const actual_k = @min(k, logits.len);
    const top = try allocator.alloc(Candidate, actual_k);
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

    std.sort.block(Candidate, top, {}, struct {
        fn lessThan(_: void, lhs: Candidate, rhs: Candidate) bool {
            return lhs.logit > rhs.logit;
        }
    }.lessThan);

    return top;
}

test "sampler argmax path returns best token" {
    const testing = std.testing;

    const logits = [_]f32{ 0.2, 1.2, 0.9 };
    var prng = std.Random.DefaultPrng.init(1);
    try testing.expectEqual(@as(usize, 1), try sampleToken(testing.allocator, prng.random(), &logits, .{
        .temperature = 0.0,
        .top_k = 20,
        .top_p = 0.95,
    }));
}

test "sampler respects top_k filtering" {
    const testing = std.testing;

    const logits = [_]f32{ 4.0, 3.0, 2.0, 1.0 };
    var prng = std.Random.DefaultPrng.init(7);
    for (0..16) |_| {
        const token = try sampleToken(testing.allocator, prng.random(), &logits, .{
            .temperature = 1.0,
            .top_k = 2,
            .top_p = 1.0,
        });
        try testing.expect(token == 0 or token == 1);
    }
}
