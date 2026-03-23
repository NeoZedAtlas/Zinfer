const forward = @import("gqa_attention/forward.zig");
const spec = @import("gqa_attention/spec.zig");

pub const AttentionSpec = spec.AttentionSpec;

pub const applyRoPEToProjectedHeadsInPlace = forward.applyRoPEToProjectedHeadsInPlace;
pub const applyRoPEToProjectedHeadsWithTableInPlace = forward.applyRoPEToProjectedHeadsWithTableInPlace;
pub const forwardProjectedSingleToken = forward.forwardProjectedSingleToken;
pub const forwardProjectedSingleTokenBf16Cache = forward.forwardProjectedSingleTokenBf16Cache;
pub const forwardProjectedSingleTokenQ8Cache = forward.forwardProjectedSingleTokenQ8Cache;
pub const forwardProjectedSingleTokenQ8CacheHeadMajor = forward.forwardProjectedSingleTokenQ8CacheHeadMajor;
pub const forwardProjectedSingleTokenQ8CachePagedHeadMajor = forward.forwardProjectedSingleTokenQ8CachePagedHeadMajor;
