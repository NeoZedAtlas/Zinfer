const basic = @import("basic.zig");
const q8_cache = @import("q8_cache.zig");
const rope = @import("rope.zig");

pub const RoPETable = rope.RoPETable;

pub const applyRoPEToHeadInPlace = rope.applyRoPEToHeadInPlace;
pub const applyRoPEToHeadWithTableInPlace = rope.applyRoPEToHeadWithTableInPlace;
pub const applyRoPEToHeadsInPlace = rope.applyRoPEToHeadsInPlace;
pub const applyRoPEToHeadsWithTableInPlace = rope.applyRoPEToHeadsWithTableInPlace;

pub const softmaxInPlace = basic.softmaxInPlace;
pub const scaledDotProductAttentionSingleQuery = basic.scaledDotProductAttentionSingleQuery;
pub const scaledDotProductAttentionSingleQueryBf16Cache = basic.scaledDotProductAttentionSingleQueryBf16Cache;

pub const q8_cache_group_size = q8_cache.q8_cache_group_size;
pub const scaledDotProductAttentionSingleQueryQ8Cache = q8_cache.scaledDotProductAttentionSingleQueryQ8Cache;
pub const scaledDotProductAttentionSingleQueryQ8CacheHeadMajor = q8_cache.scaledDotProductAttentionSingleQueryQ8CacheHeadMajor;
pub const dotQ8GroupedSlice = q8_cache.dotQ8GroupedSlice;
pub const axpyQ8GroupedSliceInPlace = q8_cache.axpyQ8GroupedSliceInPlace;
