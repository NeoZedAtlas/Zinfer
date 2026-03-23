const cache = @import("optimized_kv_cache/cache.zig");
const quantize = @import("optimized_kv_cache/quantize.zig");
const types = @import("optimized_kv_cache/types.zig");

pub const Scheme = types.Scheme;
pub const resolveScheme = types.resolveScheme;
pub const q8_group_size = types.q8_group_size;
pub const q8_page_len = types.q8_page_len;
pub const Q8Layout = types.Q8Layout;
pub const default_q8_layout = types.default_q8_layout;

pub const LayerKVCache = cache.LayerKVCache;
pub const ModelCache = cache.ModelCache;
pub const estimateBytes = cache.estimateBytes;
pub const estimateBytesWithLayout = cache.estimateBytesWithLayout;

pub const quantizeQ8Slice = quantize.quantizeQ8Slice;
pub const scaleGroupsPerToken = quantize.scaleGroupsPerToken;
