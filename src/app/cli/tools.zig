const bench = @import("tools/bench.zig");
const model = @import("tools/model.zig");
const text = @import("tools/text.zig");

pub const benchPrompt = bench.benchPrompt;
pub const benchSuite = bench.benchSuite;
pub const benchHandwrittenOps = bench.benchHandwrittenOps;

pub const quantizeModelDir = model.quantizeModelDir;

pub const tokenizeText = text.tokenizeText;
pub const decodeIds = text.decodeIds;
