const ops = @import("bench/ops.zig");
const runtime = @import("bench/runtime.zig");

pub const benchPrompt = runtime.benchPrompt;
pub const benchSuite = runtime.benchSuite;
pub const benchHandwrittenOps = ops.benchHandwrittenOps;
