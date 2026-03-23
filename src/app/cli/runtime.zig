const generator = @import("runtime/generator.zig");
const streaming = @import("runtime/streaming.zig");

pub const StopAnalysis = streaming.StopAnalysis;
pub const analyzeGeneratedText = streaming.analyzeGeneratedText;
pub const analyzeAndMaybeStream = streaming.analyzeAndMaybeStream;

pub const GeneratorRuntime = generator.GeneratorRuntime;
