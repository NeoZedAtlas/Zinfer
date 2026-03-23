const runtime_mod = @import("optimized_decoder/runtime.zig");
const workspace_mod = @import("optimized_decoder/workspace.zig");

pub const Runtime = runtime_mod.Runtime;
pub const Workspace = workspace_mod.Workspace;
