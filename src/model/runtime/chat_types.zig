pub const ThinkingMode = enum {
    enabled,
    disabled,
};

pub const Role = enum {
    system,
    user,
    assistant,
    tool,

    pub fn name(self: Role) []const u8 {
        return switch (self) {
            .system => "system",
            .user => "user",
            .assistant => "assistant",
            .tool => "tool",
        };
    }
};

pub const ToolCall = struct {
    name: []const u8,
    arguments_json: []const u8,
};

pub const Message = struct {
    role: Role,
    content: []const u8,
    tool_calls: []const ToolCall = &.{},
};
