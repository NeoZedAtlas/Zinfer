const std = @import("std");
const GenerateOptions = @import("args.zig").GenerateOptions;
const cli_messages = @import("messages.zig");
const cli_prompts = @import("prompts.zig");
const cli_runtime = @import("runtime.zig");

pub fn generateText(
    allocator: std.mem.Allocator,
    model_dir: []const u8,
    user_text: []const u8,
    options: GenerateOptions,
) !void {
    var runtime = try cli_runtime.GeneratorRuntime.init(allocator, model_dir, options.backend_scheme, options.thread_count);
    defer runtime.deinit();
    const stdout = std.fs.File.stdout().deprecatedWriter();

    try stdout.print("Zinfer generate\n", .{});
    try stdout.print("mode: {s}\n", .{cli_prompts.thinkingModeName(options.thinking_mode)});
    try stdout.print("prompt: {s}\n", .{user_text});
    try stdout.writeAll("response: ");

    const prompt = try cli_prompts.buildSingleUserPromptAlloc(
        allocator,
        runtime.model.cfg.architecture,
        user_text,
        options.system_prompt,
        options.thinking_mode,
    );
    defer allocator.free(prompt);

    const response = try runtime.generateFromPrompt(prompt, options);
    defer allocator.free(response);
    if (!options.stream_output) {
        try stdout.print("{s}", .{response});
    }
    try stdout.writeAll("\n");
}

pub fn generateChatFromFile(
    allocator: std.mem.Allocator,
    model_dir: []const u8,
    messages_json_path: []const u8,
    options: GenerateOptions,
) !void {
    var runtime = try cli_runtime.GeneratorRuntime.init(allocator, model_dir, options.backend_scheme, options.thread_count);
    defer runtime.deinit();
    const stdout = std.fs.File.stdout().deprecatedWriter();

    try stdout.print("Zinfer generate-chat\n", .{});
    try stdout.print("mode: {s}\n", .{cli_prompts.thinkingModeName(options.thinking_mode)});
    try stdout.print("messages_path: {s}\n", .{messages_json_path});
    try stdout.writeAll("response: ");

    var messages = try cli_messages.loadChatMessages(allocator, messages_json_path);
    defer messages.deinit();

    const prompt = try cli_prompts.buildMessagesPromptAlloc(
        allocator,
        runtime.model.cfg.architecture,
        messages.items,
        options.system_prompt,
        options.thinking_mode,
    );
    defer allocator.free(prompt);

    const response = try runtime.generateFromPrompt(prompt, options);
    defer allocator.free(response);
    if (!options.stream_output) {
        try stdout.print("{s}", .{response});
    }
    try stdout.writeAll("\n");
}
