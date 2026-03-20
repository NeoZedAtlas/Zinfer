const std = @import("std");
const cli_args = @import("cli/args.zig");
const cli_chat = @import("cli/chat.zig");
const cli_generate = @import("cli/generate.zig");
const cli_inspect = @import("cli/inspect.zig");
const cli_tools = @import("cli/tools.zig");
const cli_usage = @import("cli/usage.zig");

const default_model_dir = cli_args.default_model_dir;

pub fn run(allocator: std.mem.Allocator) !void {
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len == 1) {
        try cli_inspect.inspectConfig(allocator, default_model_dir);
        return;
    }

    const command = args[1];
    if (std.mem.eql(u8, command, "inspect-config")) {
        const model_dir = if (args.len >= 3) args[2] else default_model_dir;
        try cli_inspect.inspectConfig(allocator, model_dir);
        return;
    }

    if (std.mem.eql(u8, command, "inspect-weights")) {
        const model_dir = if (args.len >= 3) args[2] else default_model_dir;
        try cli_inspect.inspectWeights(allocator, model_dir);
        return;
    }

    if (std.mem.eql(u8, command, "inspect-tensor")) {
        if (args.len == 3) {
            try cli_inspect.inspectTensor(allocator, default_model_dir, args[2], 8);
            return;
        }
        if (args.len >= 4) {
            const count = if (args.len >= 5) try std.fmt.parseInt(usize, args[4], 10) else 8;
            try cli_inspect.inspectTensor(allocator, args[2], args[3], count);
            return;
        }
        try printUsage();
        return error.InvalidCommand;
    }

    if (std.mem.eql(u8, command, "probe-linear")) {
        if (args.len == 3) {
            try cli_inspect.probeLinear(allocator, default_model_dir, args[2], 0, 8);
            return;
        }
        if (args.len == 4) {
            const input_index = try std.fmt.parseInt(usize, args[3], 10);
            try cli_inspect.probeLinear(allocator, default_model_dir, args[2], input_index, 8);
            return;
        }
        if (args.len == 5) {
            const input_index = try std.fmt.parseInt(usize, args[3], 10);
            const rows_to_print = try std.fmt.parseInt(usize, args[4], 10);
            try cli_inspect.probeLinear(allocator, default_model_dir, args[2], input_index, rows_to_print);
            return;
        }
        if (args.len >= 6) {
            const input_index = try std.fmt.parseInt(usize, args[4], 10);
            const rows_to_print = try std.fmt.parseInt(usize, args[5], 10);
            try cli_inspect.probeLinear(allocator, args[2], args[3], input_index, rows_to_print);
            return;
        }
        try printUsage();
        return error.InvalidCommand;
    }

    if (std.mem.eql(u8, command, "probe-block")) {
        if (args.len == 2) {
            try cli_inspect.probeBlock(allocator, default_model_dir, 0, 0, 8);
            return;
        }
        if (args.len == 3) {
            const layer_index = try std.fmt.parseInt(usize, args[2], 10);
            try cli_inspect.probeBlock(allocator, default_model_dir, layer_index, 0, 8);
            return;
        }
        if (args.len == 4) {
            const layer_index = try std.fmt.parseInt(usize, args[2], 10);
            const input_index = try std.fmt.parseInt(usize, args[3], 10);
            try cli_inspect.probeBlock(allocator, default_model_dir, layer_index, input_index, 8);
            return;
        }
        if (args.len == 5) {
            const layer_index = try std.fmt.parseInt(usize, args[2], 10);
            const input_index = try std.fmt.parseInt(usize, args[3], 10);
            const count = try std.fmt.parseInt(usize, args[4], 10);
            try cli_inspect.probeBlock(allocator, default_model_dir, layer_index, input_index, count);
            return;
        }
        if (args.len >= 6) {
            const layer_index = try std.fmt.parseInt(usize, args[3], 10);
            const input_index = try std.fmt.parseInt(usize, args[4], 10);
            const count = try std.fmt.parseInt(usize, args[5], 10);
            try cli_inspect.probeBlock(allocator, args[2], layer_index, input_index, count);
            return;
        }
        try printUsage();
        return error.InvalidCommand;
    }

    if (std.mem.eql(u8, command, "probe-model")) {
        if (args.len == 2) {
            try cli_inspect.probeModel(allocator, default_model_dir, 0, 8);
            return;
        }
        if (args.len == 3) {
            const token_id = try std.fmt.parseInt(usize, args[2], 10);
            try cli_inspect.probeModel(allocator, default_model_dir, token_id, 8);
            return;
        }
        if (args.len == 4) {
            const token_id = try std.fmt.parseInt(usize, args[2], 10);
            const top_k = try std.fmt.parseInt(usize, args[3], 10);
            try cli_inspect.probeModel(allocator, default_model_dir, token_id, top_k);
            return;
        }
        if (args.len >= 5) {
            const token_id = try std.fmt.parseInt(usize, args[3], 10);
            const top_k = try std.fmt.parseInt(usize, args[4], 10);
            try cli_inspect.probeModel(allocator, args[2], token_id, top_k);
            return;
        }
        try printUsage();
        return error.InvalidCommand;
    }

    if (std.mem.eql(u8, command, "generate-token-ids")) {
        if (args.len == 2) {
            try cli_inspect.generateTokenIds(allocator, default_model_dir, "0", 5);
            return;
        }
        if (args.len == 3) {
            try cli_inspect.generateTokenIds(allocator, default_model_dir, args[2], 5);
            return;
        }
        if (args.len == 4) {
            const steps = try std.fmt.parseInt(usize, args[3], 10);
            try cli_inspect.generateTokenIds(allocator, default_model_dir, args[2], steps);
            return;
        }
        if (args.len >= 5) {
            const steps = try std.fmt.parseInt(usize, args[4], 10);
            try cli_inspect.generateTokenIds(allocator, args[2], args[3], steps);
            return;
        }
        try printUsage();
        return error.InvalidCommand;
    }

    if (std.mem.eql(u8, command, "bench")) {
        var invocation = try cli_args.parseBenchInvocation(allocator, args);
        defer invocation.deinit(allocator);
        try cli_tools.benchPrompt(allocator, invocation.model_dir, invocation.user_text, invocation.options);
        return;
    }

    if (std.mem.eql(u8, command, "bench-ops")) {
        var model_dir: []const u8 = default_model_dir;
        var iterations: usize = 0;

        if (args.len >= 3) {
            if (std.fmt.parseInt(usize, args[2], 10)) |parsed_iterations| {
                iterations = parsed_iterations;
            } else |_| {
                model_dir = args[2];
                if (args.len >= 4) {
                    iterations = try std.fmt.parseInt(usize, args[3], 10);
                }
            }
        }

        try cli_tools.benchHandwrittenOps(allocator, model_dir, iterations);
        return;
    }

    if (std.mem.eql(u8, command, "quantize")) {
        if (args.len == 3) {
            try cli_tools.quantizeModelDir(allocator, default_model_dir, args[2]);
            return;
        }
        if (args.len >= 4) {
            try cli_tools.quantizeModelDir(allocator, args[3], args[2]);
            return;
        }
        try printUsage();
        return error.InvalidCommand;
    }

    if (std.mem.eql(u8, command, "tokenize")) {
        if (args.len == 3) {
            try cli_tools.tokenizeText(allocator, default_model_dir, args[2]);
            return;
        }
        if (args.len >= 4) {
            try cli_tools.tokenizeText(allocator, args[2], args[3]);
            return;
        }
        try printUsage();
        return error.InvalidCommand;
    }

    if (std.mem.eql(u8, command, "decode-ids")) {
        if (args.len == 3) {
            try cli_tools.decodeIds(allocator, default_model_dir, args[2]);
            return;
        }
        if (args.len >= 4) {
            try cli_tools.decodeIds(allocator, args[2], args[3]);
            return;
        }
        try printUsage();
        return error.InvalidCommand;
    }

    if (std.mem.eql(u8, command, "generate")) {
        var invocation = try cli_args.parseGenerateInvocation(allocator, args);
        defer invocation.deinit(allocator);
        try cli_generate.generateText(allocator, invocation.model_dir, invocation.user_text, invocation.options);
        return;
    }

    if (std.mem.eql(u8, command, "generate-chat")) {
        var invocation = try cli_args.parseGenerateChatInvocation(allocator, args);
        defer invocation.deinit(allocator);
        try cli_generate.generateChatFromFile(allocator, invocation.model_dir, invocation.messages_json_path, invocation.options);
        return;
    }

    if (std.mem.eql(u8, command, "chat")) {
        var invocation = try cli_args.parseChatInvocation(allocator, args);
        defer invocation.deinit(allocator);
        try cli_chat.chatLoop(
            allocator,
            invocation.model_dir,
            invocation.options,
            invocation.load_path,
            invocation.save_path,
        );
        return;
    }

    if (std.mem.eql(u8, command, "--help") or std.mem.eql(u8, command, "-h")) {
        try printUsage();
        return;
    }

    std.log.err("unknown command: {s}", .{command});
    try printUsage();
    return error.InvalidCommand;
}

fn printUsage() !void {
    try cli_usage.printUsage();
}
