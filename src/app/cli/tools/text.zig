const std = @import("std");
const cli_token_ids = @import("../token_ids.zig");
const decoder_family = @import("../../../model/runtime/decoder_family.zig");

pub fn tokenizeText(
    allocator: std.mem.Allocator,
    model_dir: []const u8,
    text: []const u8,
) !void {
    const config_path = try std.fs.path.join(allocator, &.{ model_dir, "config.json" });
    defer allocator.free(config_path);

    var parsed_config = try decoder_family.loadConfigFromFile(allocator, config_path);
    defer parsed_config.deinit();

    var tokenizer = try decoder_family.loadTokenizerFromModelDir(
        allocator,
        parsed_config.value.architecture,
        model_dir,
    );
    defer tokenizer.deinit();

    const ids = try tokenizer.encodeAlloc(allocator, text);
    defer allocator.free(ids);

    const stdout = std.fs.File.stdout().deprecatedWriter();
    try stdout.print("Zinfer tokenize\n", .{});
    try stdout.print("text: {s}\n", .{text});
    try stdout.print("ids: [", .{});
    for (ids, 0..) |id, idx| {
        if (idx != 0) try stdout.print(", ", .{});
        try stdout.print("{d}", .{id});
    }
    try stdout.print("]\n", .{});
}

pub fn decodeIds(
    allocator: std.mem.Allocator,
    model_dir: []const u8,
    ids_csv: []const u8,
) !void {
    const config_path = try std.fs.path.join(allocator, &.{ model_dir, "config.json" });
    defer allocator.free(config_path);

    var parsed_config = try decoder_family.loadConfigFromFile(allocator, config_path);
    defer parsed_config.deinit();

    var tokenizer = try decoder_family.loadTokenizerFromModelDir(
        allocator,
        parsed_config.value.architecture,
        model_dir,
    );
    defer tokenizer.deinit();

    const ids_usize = try cli_token_ids.parseTokenIdsAlloc(allocator, ids_csv);
    defer allocator.free(ids_usize);
    const ids = try allocator.alloc(u32, ids_usize.len);
    defer allocator.free(ids);
    for (ids_usize, 0..) |value, idx| {
        ids[idx] = std.math.cast(u32, value) orelse return error.TokenIdOutOfRange;
    }

    const text = try tokenizer.decodeAlloc(allocator, ids);
    defer allocator.free(text);

    const stdout = std.fs.File.stdout().deprecatedWriter();
    try stdout.print("Zinfer decode\n", .{});
    try stdout.print("ids: {s}\n", .{ids_csv});
    try stdout.print("text: {s}\n", .{text});
}
