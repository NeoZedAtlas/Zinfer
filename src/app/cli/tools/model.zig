const std = @import("std");
const quantized = @import("../../../tensor/formats/quantized.zig");

pub fn quantizeModelDir(
    allocator: std.mem.Allocator,
    model_dir: []const u8,
    scheme_text: []const u8,
) !void {
    const scheme: quantized.Scheme = if (std.mem.eql(u8, scheme_text, "q8"))
        .q8
    else if (std.mem.eql(u8, scheme_text, "q6"))
        .q6
    else if (std.mem.eql(u8, scheme_text, "q4"))
        .q4
    else
        return error.InvalidQuantizationScheme;

    const input_path = try std.fs.path.join(allocator, &.{ model_dir, "model.safetensors" });
    defer allocator.free(input_path);
    const output_path = try std.fs.path.join(allocator, &.{ model_dir, scheme.fileName() });
    defer allocator.free(output_path);

    const stdout = std.fs.File.stdout().deprecatedWriter();
    try stdout.print("Zinfer quantize\n", .{});
    try stdout.print("model_dir: {s}\n", .{model_dir});
    try stdout.print("scheme: {s}\n", .{scheme.name()});
    try stdout.print("output: {s}\n", .{output_path});

    var timer = try std.time.Timer.start();
    try quantized.quantizeModel(allocator, input_path, output_path, scheme);
    const elapsed_ns = timer.read();
    const output_size = try fileSizeAtPath(output_path);

    try stdout.print("elapsed_ms: {d:.3}\n", .{nsToMs(elapsed_ns)});
    try stdout.print("output_bytes: {d}\n", .{output_size});
    try stdout.print("output_mib: {d:.3}\n", .{bytesToMiB(output_size)});
}

fn fileSizeAtPath(path: []const u8) !u64 {
    const file = if (std.fs.path.isAbsolute(path))
        try std.fs.openFileAbsolute(path, .{})
    else
        try std.fs.cwd().openFile(path, .{});
    defer file.close();
    const stat = try file.stat();
    return stat.size;
}

fn nsToMs(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1_000_000.0;
}

fn bytesToMiB(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / (1024.0 * 1024.0);
}
