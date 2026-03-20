const std = @import("std");
const chat_types = @import("chat_types.zig");
const decoder_types = @import("decoder_types.zig");
const generic_block = @import("../layers/rmsnorm_gqa_swiglu_block.zig");
const weights_layout = @import("../layers/weights_layout.zig");

pub fn Entry(comptime Tokenizer: type) type {
    return struct {
        model_type: []const u8,
        load_config_from_file: *const fn (std.mem.Allocator, []const u8) anyerror!decoder_types.ParsedConfig,
        layer_layout: generic_block.LayerLayout,
        eos_token_ids: []const u32,
        default_stop_sequences: []const []const u8,
        common_weights: weights_layout.CommonWeights,
        layer_tensor_name_alloc: *const fn (std.mem.Allocator, usize, weights_layout.LayerTensorKind) anyerror![]u8,
        load_tokenizer: *const fn (std.mem.Allocator, []const u8) anyerror!Tokenizer,
        render_messages_prompt_alloc: *const fn (std.mem.Allocator, []const chat_types.Message, chat_types.ThinkingMode) anyerror![]u8,
        render_single_user_prompt_alloc: *const fn (std.mem.Allocator, []const u8, chat_types.ThinkingMode) anyerror![]u8,
        assistant_history_content: *const fn ([]const u8) []const u8,
    };
}
