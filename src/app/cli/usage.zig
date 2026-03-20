const std = @import("std");

pub fn printUsage() !void {
    const stdout = std.fs.File.stdout().deprecatedWriter();
    try stdout.writeAll(
        \\Usage:
        \\  zinfer
        \\  zinfer inspect-config [model_dir]
        \\  zinfer inspect-weights [model_dir]
        \\  zinfer inspect-tensor <tensor_name>
        \\  zinfer inspect-tensor [model_dir] <tensor_name> [count]
        \\  zinfer probe-linear <tensor_name> [input_index] [rows_to_print]
        \\  zinfer probe-linear [model_dir] <tensor_name> <input_index> <rows_to_print>
        \\  zinfer probe-block [layer_index] [input_index] [count]
        \\  zinfer probe-block [model_dir] <layer_index> <input_index> <count>
        \\  zinfer probe-model [token_id] [top_k]
        \\  zinfer probe-model [model_dir] <token_id> <top_k>
        \\  zinfer generate-token-ids [seed_ids_csv] [steps]
        \\  zinfer generate-token-ids [model_dir] <seed_ids_csv> <steps>
        \\  zinfer bench <text> [max_new_tokens]
        \\  zinfer bench [model_dir] <text> <max_new_tokens>
        \\  zinfer quantize <q8|q6|q4>
        \\  zinfer quantize <q8|q6|q4> [model_dir]
        \\  zinfer tokenize <text>
        \\  zinfer tokenize [model_dir] <text>
        \\  zinfer decode-ids <ids_csv>
        \\  zinfer decode-ids [model_dir] <ids_csv>
        \\  zinfer generate <text> [max_new_tokens] [think|no-think] [flags...]
        \\  zinfer generate [model_dir] <text> <max_new_tokens> [think|no-think] [flags...]
        \\  zinfer generate-chat <messages_json_path> [max_new_tokens] [think|no-think] [flags...]
        \\  zinfer generate-chat [model_dir] <messages_json_path> <max_new_tokens> [think|no-think] [flags...]
        \\  zinfer chat [max_new_tokens] [think|no-think] [flags...]
        \\  zinfer chat [model_dir] [max_new_tokens] [think|no-think] [flags...]
        \\
        \\Defaults:
        \\  model_dir = models/Qwen3-0.6B
        \\  generate max_new_tokens = 64
        \\  generate-chat/chat max_new_tokens = 128
        \\
        \\Flags:
        \\  --system <text>
        \\  --seed <u64>
        \\  --temperature <f32>
        \\  --top-p <f32>
        \\  --top-k <usize>
        \\  --min-p <f32>
        \\  --presence-penalty <f32>
        \\  --frequency-penalty <f32>
        \\  --repetition-penalty <f32>
        \\  --stop <text>           (repeatable)
        \\  --backend <auto|bf16|q8|q6|q4>
        \\  --kv-cache <auto|bf16|q8>
        \\  --threads <usize>       (0 = auto)
        \\  --stream
        \\  --load <path>           (chat only)
        \\  --save <path>           (chat only)
        \\
    );
}
