const std = @import("std");
const Model = @import("model/gpt_neox.zig").Model;

pub fn main() !void {
    // Allocator
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer std.debug.assert(gpa.deinit() == .ok);
    const allocator = gpa.allocator();
    // Command line arguments
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);
    if (args.len < 2) {
        std.debug.print("Usage: {s} <MODEL_DIR_PATH>\n", .{args[0]});
        std.os.exit(1);
    }
    const model_dir_path = args[1];
    // Initialize model
    const model = try Model.init(allocator, .{
        // See: https://huggingface.co/rinna/japanese-gpt-neox-small/blob/f33d445/config.json
        .num_attention_heads = 12,
        .rotary_pct = 1.0,
        .use_parallel_residual = false,
        .hidden_act = "gelu",
    }, model_dir_path);
    defer model.deinit();
    // Estimate the required memory for each token in advance using a set of initial arbitrary tokens
    // TODO: Choose a better way
    const default_context_mem_size = 256 * 1024 * 1024;
    const context_mem_size_one = model.forward(default_context_mem_size, &[_]i32{0});
    const context_mem_size_two = model.forward(default_context_mem_size, &[_]i32{ 0, 0 });
    // Inference
    const tokens = [_]i32{ 14041, 7, 1967, 12, 741, 699, 31 }; // "こんにちは、猫は好きですか？". TODO: Use tokenizer
    const context_mem_size = context_mem_size_one + tokens.len * (context_mem_size_two - context_mem_size_one);
    _ = model.forward(context_mem_size, &tokens);
}
