const std = @import("std");
const Model = @import("model/gpt_neox.zig").GptNeox;

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
    const model = try Model.init(allocator, model_dir_path);
    defer model.deinit();
    // Estimate the required memory for each token in advance using a set of initial arbitrary tokens
    // Remember: actually, the total memory required for running graph computation doesn't scale linearly with the number of tokens
    const arbitrary_tokens = [_]i32{ 0, 1, 2, 3, 4 };
    const default_context_mem_size = 256 * 1024 * 1024;
    const per_token_mem_size = model.forward(default_context_mem_size, &arbitrary_tokens) /
        arbitrary_tokens.len;
    // Inference
    const tokens = [_]i32{14041}; // "こんにちは". TODO: Use tokenizer
    const context_mem_size = @max(per_token_mem_size * tokens.len, default_context_mem_size); // TODO: Choose a better way
    _ = model.forward(context_mem_size, &tokens);
}
