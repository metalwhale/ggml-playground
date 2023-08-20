const std = @import("std");
const ggml = @cImport({
    @cInclude("ggml/ggml.h");
});

pub fn main() !void {
    const ctx = ggml.ggml_init(.{
        .mem_size = 128 * 1024 * 1024,
        .mem_buffer = null,
        .no_alloc = false,
    });
    defer ggml.ggml_free(ctx);
    const tensor = ggml.ggml_new_tensor_1d(ctx, ggml.GGML_TYPE_F32, 10);
    std.debug.print("{}\n", .{tensor.*.n_dims});
}
