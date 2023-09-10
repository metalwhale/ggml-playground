const std = @import("std");
pub const ggml = @cImport({
    @cInclude("ggml/ggml.h");
});
pub const Context = *ggml.struct_ggml_context;
pub const Tensor = [*c]ggml.struct_ggml_tensor;

// See: https://pytorch.org/docs/2.0/generated/torch.nn.Embedding.html
pub fn embedding(context: Context, input: Tensor, weight: Tensor) Tensor {
    return ggml.ggml_get_rows(context, weight, input);
}

// See: https://pytorch.org/docs/2.0/generated/torch.nn.LayerNorm.html
pub fn layernorm(context: Context, input: Tensor, weight: Tensor, bias: Tensor) Tensor {
    var cur: Tensor = undefined;
    cur = ggml.ggml_norm(context, input);
    cur = ggml.ggml_mul(context, cur, ggml.ggml_repeat(context, weight, cur));
    const output = ggml.ggml_add(context, cur, ggml.ggml_repeat(context, bias, cur));
    return output;
}

// See: https://pytorch.org/docs/2.0/generated/torch.nn.Linear.html
pub fn linear(context: Context, input: Tensor, weight: Tensor, bias: ?Tensor) Tensor {
    var cur: Tensor = undefined;
    cur = ggml.ggml_mul_mat(context, weight, input);
    if (bias) |b| {
        cur = ggml.ggml_add(context, cur, ggml.ggml_repeat(context, b, cur));
    }
    return cur;
}

// See:
// - https://github.com/huggingface/transformers/blob/v4.32.0/src/transformers/models/gpt_neox/modeling_gpt_neox.py#L236
// - https://github.com/huggingface/transformers/blob/v4.32.0/src/transformers/models/gpt_neox/modeling_gpt_neox.py#L225
// Inputs: each tensor has ne = (head_size, num_attention_heads, sequence_length, 1)
pub fn attention(context: Context, query: Tensor, key: Tensor, value: Tensor, num_past: c_int) Tensor {
    const head_size = query.*.ne[0];
    const num_attention_heads = query.*.ne[1];
    const sequence_length = query.*.ne[2];
    const norm_factor = @sqrt(@as(f32, @floatFromInt(head_size)));
    const attn_scores = ggml.ggml_diag_mask_inf_inplace(
        context,
        ggml.ggml_scale_inplace(context, ggml.ggml_mul_mat(
            context,
            ggml.ggml_permute(context, key, 0, 2, 1, 3), // Choose key as row
            ggml.ggml_permute(context, query, 0, 2, 1, 3), // Choose query as column
        ), ggml.ggml_new_f32(context, 1 / norm_factor)),
        num_past,
    ); // ne = (sequence_length, sequence_length, num_attention_heads, 1)
    const attn_weights = ggml.ggml_soft_max_inplace(context, attn_scores);
    const attn_output = ggml.ggml_cont(context, ggml.ggml_permute(context, ggml.ggml_mul_mat(
        context,
        ggml.ggml_cont(context, ggml.ggml_permute(context, value, 1, 2, 0, 3)),
        attn_weights,
    ), 0, 2, 1, 3)); // ne = (head_size, num_attention_heads, sequence_length, 1)
    const attn_output_merged = ggml.ggml_cont(context, ggml.ggml_view_2d(
        context,
        attn_output,
        head_size * num_attention_heads,
        sequence_length,
        @as(usize, @intCast(head_size * num_attention_heads)) * ggml.ggml_type_size(ggml.GGML_TYPE_F32),
        0,
    )); // ne = (hidden_size, sequence_length, 1, 1)
    return attn_output_merged;
}

pub fn readParam(param_file: std.fs.File, tensor: Tensor) !usize {
    return try param_file.reader().read(getData(u8, tensor));
}

pub fn copyData(comptime T: type, data: []const T, tensor: Tensor) void {
    std.mem.copy(T, getData(T, tensor), data);
}

pub fn getData(comptime T: type, tensor: Tensor) []T {
    // Copied from: https://github.com/sjinzh/ggml-zig/blob/c6e32cb/tests/test2.zig#L36
    const data_pointer = @as([*]T, @ptrCast(@alignCast(tensor.*.data)));
    return data_pointer[0 .. ggml.ggml_nbytes(tensor) / @sizeOf(T)];
}
