const std = @import("std");
const ggml = @cImport({
    @cInclude("ggml/ggml.h");
});
const Allocator = std.mem.Allocator;

// See: https://github.com/huggingface/transformers/blob/v4.32.0/src/transformers/models/gpt_neox/modeling_gpt_neox.py#L692
pub const GptNeox = struct {
    const Self = @This();
    const arch_file_name = "arch.json";
    const param_file_name = "param.bin";
    allocator: Allocator,
    param: Param,

    pub fn init(allocator: Allocator, model_dir_path: []const u8) !Self {
        // Read arch
        const arch_file_path = try std.fmt.allocPrint(
            allocator,
            "{s}/{s}",
            .{ model_dir_path, arch_file_name },
        );
        defer allocator.free(arch_file_path);
        const arch = try Arch.init(allocator, arch_file_path);
        // Read param
        const param_file_path = try std.fmt.allocPrint(
            allocator,
            "{s}/{s}",
            .{ model_dir_path, param_file_name },
        );
        defer allocator.free(param_file_path);
        const param = try Param.init(allocator, arch, param_file_path);
        return Self{ .allocator = allocator, .param = param };
    }

    pub fn deinit(self: Self) void {
        self.param.deinit();
    }

    // See: https://github.com/huggingface/transformers/blob/v4.32.0/src/transformers/models/gpt_neox/modeling_gpt_neox.py#L712
    pub fn forward(self: Self, context_mem_size: usize, tokens: []const i32) usize {
        // Init context
        const context = ggml.ggml_init(.{
            .mem_size = context_mem_size,
            .mem_buffer = null,
            .no_alloc = false,
        }).?;
        defer ggml.ggml_free(context);
        // Input tokens
        const input_ids = ggml.ggml_new_tensor_1d(context, ggml.GGML_TYPE_I32, @intCast(tokens.len));
        copyData(i32, tokens, input_ids);
        // Embedding
        const inputs_embeds = self.param.embed_in.forward(context, input_ids);
        const gf = ggml.ggml_new_graph(context);
        ggml.ggml_build_forward_expand(gf, inputs_embeds);
        ggml.ggml_graph_compute_with_ctx(context, gf, 4); // TODO: Use n_threads
        return ggml.ggml_used_mem(context);
    }
};

const Arch = struct {
    const Self = @This();
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    mem_size: usize,

    fn init(allocator: Allocator, arch_file_path: []const u8) !Self {
        const buffer = try std.fs.cwd().readFileAlloc(allocator, arch_file_path, 1024 * 1024);
        defer allocator.free(buffer);
        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, buffer, .{});
        defer parsed.deinit();
        var mem_size: usize = 0;
        for (parsed.value.object.values()) |layer| {
            var layer_size: usize = 1;
            for (layer.array.items) |dim| {
                layer_size *= @intCast(dim.integer);
            }
            mem_size += layer_size * ggml.ggml_type_size(ggml.GGML_TYPE_F32) + ggml.ggml_tensor_overhead();
        }
        const embed_in_weight_size = parsed.value.object.get("gpt_neox.embed_in.weight").?.array;
        const first_mlp_dense_weight_size = parsed.value.object.get("gpt_neox.layers.0.mlp.dense_h_to_4h.weight").?.array;
        var layers_count: usize = 0;
        while (true) {
            const layer_name = try std.fmt.allocPrint(
                allocator,
                "gpt_neox.layers.{d}.input_layernorm.weight",
                .{layers_count},
            );
            defer allocator.free(layer_name);
            if (parsed.value.object.get(layer_name)) |_| {
                layers_count += 1;
            } else {
                break;
            }
        }
        return Self{
            .vocab_size = @intCast(embed_in_weight_size.items[0].integer),
            .hidden_size = @intCast(embed_in_weight_size.items[1].integer),
            .intermediate_size = @intCast(first_mlp_dense_weight_size.items[0].integer),
            .num_hidden_layers = layers_count,
            .mem_size = mem_size,
        };
    }
};

const Param = struct {
    const Self = @This();
    allocator: Allocator,
    context: *ggml.struct_ggml_context,
    embed_in: EmbedIn,
    layers: []GptNeoxLayer,
    final_layer_norm: FinalLayerNorm,
    embed_out: EmbedOut,

    fn init(allocator: Allocator, arch: Arch, param_file_path: []const u8) !Self {
        // Init param context
        const context = ggml.ggml_init(.{
            .mem_size = arch.mem_size,
            .mem_buffer = null,
            .no_alloc = false,
        }).?;
        const param_file = try std.fs.cwd().openFile(param_file_path, .{});
        defer param_file.close();
        // Read EmbedIn
        const embed_in = EmbedIn.init(context, arch);
        _ = try readParam(param_file, embed_in.weight);
        // Read hidden GptNeoxLayer(s)
        const layers = try allocator.alloc(GptNeoxLayer, arch.num_hidden_layers);
        for (layers) |*layer| {
            layer.* = GptNeoxLayer.init(context, arch);
            _ = try readParam(param_file, layer.*.input_layernorm_weight);
            _ = try readParam(param_file, layer.*.input_layernorm_bias);
            _ = try readParam(param_file, layer.*.post_attention_layernorm_weight);
            _ = try readParam(param_file, layer.*.post_attention_layernorm_bias);
            _ = try readParam(param_file, layer.*.attention_query_key_value_weight);
            _ = try readParam(param_file, layer.*.attention_query_key_value_bias);
            _ = try readParam(param_file, layer.*.attention_dense_weight);
            _ = try readParam(param_file, layer.*.attention_dense_bias);
            _ = try readParam(param_file, layer.*.mlp_dense_h_to_4h_weight);
            _ = try readParam(param_file, layer.*.mlp_dense_h_to_4h_bias);
            _ = try readParam(param_file, layer.*.mlp_dense_4h_to_h_weight);
            _ = try readParam(param_file, layer.*.mlp_dense_4h_to_h_bias);
        }
        // Read FinalLayerNorm
        const final_layer_norm = FinalLayerNorm.init(context, arch);
        _ = try readParam(param_file, final_layer_norm.weight);
        _ = try readParam(param_file, final_layer_norm.bias);
        // Read EmbedOut
        const embed_out = EmbedOut.init(context, arch);
        _ = try readParam(param_file, embed_out.weight);
        return Self{
            .allocator = allocator,
            .context = context,
            .embed_in = embed_in,
            .layers = layers,
            .final_layer_norm = final_layer_norm,
            .embed_out = embed_out,
        };
    }

    fn deinit(self: Self) void {
        ggml.ggml_free(self.context);
        self.allocator.free(self.layers);
    }
};

// See: https://pytorch.org/docs/2.0/generated/torch.nn.Embedding.html
const EmbedIn = struct {
    const Self = @This();
    weight: [*c]ggml.struct_ggml_tensor,

    fn init(context: *ggml.struct_ggml_context, arch: Arch) Self {
        const weight = ggml.ggml_new_tensor_2d(
            context,
            ggml.GGML_TYPE_F32,
            @intCast(arch.hidden_size),
            @intCast(arch.vocab_size),
        );
        return Self{ .weight = weight };
    }

    fn forward(self: Self, context: *ggml.struct_ggml_context, input: [*c]ggml.struct_ggml_tensor) [*c]ggml.struct_ggml_tensor {
        return ggml.ggml_get_rows(context, self.weight, input);
    }
};

// See: https://github.com/huggingface/transformers/blob/v4.32.0/src/transformers/models/gpt_neox/modeling_gpt_neox.py#L399
const GptNeoxLayer = struct {
    const Self = @This();
    input_layernorm_weight: [*c]ggml.struct_ggml_tensor,
    input_layernorm_bias: [*c]ggml.struct_ggml_tensor,
    post_attention_layernorm_weight: [*c]ggml.struct_ggml_tensor,
    post_attention_layernorm_bias: [*c]ggml.struct_ggml_tensor,
    attention_query_key_value_weight: [*c]ggml.struct_ggml_tensor,
    attention_query_key_value_bias: [*c]ggml.struct_ggml_tensor,
    attention_dense_weight: [*c]ggml.struct_ggml_tensor,
    attention_dense_bias: [*c]ggml.struct_ggml_tensor,
    mlp_dense_h_to_4h_weight: [*c]ggml.struct_ggml_tensor,
    mlp_dense_h_to_4h_bias: [*c]ggml.struct_ggml_tensor,
    mlp_dense_4h_to_h_weight: [*c]ggml.struct_ggml_tensor,
    mlp_dense_4h_to_h_bias: [*c]ggml.struct_ggml_tensor,

    fn init(context: *ggml.struct_ggml_context, arch: Arch) Self {
        const hidden_size: i64 = @intCast(arch.hidden_size);
        const intermediate_size: i64 = @intCast(arch.intermediate_size);
        const input_layernorm_weight = ggml.ggml_new_tensor_1d(context, ggml.GGML_TYPE_F32, hidden_size);
        const input_layernorm_bias = ggml.ggml_new_tensor_1d(context, ggml.GGML_TYPE_F32, hidden_size);
        const post_attention_layernorm_weight = ggml.ggml_new_tensor_1d(context, ggml.GGML_TYPE_F32, hidden_size);
        const post_attention_layernorm_bias = ggml.ggml_new_tensor_1d(context, ggml.GGML_TYPE_F32, hidden_size);
        const attention_query_key_value_weight = ggml.ggml_new_tensor_2d(
            context,
            ggml.GGML_TYPE_F32,
            hidden_size,
            3 * hidden_size,
        );
        const attention_query_key_value_bias = ggml.ggml_new_tensor_1d(
            context,
            ggml.GGML_TYPE_F32,
            3 * hidden_size,
        );
        const attention_dense_weight = ggml.ggml_new_tensor_2d(
            context,
            ggml.GGML_TYPE_F32,
            hidden_size,
            hidden_size,
        );
        const attention_dense_bias = ggml.ggml_new_tensor_1d(
            context,
            ggml.GGML_TYPE_F32,
            hidden_size,
        );
        const mlp_dense_h_to_4h_weight = ggml.ggml_new_tensor_2d(
            context,
            ggml.GGML_TYPE_F32,
            hidden_size,
            intermediate_size,
        );
        const mlp_dense_h_to_4h_bias = ggml.ggml_new_tensor_1d(
            context,
            ggml.GGML_TYPE_F32,
            intermediate_size,
        );
        const mlp_dense_4h_to_h_weight = ggml.ggml_new_tensor_2d(
            context,
            ggml.GGML_TYPE_F32,
            intermediate_size,
            hidden_size,
        );
        const mlp_dense_4h_to_h_bias = ggml.ggml_new_tensor_1d(
            context,
            ggml.GGML_TYPE_F32,
            hidden_size,
        );
        return Self{
            .input_layernorm_weight = input_layernorm_weight,
            .input_layernorm_bias = input_layernorm_bias,
            .post_attention_layernorm_weight = post_attention_layernorm_weight,
            .post_attention_layernorm_bias = post_attention_layernorm_bias,
            .attention_query_key_value_weight = attention_query_key_value_weight,
            .attention_query_key_value_bias = attention_query_key_value_bias,
            .attention_dense_weight = attention_dense_weight,
            .attention_dense_bias = attention_dense_bias,
            .mlp_dense_h_to_4h_weight = mlp_dense_h_to_4h_weight,
            .mlp_dense_h_to_4h_bias = mlp_dense_h_to_4h_bias,
            .mlp_dense_4h_to_h_weight = mlp_dense_4h_to_h_weight,
            .mlp_dense_4h_to_h_bias = mlp_dense_4h_to_h_bias,
        };
    }
};

// See: https://pytorch.org/docs/2.0/generated/torch.nn.LayerNorm.html
const FinalLayerNorm = struct {
    const Self = @This();
    weight: [*c]ggml.struct_ggml_tensor,
    bias: [*c]ggml.struct_ggml_tensor,

    fn init(context: *ggml.struct_ggml_context, arch: Arch) Self {
        const hidden_size: i64 = @intCast(arch.hidden_size);
        const weight = ggml.ggml_new_tensor_1d(context, ggml.GGML_TYPE_F32, hidden_size);
        const bias = ggml.ggml_new_tensor_1d(context, ggml.GGML_TYPE_F32, hidden_size);
        return Self{
            .weight = weight,
            .bias = bias,
        };
    }
};

// See: https://pytorch.org/docs/2.0/generated/torch.nn.Linear.html
const EmbedOut = struct {
    const Self = @This();
    weight: [*c]ggml.struct_ggml_tensor,

    fn init(context: *ggml.struct_ggml_context, arch: Arch) Self {
        const weight = ggml.ggml_new_tensor_2d(
            context,
            ggml.GGML_TYPE_F32,
            @intCast(arch.hidden_size),
            @intCast(arch.vocab_size),
        );
        return Self{ .weight = weight };
    }
};

fn readParam(param_file: std.fs.File, tensor: [*c]ggml.struct_ggml_tensor) !usize {
    return try param_file.reader().read(getData(u8, tensor));
}

fn copyData(comptime T: type, data: []const T, tensor: [*c]ggml.struct_ggml_tensor) void {
    std.mem.copy(T, getData(T, tensor), data);
}

fn getData(comptime T: type, tensor: [*c]ggml.struct_ggml_tensor) []T {
    // Copied from: https://github.com/sjinzh/ggml-zig/blob/c6e32cb/tests/test2.zig#L36
    const data_pointer = @as([*]T, @ptrCast(@alignCast(tensor.*.data)));
    return data_pointer[0 .. ggml.ggml_nbytes(tensor) / @sizeOf(T)];
}
