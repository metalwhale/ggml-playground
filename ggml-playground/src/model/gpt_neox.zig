const std = @import("std");
const common = @import("common.zig");
const Allocator = std.mem.Allocator;
const ggml = common.ggml;
const Context = common.Context;
const Tensor = common.Tensor;
const F32 = ggml.GGML_TYPE_F32;
const I32 = ggml.GGML_TYPE_I32;

const Config = struct {
    num_attention_heads: usize,
    rotary_pct: f32,
    use_parallel_residual: bool,
    hidden_act: []const u8,
};

const Output = struct {
    const Self = @This();
    allocator: Allocator,
    logits: [][]f32,
    used_mem: usize,

    fn init(allocator: Allocator, lm_logits: Tensor, used_mem: usize) !Self {
        const vocab_size: usize = @intCast(lm_logits.*.ne[0]);
        const sequence_length: usize = @intCast(lm_logits.*.ne[1]);
        const logits = try allocator.alloc([]f32, sequence_length);
        const logits_data = common.getData(f32, lm_logits);
        for (logits, 0..) |*logit, i| {
            logit.* = try allocator.alloc(f32, vocab_size);
            std.mem.copy(f32, logit.*, logits_data[i * vocab_size .. (i + 1) * vocab_size]);
        }
        return Self{ .allocator = allocator, .logits = logits, .used_mem = used_mem };
    }

    pub fn deinit(self: Self) void {
        for (self.logits) |logit| {
            self.allocator.free(logit);
        }
        self.allocator.free(self.logits);
    }
};

// See:
// - https://github.com/huggingface/transformers/blob/v4.32.0/src/transformers/models/gpt_neox/modeling_gpt_neox.py#L692
// - https://github.com/huggingface/transformers/blob/v4.32.0/src/transformers/models/gpt_neox/modeling_gpt_neox.py#L513
pub const Model = struct {
    const Self = @This();
    const arch_file_name = "arch.json";
    const param_file_name = "param.bin";
    allocator: Allocator,
    param: Param,

    pub fn init(allocator: Allocator, config: Config, model_dir_path: []const u8) !Self {
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
        const param = try Param.init(allocator, config, arch, param_file_path);
        return Self{ .allocator = allocator, .param = param };
    }

    pub fn deinit(self: Self) void {
        self.param.deinit();
    }

    // See:
    // - https://github.com/huggingface/transformers/blob/v4.32.0/src/transformers/models/gpt_neox/modeling_gpt_neox.py#L712
    // - https://github.com/huggingface/transformers/blob/v4.32.0/src/transformers/models/gpt_neox/modeling_gpt_neox.py#L541
    // - https://github.com/ggerganov/ggml/blob/08c57df/examples/gpt-neox/main.cpp#L429
    pub fn forward(self: Self, context_mem_size: usize, tokens: []const i32) !Output {
        // Init context
        const context = ggml.ggml_init(.{
            .mem_size = context_mem_size,
            .mem_buffer = null,
            .no_alloc = false,
        }).?;
        defer ggml.ggml_free(context);
        // Input tokens
        const input_ids = ggml.ggml_new_tensor_1d(context, I32, @intCast(tokens.len));
        common.copyData(i32, tokens, input_ids);
        // EmbedIn
        // https://github.com/huggingface/transformers/blob/v4.32.0/src/transformers/models/gpt_neox/modeling_gpt_neox.py#L623
        const inputs_embeds = self.param.embed_in.forward(context, input_ids);
        // Hidden Layer(s)
        var hidden_states = inputs_embeds;
        for (self.param.layers) |layer| {
            // https://github.com/huggingface/transformers/blob/v4.32.0/src/transformers/models/gpt_neox/modeling_gpt_neox.py#L658
            hidden_states = layer.forward(context, hidden_states);
        }
        // FinalLayerNorm
        // https://github.com/huggingface/transformers/blob/v4.32.0/src/transformers/models/gpt_neox/modeling_gpt_neox.py#L673
        hidden_states = self.param.final_layer_norm.forward(context, hidden_states);
        // EmbedOut
        const lm_logits = self.param.embed_out.forward(context, hidden_states);
        // Run computation
        const gf = ggml.ggml_new_graph(context);
        ggml.ggml_build_forward_expand(gf, lm_logits);
        ggml.ggml_graph_compute_with_ctx(context, gf, 4); // TODO: Use num_threads
        const output = try Output.init(self.allocator, lm_logits, ggml.ggml_used_mem(context));
        return output;
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
            mem_size += layer_size * ggml.ggml_type_size(F32) + ggml.ggml_tensor_overhead();
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
    context: Context,
    embed_in: EmbedIn,
    layers: []Layer,
    final_layer_norm: FinalLayerNorm,
    embed_out: EmbedOut,

    fn init(allocator: Allocator, config: Config, arch: Arch, param_file_path: []const u8) !Self {
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
        _ = try common.readParam(param_file, embed_in.weight);
        // Read hidden Layer(s)
        const layers = try allocator.alloc(Layer, arch.num_hidden_layers);
        for (layers) |*layer| {
            layer.* = Layer.init(
                context,
                arch,
                config.num_attention_heads,
                config.rotary_pct,
                config.use_parallel_residual,
                config.hidden_act,
            );
            _ = try common.readParam(param_file, layer.*.input_layernorm_weight);
            _ = try common.readParam(param_file, layer.*.input_layernorm_bias);
            _ = try common.readParam(param_file, layer.*.post_attention_layernorm_weight);
            _ = try common.readParam(param_file, layer.*.post_attention_layernorm_bias);
            _ = try common.readParam(param_file, layer.*.attention_query_key_value_weight);
            _ = try common.readParam(param_file, layer.*.attention_query_key_value_bias);
            _ = try common.readParam(param_file, layer.*.attention_dense_weight);
            _ = try common.readParam(param_file, layer.*.attention_dense_bias);
            _ = try common.readParam(param_file, layer.*.mlp_dense_h_to_4h_weight);
            _ = try common.readParam(param_file, layer.*.mlp_dense_h_to_4h_bias);
            _ = try common.readParam(param_file, layer.*.mlp_dense_4h_to_h_weight);
            _ = try common.readParam(param_file, layer.*.mlp_dense_4h_to_h_bias);
        }
        // Read FinalLayerNorm
        const final_layer_norm = FinalLayerNorm.init(context, arch);
        _ = try common.readParam(param_file, final_layer_norm.weight);
        _ = try common.readParam(param_file, final_layer_norm.bias);
        // Read EmbedOut
        const embed_out = EmbedOut.init(context, arch);
        _ = try common.readParam(param_file, embed_out.weight);
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

const EmbedIn = struct {
    const Self = @This();
    weight: Tensor,

    fn init(context: Context, arch: Arch) Self {
        const weight = ggml.ggml_new_tensor_2d(
            context,
            F32,
            @intCast(arch.hidden_size),
            @intCast(arch.vocab_size),
        );
        return Self{ .weight = weight };
    }

    fn forward(self: Self, context: Context, input: Tensor) Tensor {
        return common.embedding(context, input, self.weight);
    }
};

// See: https://github.com/huggingface/transformers/blob/v4.32.0/src/transformers/models/gpt_neox/modeling_gpt_neox.py#L399
const Layer = struct {
    const Self = @This();
    num_attention_heads: usize,
    rotary_pct: f32,
    use_parallel_residual: bool,
    hidden_act: []const u8,
    input_layernorm_weight: Tensor,
    input_layernorm_bias: Tensor,
    post_attention_layernorm_weight: Tensor,
    post_attention_layernorm_bias: Tensor,
    attention_query_key_value_weight: Tensor,
    attention_query_key_value_bias: Tensor,
    attention_dense_weight: Tensor,
    attention_dense_bias: Tensor,
    mlp_dense_h_to_4h_weight: Tensor,
    mlp_dense_h_to_4h_bias: Tensor,
    mlp_dense_4h_to_h_weight: Tensor,
    mlp_dense_4h_to_h_bias: Tensor,

    fn init(
        context: Context,
        arch: Arch,
        num_attention_heads: usize,
        rotary_pct: f32,
        use_parallel_residual: bool,
        hidden_act: []const u8,
    ) Self {
        const hidden_size: i64 = @intCast(arch.hidden_size);
        const intermediate_size: i64 = @intCast(arch.intermediate_size);
        const input_layernorm_weight = ggml.ggml_new_tensor_1d(context, F32, hidden_size);
        const input_layernorm_bias = ggml.ggml_new_tensor_1d(context, F32, hidden_size);
        const post_attention_layernorm_weight = ggml.ggml_new_tensor_1d(context, F32, hidden_size);
        const post_attention_layernorm_bias = ggml.ggml_new_tensor_1d(context, F32, hidden_size);
        const attention_query_key_value_weight = ggml.ggml_new_tensor_2d(
            context,
            F32,
            hidden_size,
            3 * hidden_size, // 3 ~ query, key, value
        );
        const attention_query_key_value_bias = ggml.ggml_new_tensor_1d(
            context,
            F32,
            3 * hidden_size, // 3 ~ query, key, value
        );
        const attention_dense_weight = ggml.ggml_new_tensor_2d(
            context,
            F32,
            hidden_size,
            hidden_size,
        );
        const attention_dense_bias = ggml.ggml_new_tensor_1d(
            context,
            F32,
            hidden_size,
        );
        const mlp_dense_h_to_4h_weight = ggml.ggml_new_tensor_2d(
            context,
            F32,
            hidden_size,
            intermediate_size,
        );
        const mlp_dense_h_to_4h_bias = ggml.ggml_new_tensor_1d(
            context,
            F32,
            intermediate_size,
        );
        const mlp_dense_4h_to_h_weight = ggml.ggml_new_tensor_2d(
            context,
            F32,
            intermediate_size,
            hidden_size,
        );
        const mlp_dense_4h_to_h_bias = ggml.ggml_new_tensor_1d(
            context,
            F32,
            hidden_size,
        );
        return Self{
            .num_attention_heads = num_attention_heads,
            .rotary_pct = rotary_pct,
            .use_parallel_residual = use_parallel_residual,
            .hidden_act = hidden_act,
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

    // See: https://github.com/huggingface/transformers/blob/v4.32.0/src/transformers/models/gpt_neox/modeling_gpt_neox.py#L410
    // Inputs: `hidden_states` has ne = (hidden_size, sequence_length, 1, 1)
    fn forward(self: Self, context: Context, hidden_states: Tensor) Tensor {
        var attn_output = self.attention(context, common.layernorm(
            context,
            hidden_states,
            self.input_layernorm_weight,
            self.input_layernorm_bias,
        )); // ne = (hidden_size, sequence_length, 1, 1)
        attn_output = ggml.ggml_add(context, attn_output, hidden_states); // ne = (hidden_size, sequence_length, 1, 1)
        // NOTE: Currently, only handle one case where `use_parallel_residual` = false
        var mlp_output: Tensor = undefined;
        if (self.use_parallel_residual) {
            unreachable;
        } else {
            mlp_output = self.mlp(
                context,
                common.layernorm(
                    context,
                    attn_output,
                    self.post_attention_layernorm_weight,
                    self.post_attention_layernorm_bias,
                ),
                self.hidden_act,
            ); // ne = (hidden_size, sequence_length, 1, 1)
        }
        // https://github.com/huggingface/transformers/blob/v4.32.0/src/transformers/models/gpt_neox/modeling_gpt_neox.py#L446
        const output = ggml.ggml_add(context, mlp_output, attn_output); // ne = (hidden_size, sequence_length, 1, 1)
        return output;
    }

    // See: https://github.com/huggingface/transformers/blob/v4.32.0/src/transformers/models/gpt_neox/modeling_gpt_neox.py#L148
    // Inputs: `hidden_states` has ne = (hidden_size, sequence_length, 1, 1)
    fn attention(self: Self, context: Context, hidden_states: Tensor) Tensor {
        const num_past = 0; // TODO: Calculate this
        const sequence_length = hidden_states.*.ne[1];
        const hidden_size: usize = @intCast(self.attention_query_key_value_weight.*.ne[0]);
        const head_size = hidden_size / self.num_attention_heads;
        const rotary_ndims: c_int = @intFromFloat(@as(f32, @floatFromInt(head_size)) * self.rotary_pct);
        // Example of `qkv.*.data` layout:
        //   Q   K   V   Q   K   V   Q   K   V   Q   K   V   Q   K   V   Q   K   V   Q   K   V   Q   K   V
        // ║---.---.---|---.---.---|---.---.---|---.---.---║---.---.---|---.---.---|---.---.---|---.---.---║
        // With:
        // sequence_length = 2, num_attention_heads = 4
        // |---|                                             = head_size = hidden_size / num_attention_heads
        // ║-----------------------------------------------║ = 3 * hidden_size (per token)
        // https://github.com/huggingface/transformers/blob/v4.32.0/src/transformers/models/gpt_neox/modeling_gpt_neox.py#L163
        const qkv = common.linear(
            context,
            hidden_states,
            self.attention_query_key_value_weight,
            self.attention_query_key_value_bias,
        ); // ne = (3 * hidden_size, sequence_length, 1, 1)
        // https://github.com/huggingface/transformers/blob/v4.32.0/src/transformers/models/gpt_neox/modeling_gpt_neox.py#L187
        const query = ggml.ggml_rope_inplace(context, ggml.ggml_cont(context, ggml.ggml_view_3d(
            context,
            qkv,
            @intCast(head_size),
            @intCast(self.num_attention_heads),
            sequence_length,
            3 * head_size * ggml.ggml_type_size(F32), // = qkv.*.nb[1] / self.num_attention_heads (size of QKV)
            3 * head_size * self.num_attention_heads * ggml.ggml_type_size(F32), // = qkv.*.nb[1]
            0 * head_size * ggml.ggml_type_size(F32),
        )), num_past, rotary_ndims, 2, 0); // ne = (head_size, num_attention_heads, sequence_length, 1)
        // https://github.com/huggingface/transformers/blob/v4.32.0/src/transformers/models/gpt_neox/modeling_gpt_neox.py#L188
        const key = ggml.ggml_rope_inplace(context, ggml.ggml_cont(context, ggml.ggml_view_3d(
            context,
            qkv,
            @intCast(head_size),
            @intCast(self.num_attention_heads),
            sequence_length,
            3 * head_size * ggml.ggml_type_size(F32), // = qkv.*.nb[1] / self.num_attention_heads (size of QKV)
            3 * head_size * self.num_attention_heads * ggml.ggml_type_size(F32), // = qkv.*.nb[1]
            1 * head_size * ggml.ggml_type_size(F32),
        )), num_past, rotary_ndims, 2, 0); // ne = (head_size, num_attention_heads, sequence_length, 1)
        // https://github.com/huggingface/transformers/blob/v4.32.0/src/transformers/models/gpt_neox/modeling_gpt_neox.py#L173
        const value = ggml.ggml_cont(context, ggml.ggml_view_3d(
            context,
            qkv,
            @intCast(head_size),
            @intCast(self.num_attention_heads),
            sequence_length,
            3 * head_size * ggml.ggml_type_size(F32), // = qkv.*.nb[1] / self.num_attention_heads (size of QKV)
            3 * head_size * self.num_attention_heads * ggml.ggml_type_size(F32), // = qkv.*.nb[1]
            2 * head_size * ggml.ggml_type_size(F32),
        )); // ne = (head_size, num_attention_heads, sequence_length, 1)
        // TODO: Store the current key/value in caches and retrieve the entire caches
        // https://github.com/huggingface/transformers/blob/v4.32.0/src/transformers/models/gpt_neox/modeling_gpt_neox.py#L203
        const attn_output = common.linear(
            context,
            common.attention(context, query, key, value, num_past),
            self.attention_dense_weight,
            self.attention_dense_bias,
        ); // ne = (hidden_size, sequence_length, 1, 1)
        return attn_output;
    }

    // See: https://github.com/huggingface/transformers/blob/v4.32.0/src/transformers/models/gpt_neox/modeling_gpt_neox.py#L392
    // Inputs: `hidden_states` has ne = (hidden_size, sequence_length, 1, 1)
    fn mlp(self: Self, context: Context, hidden_states: Tensor, hidden_act: []const u8) Tensor {
        // NOTE: Currently, only handle one case where `hidden_act` = "gelu"
        var act: *const fn (?Context, Tensor) callconv(.C) Tensor = undefined;
        if (std.mem.eql(u8, hidden_act, "gelu")) {
            act = ggml.ggml_gelu;
        } else {
            unreachable;
        }
        // https://github.com/huggingface/transformers/blob/v4.32.0/src/transformers/models/gpt_neox/modeling_gpt_neox.py#L393
        var mlp_output = common.linear(
            context,
            hidden_states,
            self.mlp_dense_h_to_4h_weight,
            self.mlp_dense_h_to_4h_bias,
        ); // ne = (intermediate_size, sequence_length, 1, 1)
        // https://github.com/huggingface/transformers/blob/v4.32.0/src/transformers/models/gpt_neox/modeling_gpt_neox.py#L395
        mlp_output = common.linear(
            context,
            act(context, mlp_output),
            self.mlp_dense_4h_to_h_weight,
            self.mlp_dense_4h_to_h_bias,
        ); // ne = (hidden_size, sequence_length, 1, 1)
        return mlp_output;
    }
};

const FinalLayerNorm = struct {
    const Self = @This();
    weight: Tensor,
    bias: Tensor,

    fn init(context: Context, arch: Arch) Self {
        const hidden_size: i64 = @intCast(arch.hidden_size);
        const weight = ggml.ggml_new_tensor_1d(context, F32, hidden_size);
        const bias = ggml.ggml_new_tensor_1d(context, F32, hidden_size);
        return Self{
            .weight = weight,
            .bias = bias,
        };
    }

    fn forward(self: Self, context: Context, input: Tensor) Tensor {
        return common.layernorm(context, input, self.weight, self.bias);
    }
};

const EmbedOut = struct {
    const Self = @This();
    weight: Tensor,

    fn init(context: Context, arch: Arch) Self {
        const weight = ggml.ggml_new_tensor_2d(
            context,
            F32,
            @intCast(arch.hidden_size),
            @intCast(arch.vocab_size),
        );
        return Self{ .weight = weight };
    }

    fn forward(self: Self, context: Context, input: Tensor) Tensor {
        return common.linear(context, input, self.weight, null);
    }
};

test "Simple forward" {
    // Allocator
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer std.debug.assert(gpa.deinit() == .ok);
    const allocator = gpa.allocator();
    const model_dir_path = "../models/gpt_neox/rinna_japanese-gpt-neox-small/";
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
    const output_one = try model.forward(default_context_mem_size, &[_]i32{0});
    output_one.deinit();
    const output_two = try model.forward(default_context_mem_size, &[_]i32{ 0, 0 });
    output_two.deinit();
    // Inference
    const tokens = [_]i32{ 14041, 7, 1967, 12, 741, 699, 31 }; // "こんにちは、猫は好きですか？". TODO: Use tokenizer
    const context_mem_size = output_one.used_mem + (tokens.len - 1) * (output_two.used_mem - output_one.used_mem);
    const output = try model.forward(context_mem_size, &tokens);
    defer output.deinit();
    // Assertions
    const expect = std.testing.expect;
    const last_logits = output.logits[output.logits.len - 1];
    const next_token = std.sort.argMax(f32, last_logits, {}, std.sort.asc(f32));
    try expect(next_token == 8);
}
