import json
import struct
import sys
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

if len(sys.argv) < 3:
    print(f"Usage: python3 {sys.argv[0]} <MODEL_NAME> <MODEL_DIR_PATH>")
    exit(1)
_, model_name, model_dir_path = sys.argv[:3]

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_name)

with open(f"{model_dir_path}/param.bin", "wb") as param_file, \
        open(f"{model_dir_path}/arch.json", "w") as arch_file:
    arch = {}
    for name, param in model.named_parameters():
        flattened_param = param.data.float().numpy().astype(np.float32).flatten()
        param_file.write(struct.pack(f"{len(flattened_param)}f", *flattened_param))
        arch[name] = list(param.data.shape)
    json.dump(arch, arch_file, indent=4)
