import json
import os
import struct
import sys
from pathlib import Path

import numpy as np
from transformers import AutoModelForCausalLM

if len(sys.argv) < 3:
    print(f"Usage: python3 {sys.argv[0]} <MODEL_TYPE> <MODEL_NAME>")
    exit(1)
_, model_type, model_name = sys.argv[:3]

model_dir_path = str(Path(__file__).parent / model_type / model_name.replace("/", "_"))
os.makedirs(model_dir_path, exist_ok=True)
model = AutoModelForCausalLM.from_pretrained(model_name)
with open(f"{model_dir_path}/param.bin", "wb") as param_file, \
        open(f"{model_dir_path}/arch.json", "w") as arch_file:
    arch = {}
    for name, param in model.named_parameters():
        flattened_param = param.data.float().numpy().astype(np.float32).flatten()
        param_file.write(struct.pack(f"{len(flattened_param)}f", *flattened_param))
        arch[name] = list(param.data.shape)
    json.dump(arch, arch_file, indent=4)
