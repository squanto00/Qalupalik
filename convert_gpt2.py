import os

import numpy as np
import tensorflow as tf
import torch
from transformers import GPT2Config, GPT2LMHeadModel


CHECKPOINT = "models/124M/model.ckpt"
OUTPUT_DIR = "models/124M-pytorch"


def load(name):
    return tf.train.load_variable(CHECKPOINT, name)


def tensor(name):
    return torch.from_numpy(np.asarray(load(name)))


def squeeze_tf_weight(name):
    value = tensor(name)

    # OpenAI GPT-2 stores Conv1D weights as [1, in, out].
    if value.ndim == 3 and value.shape[0] == 1:
        value = value.squeeze(0)

    return value


print("Creating GPT-2 configuration...")

config = GPT2Config(
    vocab_size=50257,
    n_positions=1024,
    n_ctx=1024,
    n_embd=768,
    n_layer=12,
    n_head=12,
)

model = GPT2LMHeadModel(config)

state = model.state_dict()

print("Loading TensorFlow checkpoint...")


# Token + positional embeddings
state["transformer.wte.weight"] = tensor("model/wte")
state["transformer.wpe.weight"] = tensor("model/wpe")


# Transformer blocks
for i in range(12):
    prefix = f"model/h{i}"

    # LayerNorm 1
    state[f"transformer.h.{i}.ln_1.weight"] = tensor(
        f"{prefix}/ln_1/g"
    )
    state[f"transformer.h.{i}.ln_1.bias"] = tensor(
        f"{prefix}/ln_1/b"
    )

    # Attention QKV
    state[f"transformer.h.{i}.attn.c_attn.weight"] = squeeze_tf_weight(
        f"{prefix}/attn/c_attn/w"
    )
    state[f"transformer.h.{i}.attn.c_attn.bias"] = tensor(
        f"{prefix}/attn/c_attn/b"
    )

    # Attention output projection
    state[f"transformer.h.{i}.attn.c_proj.weight"] = squeeze_tf_weight(
        f"{prefix}/attn/c_proj/w"
    )
    state[f"transformer.h.{i}.attn.c_proj.bias"] = tensor(
        f"{prefix}/attn/c_proj/b"
    )

    # LayerNorm 2
    state[f"transformer.h.{i}.ln_2.weight"] = tensor(
        f"{prefix}/ln_2/g"
    )
    state[f"transformer.h.{i}.ln_2.bias"] = tensor(
        f"{prefix}/ln_2/b"
    )

    # MLP input projection
    state[f"transformer.h.{i}.mlp.c_fc.weight"] = squeeze_tf_weight(
        f"{prefix}/mlp/c_fc/w"
    )
    state[f"transformer.h.{i}.mlp.c_fc.bias"] = tensor(
        f"{prefix}/mlp/c_fc/b"
    )

    # MLP output projection
    state[f"transformer.h.{i}.mlp.c_proj.weight"] = squeeze_tf_weight(
        f"{prefix}/mlp/c_proj/w"
    )
    state[f"transformer.h.{i}.mlp.c_proj.bias"] = tensor(
        f"{prefix}/mlp/c_proj/b"
    )


# Final LayerNorm
state["transformer.ln_f.weight"] = tensor("model/ln_f/g")
state["transformer.ln_f.bias"] = tensor("model/ln_f/b")


print("Checking tensor shapes...")

for name, value in state.items():
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} is not a tensor")

model.load_state_dict(state, strict=True)

print("Checkpoint loaded successfully.")


# GPT-2 ties the LM head weights to the token embeddings.
model.tie_weights()

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Saving converted model to {OUTPUT_DIR}...")

model.save_pretrained(OUTPUT_DIR, safe_serialization=True)

print("Conversion complete!")
