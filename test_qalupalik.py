import sys
import torch
from transformers import GPT2LMHeadModel

sys.path.insert(0, "src")
from encoder import get_encoder


MODEL = "models/124M-pytorch"

print("Loading Qalupalik...")

# Load the original GPT-2 tokenizer locally
tokenizer = get_encoder("124M", "models")

# Load the converted Qalupalik model
model = GPT2LMHeadModel.from_pretrained(MODEL)

model.eval()

prompt = "The adventurer entered the ancient dungeon and"

# Tokenize locally
input_ids = tokenizer.encode(prompt)
inputs = {
    "input_ids": torch.tensor([input_ids], dtype=torch.long)
}

# Generate
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
        pad_token_id=50256,
    )

# Decode locally
text = tokenizer.decode(output[0].tolist())

print("\nQalupalik says:\n")
print(text)
