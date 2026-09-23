import sys
import torch
from transformers import GPT2LMHeadModel

sys.path.insert(0, "src")
from encoder import get_encoder


MODEL = "models/124M-pytorch"
PERSONALITY_FILE = "personality.txt"

MAX_NEW_TOKENS = 150
MAX_CONTEXT_TOKENS = 850


print("Loading Qalupalik...")

# Load the original GPT-2 tokenizer locally
tokenizer = get_encoder("124M", "models")

# Load the converted Qalupalik model
model = GPT2LMHeadModel.from_pretrained(MODEL)
model.eval()

# Load personality
with open(PERSONALITY_FILE, "r", encoding="utf-8") as f:
    personality = f.read().strip()

print("Qalupalik is ready.")
print("Type 'quit' or 'exit' to stop.\n")


history = []


while True:
    user_input = input("You: ")

    if user_input.lower() in ("quit", "exit"):
        print("Goodbye.")
        break

    if not user_input.strip():
        continue

    history.append(f"User: {user_input}")

    # Build the conversation prompt
    conversation = personality + "\n\n"

    for message in history:
        conversation += message + "\n"

    conversation += "Qalupalik:"

    # Tokenize the entire prompt
    input_ids = tokenizer.encode(conversation)

    # Keep enough room for Qalupalik's response
    if len(input_ids) > MAX_CONTEXT_TOKENS:
        input_ids = input_ids[-MAX_CONTEXT_TOKENS:]

    inputs = {
        "input_ids": torch.tensor([input_ids], dtype=torch.long)
    }

    # Generate response
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.9,
            top_p=0.95,
            pad_token_id=50256,
        )

    # Only decode the newly generated tokens
    response_ids = output[0][len(input_ids):]
    response = tokenizer.decode(response_ids.tolist()).strip()

    # Stop if the model starts generating another speaker
    if "User:" in response:
        response = response.split("User:", 1)[0].strip()

    if "Qalupalik:" in response:
        response = response.split("Qalupalik:", 1)[0].strip()

    print(f"\nQalupalik: {response}\n")

    # Save response to conversation history
    history.append(f"Qalupalik: {response}")
