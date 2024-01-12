import argparse
import time
import torch

from transformer import ByteTransformer
from helpers import text_to_ascii_tensor


parser = argparse.ArgumentParser()
parser.add_argument("-t", type=int, help="Number of new tokens to generate", default=750)
parser.add_argument("-m", type=str, help="Model name", default=None)
args = parser.parse_args()


MAX_NEW_TOKENS = args.t
BLOCK_SIZE = 256
D_MODEL = 256
N_HEADS = 4
N_BLOCKS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = ByteTransformer(block_size=BLOCK_SIZE, d_model=D_MODEL, n_heads=N_HEADS, n_blocks=N_BLOCKS, device=DEVICE)
if args.m is not None:
    model.load_state_dict(torch.load(args.m))
model = model.to(DEVICE)

prompt = torch.zeros(1, BLOCK_SIZE, dtype=torch.long)
str_prompt = input("PROMPT: ")
print(str_prompt, end="", flush=True)
for i, char in enumerate(str_prompt):
    prompt[0, BLOCK_SIZE - len(str_prompt) + i] = ord(char) if ord(char) < 256 else 256
prompt = prompt.to(DEVICE)

model.eval()
with torch.no_grad():
    for i in range(MAX_NEW_TOKENS):
        logits = model(prompt)
        logits = logits[:, -1, :]

        # Sample token from top 5 most likely tokens
        v, _ = torch.topk(logits, 5)
        logits[logits < v[:, [-1]]] = -float("inf")
        logits = torch.softmax(logits, dim=-1)
        new_token = torch.multinomial(logits, num_samples=1)

        str_token = chr(new_token.item())

        prompt = prompt[:, 1:]
        new_token = new_token.reshape(1, 1)
        prompt = torch.cat((prompt, new_token), dim=1)

        print(str_token, end="", flush=True)

print()
