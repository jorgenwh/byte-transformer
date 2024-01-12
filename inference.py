import argparse
import time
import torch

from mlp import MLP
from rnn import RNN
from lstm import LSTM
from transformer import Transformer
from helpers import read_text, preprocess_text, AverageMeter


parser = argparse.ArgumentParser()
parser.add_argument("-t", type=int, help="Number of new tokens to generate", default=750)
parser.add_argument("-m", type=str, help="Model name", default=None)
args = parser.parse_args()

text = "mental_health.csv"

data, vocab_size, char_to_index, index_to_char = preprocess_text(read_text("data/" + text))


MAX_NEW_TOKENS = args.t
EMBD_DIM = 32
SEQ_LEN = 256
D_MODEL = 512
N_HEADS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = Transformer(vocab_size=vocab_size, seq_len=SEQ_LEN, d_model=D_MODEL, n_heads=N_HEADS, device=DEVICE)
if args.m is not None:
    model.load_state_dict(torch.load(args.m))
model = model.to(DEVICE)

prompt = torch.zeros(1, SEQ_LEN, dtype=torch.long)
str_prompt = input("PROMPT: ")
for i, char in enumerate(str_prompt):
    prompt[0, SEQ_LEN - len(str_prompt) + i] = char_to_index[char]
prompt = prompt.to(DEVICE)

model.eval()
with torch.no_grad():
    for i in range(MAX_NEW_TOKENS):
        logits = model(prompt)

        # Sample token from top 5 most likely tokens
        v, _ = torch.topk(logits, 5)
        logits[logits < v[:, [-1]]] = -float("inf")
        logits = torch.softmax(logits, dim=-1)
        new_token = torch.multinomial(logits, num_samples=1)

        str_token = index_to_char[new_token.item()]

        prompt = prompt[:, 1:]
        new_token = new_token.reshape(1, 1)
        prompt = torch.cat((prompt, new_token), dim=1)

        print(str_token, end="", flush=True)

print()
