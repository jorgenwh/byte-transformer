import torch
import torch.nn as nn
import torch.nn.functional as F


from pytorch_model_summary import summary


class Attention(nn.Module):
    def __init__(self, d_model, seq_len, n_heads, device):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        self.c_attn = nn.Linear(d_model, 3*d_model)
        self.c_proj = nn.Linear(d_model, d_model)
        self.mask = torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len).to(device)

    def forward(self, x):
        B, L, D = x.shape # (BATCH, SEQ_LEN, D_MODEL)

        q, k, v = self.c_attn(x).split(self.d_model, dim=2)
        q = k.reshape(B, L, self.n_heads, D // self.n_heads).transpose(1, 2)
        k = k.reshape(B, L, self.n_heads, D // self.n_heads).transpose(1, 2)
        v = k.reshape(B, L, self.n_heads, D // self.n_heads).transpose(1, 2)

        att = torch.matmul(q, k.transpose(2, 3))
        att = att / (k.shape[-1] ** 0.5)
        att = att.masked_fill(self.mask[:, :, :L, :L] == 0, -1e9)
        att = F.softmax(att, dim=3)
        att = torch.matmul(att, v)

        att = att.transpose(1, 2).reshape(B, L, D)

        att = self.c_proj(att)
        return att


class FeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.fc1 = nn.Linear(d_model, 4*d_model)
        self.fc2 = nn.Linear(4*d_model, d_model)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class Block(nn.Module):
    def __init__(self, d_model, seq_len, n_heads, device):
        super().__init__()
        self.d_model = d_model

        self.ln1 = nn.LayerNorm(d_model)
        self.attn = Attention(d_model=d_model, seq_len=seq_len, n_heads=n_heads, device=device)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model=d_model)

    def forward(self, x):
        residual = x
        x = self.ln1(x)
        x = residual + self.attn(x)

        residual = x
        x = self.ln2(x)
        x = residual + self.ff(x)
        return x


class Transformer(nn.Module):
    def __init__(self, vocab_size, seq_len, d_model, n_heads, n_blocks, device='cpu'):
        super().__init__()
        self.device = device

        self.input_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = nn.Embedding(seq_len, d_model)

        self.blocks = nn.Sequential(
            *[Block(d_model=d_model, seq_len=seq_len, n_heads=n_heads, device=device) for _ in range(n_blocks)]
        )

        self.ln = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        pos = torch.arange(0, L, dtype=torch.long, device=x.device).reshape(1, -1).to(self.device)
        pos_emb = self.positional_encoding(pos)
        tok_emb = self.input_embedding(x)

        x = tok_emb + pos_emb

        x = self.blocks(x)

        x = self.ln(x)
        x = self.fc(x)

        x = x[:, -1, :]
        return x


if __name__ == '__main__':
    BATCH_SIZE = 1
    SEQ_LEN = 256
    D_MODEL = 512
    N_HEADS = 4
    x = torch.randint(0, 111, size=(BATCH_SIZE, SEQ_LEN))
    model = Transformer(vocab_size=111, seq_len=SEQ_LEN, d_model=D_MODEL, n_heads=N_HEADS)
    #model(x)
    print(summary(model, x))
