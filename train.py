import time
import torch
from collections import deque

from mlp import MLP
from rnn import RNN
from lstm import LSTM
from transformer import Transformer
from helpers import read_text, preprocess_text, get_time_stamp, AverageMeter


text = "mental_health.csv"


data, vocab_size, char_to_index, index_to_char = preprocess_text(read_text("data/" + text))
assert data.dtype == torch.int64


BATCH_SIZE = 128
LEARNING_RATE = 0.001
START_EPOCH = 0
EPOCHS = 100
EMBD_DIM = 32
SEQ_LEN = 256
D_MODEL = 512
N_HEADS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_batch():
    X = torch.zeros((BATCH_SIZE, SEQ_LEN), dtype=torch.int64)
    y = torch.zeros((BATCH_SIZE), dtype=torch.int64)
    indices = torch.randint(0, data.size(0) - SEQ_LEN, (BATCH_SIZE,))
    
    for i in range(BATCH_SIZE):
        start_index = indices[i]
        end_index = start_index + SEQ_LEN
        X[i, :] = data[start_index:end_index]
        y[i] = data[end_index]

    return X, y


model = Transformer(
        vocab_size=vocab_size,
        seq_len=SEQ_LEN,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        device=DEVICE
)

if START_EPOCH > 0:
    model.load_state_dict(torch.load("models/model_epoch" + str(START_EPOCH) + ".pth"))
model = model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = torch.nn.CrossEntropyLoss()

iters = 3000
min_loss = float("inf")

for epoch in range(START_EPOCH, EPOCHS):
    # adjust learning rate
    lr = LEARNING_RATE * (0.1 ** (epoch // 5))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    print("Epoch: " + str(epoch+1) + "/" + str(EPOCHS))

    iloss = AverageMeter()
    elapsed_s = 0
    elapsed_s_last_100 = deque(maxlen=100)

    model.train()
    for i in range(iters):
        t0 = time.time()
        X, y = get_batch()
        X = X.to(DEVICE)
        y = y.to(DEVICE)

        # forward pass
        output = model(X)

        # compute loss
        loss = loss_fn(output, y)
        iloss.update(loss.item(), X.size(0))

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        elapsed_s += time.time() - t0
        elapsed_s_last_100.append(time.time() - t0)
        remaining_iters = iters - i - 1
        s_per_iter_last_100 = sum(elapsed_s_last_100) / len(elapsed_s_last_100)
        remaining_s = remaining_iters * s_per_iter_last_100
        str_loss = str(iloss)
        if len(str_loss.split(".")[1]) < 4:
            str_loss += "0" * (4 - len(str_loss.split(".")[1]))

        print(
            "batch: " + str(i + 1) + "/" + str(iters) + " | " +
            "loss: " + str_loss + " | " +
            "elapsed: " + get_time_stamp(elapsed_s) + " | remaining: " + get_time_stamp(remaining_s) + 
            " "*10, end="\r"
        )

    remaining_s = 0
    print(
        "batch: " + str(iters) + "/" + str(iters) + " | " +
        "loss: " + str_loss + " | " + 
        "elapsed: " + get_time_stamp(elapsed_s) + " | remaining: " + get_time_stamp(remaining_s) + " "*10
    )

    if iloss.avg < min_loss:
        min_loss = iloss.avg
        torch.save(model.state_dict(), "models/model_epoch" + str(epoch + 1) + ".pth")
        print("saved models/model_epoch" + str(epoch + 1) + ".pth")

