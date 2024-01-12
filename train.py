import time
import torch
from collections import deque

from transformer import ByteTransformer
from helpers import read_text, text_to_ascii_tensor, get_time_stamp, AverageMeter


text = "tinyshakespeare.txt"


data = text_to_ascii_tensor(read_text("data/" + text))
assert data.dtype == torch.int64


BATCH_SIZE = 64
LEARNING_RATE = 0.001
START_EPOCH = 0
EPOCHS = 100
BLOCK_SIZE = 256
D_MODEL = 256
N_HEADS = 4
N_BLOCKS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_batch():
    ix = torch.randint(0, data.size(0) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix])
    return x, y


model = ByteTransformer(
        block_size=BLOCK_SIZE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_blocks=N_BLOCKS,
        device=DEVICE
)

if START_EPOCH > 0:
    model.load_state_dict(torch.load("models/model_epoch" + str(START_EPOCH) + ".pth"))
model = model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = torch.nn.CrossEntropyLoss()

iters_per_epoch = 1000
min_loss = float("inf")

for epoch in range(START_EPOCH, EPOCHS):
    # adjust learning rate
    lr = LEARNING_RATE*(0.1 ** (epoch//5))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    print("Epoch: " + str(epoch + 1) + "/" + str(EPOCHS))

    iloss = AverageMeter()
    elapsed_s = 0
    elapsed_s_last_100 = deque(maxlen=100)

    model.train()
    for i in range(iters_per_epoch):
        t0 = time.time()
        x, y = get_batch()
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        # forward pass
        output = model(x)

        B, T, D = output.shape
        output = output.view(B*T, D)
        y = y.view(B*T)

        # compute loss
        loss = loss_fn(output, y)
        iloss.update(loss.item(), x.size(0))

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        elapsed_s += time.time() - t0
        elapsed_s_last_100.append(time.time() - t0)
        remaining_iters = iters_per_epoch - i - 1
        s_per_iter_last_100 = sum(elapsed_s_last_100) / len(elapsed_s_last_100)
        remaining_s = remaining_iters * s_per_iter_last_100
        str_loss = str(iloss)
        if len(str_loss.split(".")[1]) < 4:
            str_loss += "0" * (4 - len(str_loss.split(".")[1]))

        print(
            "batch: " + str(i + 1) + "/" + str(iters_per_epoch) + " | " +
            "loss: " + str_loss + " | " +
            "elapsed: " + get_time_stamp(elapsed_s) + " | remaining: " + get_time_stamp(remaining_s) + 
            " "*10, end="\r"
        )

    remaining_s = 0
    print(
        "batch: " + str(iters_per_epoch) + "/" + str(iters_per_epoch) + " | " +
        "loss: " + str_loss + " | " + 
        "elapsed: " + get_time_stamp(elapsed_s) + " | remaining: " + get_time_stamp(remaining_s) + " "*10
    )

    if iloss.avg < min_loss:
        min_loss = iloss.avg
        torch.save(model.state_dict(), "models/model_epoch" + str(epoch + 1) + ".pth")
        print("saved models/model_epoch" + str(epoch + 1) + ".pth")

