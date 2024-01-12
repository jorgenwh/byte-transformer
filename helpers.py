import torch
import datetime


def read_text(filename):
    with open(filename, "r") as f:
        return f.read()

def text_to_ascii_tensor(text):
    asc = [ord(c) if ord(c) < 256 else 256 for c in text]
    data = torch.tensor(asc)
    return data

def ascii_tensor_to_text(tensor):
    return "".join([chr(t) for t in tensor])

def get_time_stamp(s):
    t_s = str(datetime.timedelta(seconds=round(s)))
    ts = t_s.split(':')
    return ts[0] + ':' + ts[1] + ':' + ts[2]


class AverageMeter():
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum/self.count

    def __repr__(self):
        return f"{round(self.avg, 4)}"

    def __str__(self):
        return repr(self)


if __name__ == "__main__":
    raw_text = read_text("data/tinyshakespeare.txt")
    data = preprocess_text(raw_text)
