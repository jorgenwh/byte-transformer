import torch
import datetime


def read_text(filename):
    with open(filename, "r") as f:
        return f.read()

def preprocess_text(text):
    vocab = sorted(list(set(text)))
    vocab_size = len(vocab)
    char_to_index = {char: index for index, char in enumerate(vocab)}
    index_to_char = {index: char for index, char in enumerate(vocab)}
    data = torch.tensor([char_to_index[char] for char in text])
    return data, vocab_size, char_to_index, index_to_char

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
    text = read_data()
    print(len(text))
    data, vocab_size, char_to_index, index_to_char = preprocess_text(text)
    print(data)
    print(data.dtype)
    print(data.shape)

    print(char_to_index)

    print(text[:10])
    print(data[:10])
