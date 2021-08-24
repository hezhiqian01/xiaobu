from torch.utils.data import Dataset, DataLoader
import tqdm
import json
import torch
import random
import numpy as np
from sklearn.utils import shuffle
from torch_implement import config

seq_max_len = config['seq_max_len']
BATCH_SIZE = config['pretrain']['batch_size']


class BERTDataSet(Dataset):

    def __init__(self, data, word2idx_path=None, word2idx=None, random=True):
        # directory of corpus dataset
        self.data = data
        # define special symbols
        # 0: pad, 1: unk, 2: cls, 3: sep, 4: mask, 5: no, 6: yes
        self.pad_index = 0
        self.unk_index = 1
        self.cls_index = 2
        self.sep_index = 3
        self.mask_index = 4
        # self.no_index = 5
        # self.yes_index = 6

        # 是否随机mask
        self.random = random

        # 加载字典
        if word2idx:
            self.word2idx = word2idx
        elif word2idx_path:
            with open(word2idx_path, "r", encoding="utf-8") as f:
                self.word2idx = json.load(f)
        else:
            raise RuntimeError("specify one of word2idx_path and word2idx")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # 一条数据的维度为(seq_len, 1)
        text1, text2, label = self.data[index]
        token_ids, segment_ids, output_ids, label = self.sample_convert(text1, text2, label)
        return token_ids, segment_ids, output_ids, label

    def random_mask(self, tokens, text_ids):
        """随机mask
        """
        input_ids, output_ids = [], []
        rands = np.random.random(len(text_ids))
        for r, i in zip(rands, text_ids):
            if r < 0.15 * 0.8:
                input_ids.append(4)
                output_ids.append(i)
            elif r < 0.15 * 0.9:
                input_ids.append(i)
                output_ids.append(i)
            elif r < 0.15:
                input_ids.append(np.random.choice(len(tokens)) + 7)
                output_ids.append(i)
            else:
                input_ids.append(i)
                output_ids.append(0)
        return input_ids, output_ids

    def sample_convert(self, text1, text2, label):
        """转换为MLM格式
        """
        tokens = self.word2idx
        text1_ids = [tokens.get(t, self.unk_index) for t in text1]
        text2_ids = [tokens.get(t, self.unk_index) for t in text2]
        if self.random:
            if np.random.random() < 0.5:
                text1_ids, text2_ids = text2_ids, text1_ids
            text1_ids, out1_ids = self.random_mask(tokens, text1_ids)
            text2_ids, out2_ids = self.random_mask(tokens, text2_ids)
        else:
            out1_ids = [self.pad_index] * len(text1_ids)
            out2_ids = [self.pad_index] * len(text2_ids)
        token_ids = [self.cls_index] + text1_ids + [self.sep_index] + text2_ids + [self.sep_index]
        segment_ids = [0] * (len(text1_ids) + 2) + [1] * (len(text2_ids) + 1)
        output_ids = [self.pad_index] + out1_ids + [self.pad_index] + out2_ids + [self.pad_index]

        return token_ids, segment_ids, output_ids, label


def collate_fn(batch_data, pad=0):
    token_ids, segment_ids, output_ids, label = list(zip(*batch_data))
    max_len = max([len(seq_a) for seq_a in token_ids])

    token_ids = [seq+[pad]*(max_len-len(seq)) for seq in token_ids]
    segment_ids = [seq+[pad]*(max_len-len(seq)) for seq in segment_ids]
    output_ids = [seq+[pad]*(max_len-len(seq)) for seq in output_ids]

    token_ids = torch.LongTensor(token_ids)
    segment_ids = torch.IntTensor(segment_ids)
    output_ids = torch.LongTensor(output_ids)
    label = torch.LongTensor(label)
    # print(output_ids.shape)
    # # 转成one-hot
    # output_ids = torch.zeros(output_ids.shape[0], output_ids.shape[1], vocab_size).scatter_(1, output_ids, 1)

    return token_ids, segment_ids, output_ids, label


def my_test():
    from torch_implement.pre_train import split_data, get_tokens
    train_data, valid_data, test_data = split_data()
    tokens = get_tokens(train_data+valid_data+test_data)
    train_ds = BERTDataSet(
        data=train_data,
        seq_len=x,
        word2idx_path=None,
        word2idx=tokens,
        random=True
    )

    dl_train = DataLoader(train_ds,
                          batch_size=BATCH_SIZE,
                          shuffle=False,
                          num_workers=4,
                          collate_fn=collate_fn
                          )

    seed = np.random.randint(0, 1000)
    print(seed)
    for i in range(2):
        np.random.seed(i+seed)
        for token_ids, segment_ids, output_ids, label in dl_train:
            # print(token_ids, segment_ids, output_ids)
            # print(token_ids)
            # print(output_ids)
            break


def analyze_data():
    from torch_implement.pre_train import split_data, get_tokens
    train_data, valid_data, test_data = split_data()
    train_data = train_data + valid_data

    # 统计正样本负样本数量
    print("train count:{}".format(len(train_data)))
    print("test count:{}".format(len(test_data)))

    positive_count = 0
    negative_count = 0
    for _, _, label in train_data:
        if label == 0:
            negative_count += 1
        elif label == 1:
            positive_count += 1
    print("positive_count:{}, negative_count:{}".format(positive_count, negative_count))


if __name__ == '__main__':
    my_test()
    # analyze_data()
