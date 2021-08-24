import pandas as pd
import numpy as np


def cal_dim(df):
    data = df.values
    m, n = data.shape
    print(m, n)
    res_dim = 0
    for i in range(m):
        for j in range(n-1):
            # print(type(data[i, j]))
            for num in data[i, j].split(' '):
                num = int(num)
                if num > res_dim:
                    res_dim = num
            # break
        # break
    return res_dim


def preprocess():
    train_filename = "../tcdata/gaiic_track3_round1_train_20210228.tsv"
    train_df = pd.read_csv(train_filename, sep="\t", encoding="utf-8", header=None)
    print(train_df.head())

    test_filename = "../tcdata/gaiic_track3_round1_testA_20210228.tsv"
    test_df = pd.read_csv(test_filename, sep="\t", encoding="utf-8", header=None)
    # 大赛数据对文字做了编码处理，很贱
    # 需要首先计算词典维度

    train_dim = cal_dim(train_df)
    print("train dim: {}".format(train_dim))

    test_dim = cal_dim(test_df)
    print("test dim: {}".format(test_dim))


if __name__ == '__main__':
    preprocess()



