import baseline
import json
from bert4keras.tokenizers import load_vocab
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
import numpy as np
from tqdm import tqdm
import os


min_count = 5
dict_path = '../user_data/chinese_L-12_H-768_A-12/vocab.txt'
config_path = '../user_data/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '../user_data/chinese_L-12_H-768_A-12/bert_model.ckpt'
batch_size = 32
model_path = '../user_data/models'
epochs = 100


def preprocess_data():
    # 加载数据集
    data = baseline.load_data(
        # '../oppo_breeno_round1_data/gaiic_track3_round1_train_20210228.tsv'
        # '../tcdata/gaiic_track3_round1_train_20210228.tsv'
        '../user_data/train-bpe-100.txt'
    )
    train_data = [d for i, d in enumerate(data) if i % 10 != 0]
    valid_data = [d for i, d in enumerate(data) if i % 10 == 0]
    test_data = baseline.load_data(
        # '../oppo_breeno_round1_data/gaiic_track3_round1_testA_20210228.tsv'
        # '../tcdata/gaiic_track3_round1_testA_20210228.tsv'
        '../user_data/test-bpe-100.txt'
    )
    # print(train_data[0])
    # 模拟未标注
    for d in valid_data + test_data:
        train_data.append((d[0], d[1], -5))
    return train_data, valid_data, test_data


def get_tokens(all_data):
    # 统计词频
    tokens = {}
    for d in all_data:
        for i in d[0] + d[1]:
            tokens[i] = tokens.get(i, 0) + 1

    # 舍弃一些词频小的字
    tokens = {i: j for i, j in tokens.items() if j >= min_count}
    # print(tokens)
    tokens = sorted(tokens.items(), key=lambda s: -s[1])
    # print(tokens)
    # 保留 0: pad, 1: unk, 2: cls, 3: sep, 4: mask, 5: no, 6: yes
    tokens = {
        t[0]: i + 7
        for i, t in enumerate(tokens)
    }
    # print(tokens)
    return tokens


def get_keep_tokens():
    # BERT词频
    counts = json.load(open('../user_data/counts.json'))
    del counts['[CLS]']
    del counts['[SEP]']
    token_dict = load_vocab(dict_path)
    # print(len(token_dict))
    freqs = [
        counts.get(i, 0) for i, j in sorted(token_dict.items(), key=lambda s: s[1])
    ]
    # print(len(freqs))
    # print(np.argsort(freqs))
    keep_tokens = list(np.argsort(freqs)[::-1])
    # print(keep_tokens)
    return keep_tokens


def train(train_generator, valid_generator, keep_tokens):
    # 加载预训练模型
    model = build_transformer_model(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        with_mlm=True,
        keep_tokens=[0, 100, 101, 102, 103, 100, 100] + keep_tokens[:len(tokens)]
    )
    model.compile(loss=baseline.masked_crossentropy, optimizer=Adam(1e-5))
    model.summary()

    # test_generator = baseline.data_generator(test_data, batch_size)
    evaluator = baseline.Evaluator(model, valid_generator, model_path=model_path)
    model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=[evaluator]
    )
    print("train finished, best model saved")
    return model


def predict_to_file(model, test_generator, out_file):
    """预测结果到文件
    """
    F = open(out_file, 'w')
    for x_true, _ in tqdm(test_generator):
        y_pred = model.predict(x_true)[:, 0, 5:7]
        y_pred = y_pred[:, 1] / (y_pred.sum(axis=1) + 1e-8)
        for p in y_pred:
            F.write('%f\n' % p)
    F.close()


if __name__ == '__main__':
    train_data, valid_data, test_data = preprocess_data()
    tokens = get_tokens(all_data=train_data + valid_data + test_data)
    keep_tokens = get_keep_tokens()
    # print(keep_tokens)
    # print(len(tokens))

    train_generator = baseline.data_generator(train_data, batch_size, tokens=tokens)
    valid_generator = baseline.data_generator(valid_data, batch_size, tokens=tokens)
    model = train(train_generator, valid_generator, keep_tokens)

    model.load_weights(os.path.join(model_path, "best_model.weights"))
    test_generator = baseline.data_generator(test_data, batch_size, tokens=tokens)
    predict_to_file(model, test_generator, out_file="../prediction_result/result.txt")
