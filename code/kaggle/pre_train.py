import tokenizers
import pandas as pd
from transformers import BertTokenizer, PreTrainedTokenizer
from transformers import BertConfig, BertForMaskedLM, DataCollatorForLanguageModeling, DistilBertForQuestionAnswering
from transformers import Trainer, TrainingArguments
from transformers import pipeline
from torch.utils.data.dataset import Dataset
import os
import warnings
import torch


DEPRECATION_WARNING = (
    "This dataset will be removed from the library soon, preprocessing should be handled with the 🤗 Datasets "
    "library. You can have a look at this example script for pointers: {0}"
)


class LineByLineTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int):
        warnings.warn(
            DEPRECATION_WARNING.format(
                "https://github.com/huggingface/transformers/blob/master/examples/language-modeling/run_mlm.py"
            ),
            FutureWarning,
        )
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        print("Creating features from dataset file at %s", file_path)

        lines = []
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                if len(line) <= 0 or line.isspace():
                    continue
                line = line.split('\t')
                lines.append([line[0], line[1]])

        batch_encoding = tokenizer(lines, add_special_tokens=True, truncation=True, max_length=block_size)
        self.examples = batch_encoding["input_ids"]
        token_type_ids = batch_encoding["token_type_ids"]
        self.examples = [{"input_ids": torch.tensor(e, dtype=torch.long),
                          "token_type_ids": torch.tensor(token_type_ids[i], dtype=torch.long)} for i, e in
                         enumerate(self.examples)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


def load_data(filename):
    """加载数据
    单条格式：(文本1 ids, 文本2 ids, 标签id)
    """
    D = []
    with open(filename) as f:
        for l in f:
            if l == '\n':
                continue
            l = l.strip().split('\t')
            if len(l) == 3:
                a, b, c = l[0], l[1], int(l[2])
            else:
                a, b, c = l[0], l[1], -5  # 未标注数据，标签为-5
            a = [i for i in a.split(' ')]
            b = [i for i in b.split(' ')]
            D.append((a, b, c))
    return D


def preprocess_data():
    # 加载数据集
    data = load_data(
        # '../oppo_breeno_round1_data/gaiic_track3_round1_train_20210228.tsv'
        '../../tcdata/gaiic_track3_round1_train_20210228.tsv'
        # '../user_data/train-bpe-100.txt'
    )
    train_data = [d for i, d in enumerate(data) if i % 10 != 0]
    valid_data = [d for i, d in enumerate(data) if i % 10 == 0]
    test_data = load_data(
        # '../oppo_breeno_round1_data/gaiic_track3_round1_testA_20210228.tsv'
        '../../tcdata/gaiic_track3_round1_testA_20210228.tsv'
        # '../user_data/test-bpe-100.txt'
    )
    # print(train_data[0])
    return train_data, valid_data, test_data


def get_vocab_file(all_data):
    # 统计词频
    tokens = {}
    for d in all_data:
        for i in d[0] + d[1]:
            tokens[i] = tokens.get(i, 0) + 1

    # 舍弃一些词频小的字
    tokens = {i: j for i, j in tokens.items() if j >= 5}
    # print(tokens)
    tokens = sorted(tokens.items(), key=lambda s: -s[1])
    # print(tokens)
    # 保留 0: pad, 1: unk, 2: cls, 3: sep, 4: mask, 5: no, 6: yes
    res = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
    res += [
        t[0]
        for i, t in enumerate(tokens)
    ]
    vocab_file = '../../user_data/vocab.txt'
    with open(vocab_file, 'w') as f:
        for line in res:
            f.write(line + '\n')
    # print(tokens)
    return vocab_file


def convert_data_to_sentence_pair(data, output='../../user_data/data_lines.txt'):
    with open(output, 'w') as f:
        for line1, line2, label in data:
            line1 = ' '.join(line1)
            line2 = ' '.join(line2)
            f.write(line1 + '\t' + line2 + ' \t' + str(label) + '\n')
    return output


def get_bert_model():
    config = BertConfig(
        vocab_size=8000,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        max_position_embeddings=512
    )

    model = BertForMaskedLM(config)
    model.forward()
    print('No of parameters: ', model.num_parameters())
    return model


def get_trainer(model, data_collator, dataset):
    training_args = TrainingArguments(
        output_dir='../../user_data/models/kaggle_bert/',
        overwrite_output_dir=True,
        num_train_epochs=100,
        per_device_train_batch_size=32,
        save_steps=10_000,
        save_total_limit=2,
        prediction_loss_only=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )
    trainer.train()
    return trainer


def predict(model_dir, sentence, tokenizer):
    model = BertForMaskedLM.from_pretrained(model_dir)
    fill_mask = pipeline(
        "fill-mask",
        model=model,
        tokenizer=tokenizer
    )
    return fill_mask(sentence)


def train():
    train_data, valid_data, test_data = preprocess_data()
    # 模拟未标注
    for d in valid_data + test_data:
        train_data.append((d[0], d[1], -5))

    # vocab_file = get_vocab_file(train_data)
    vocab_file = '../../user_data/vocab.txt'
    tokenizer = BertTokenizer.from_pretrained(vocab_file)
    # test tokenizer
    # sentenceA = "17 18 12 19 20 21 22 23 24"
    # sentenceB = "12 23 25 6 26 27 19"
    # print(tokenizer(text=[[sentenceA, sentenceB]]))

    # tokenA = tokenizer.encode(sentenceA)
    # tokenB = tokenizer.encode(sentenceB)
    # print(tokenA, tokenB)
    # type_ids = tokenizer.create_token_type_ids_from_sequences(tokenA, tokenB)
    # print(type_ids)

    # 获取dataset
    train_file = convert_data_to_sentence_pair(train_data)

    # train_file = '../../user_data/data_lines.txt'
    dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=train_file,
        block_size=32  # maximum sequence length
    )
    # for data in dataset:
    #     print(data)
    #     break
    #
    print('No. of lines: ', len(dataset))  # No of lines in your datset

    #
    bert_model = get_bert_model()
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    #
    trainer = get_trainer(bert_model, data_collator, dataset)
    trainer.train()
    trainer.save_model(output_dir='../../user_data/models/kaggle_bert/')


if __name__ == '__main__':
    train()
