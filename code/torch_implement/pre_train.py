import tqdm
import pandas as pd
import numpy as np
import os
from torch_implement import config, lr_warm_up_and_decay
from torch.utils.data import DataLoader
from torch_implement.dataset import BERTDataSet, collate_fn
import torch
from torch_implement.bert_model import BertConfig, BertModel, BertForPreTraining
from torch_implement.my_model import MyBertForPreTraining
from torch.nn import CrossEntropyLoss
import matplotlib.pyplot as plt
import datetime
import pytorch_warmup as warmup
from torchkeras import summary
from collections import defaultdict


class PreTrainer(object):

    def __init__(self,
                 bert_config,
                 lr,
                 warmup_steps,
                 train_data_loader,
                 with_cuda=True,
                 ):
        self.bert_config = bert_config
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.train_data_loader = train_data_loader

        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")

        self.bert_model = MyBertForPreTraining(self.bert_config)
        self.bert_model.to(self.device)

        self.hidden_dim = bert_config.hidden_size

        # self.positional_enc = self.init_positional_encoding()
        # self.positional_enc = torch.unsqueeze(self.positional_enc, dim=0)

        self.optim_parameters = list(self.bert_model.parameters())
        self.optimizer = torch.optim.Adam(self.optim_parameters, lr=lr, weight_decay=config['pretrain']['weight_decay'])
        # summary(self.bert_model, input_shape=(3, self.max_seq_len), input_dtype=torch.int32)

        print("Total Parameters:", sum([p.nelement() for p in self.bert_model.parameters()]))

    # def init_positional_encoding(self):
    #     position_enc = np.array([
    #         [pos / np.power(10000, 2 * i / self.hidden_dim) for i in range(self.hidden_dim)]
    #         if pos != 0 else np.zeros(self.hidden_dim) for pos in range(self.max_seq_len * 2)])
    #
    #     position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
    #     position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i+1
    #     denominator = np.sqrt(np.sum(position_enc ** 2, axis=1, keepdims=True))
    #     position_enc = position_enc / (denominator + 1e-8)
    #     position_enc = torch.from_numpy(position_enc).type(torch.FloatTensor)
    #     return position_enc

    def load_model(self, model, dir_path="./output"):
        # 加载模型
        checkpoint_dir = self.find_most_recent_state_dict(dir_path)
        if checkpoint_dir is None:
            return 1
        epoch = checkpoint_dir.split('.')[-1]
        checkpoint = torch.load(checkpoint_dir)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        torch.cuda.empty_cache()
        model.to(self.device)
        print("{} loaded for training!".format(checkpoint_dir))
        return int(epoch)+1

    def find_most_recent_state_dict(self, dir_path):
        dic_lis = [i for i in os.listdir(dir_path)]
        dic_lis = [i for i in dic_lis if "model" in i]
        if len(dic_lis) == 0:
            # raise FileNotFoundError("can not find any state dict in {}!".format(dir_path))
            print("can not find any state dict in {}!".format(dir_path))
            return None
        dic_lis = sorted(dic_lis, key=lambda k: int(k.split(".")[-1]))
        return dir_path + "/" + dic_lis[-1]

    def save_state_dict(self, model, epoch, dir_path="./output", file_path="bert.model"):
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        save_path = os.path.join(dir_path, file_path+".epoch.{}".format(str(epoch)))
        model.to("cpu")
        torch.save({"model_state_dict": model.state_dict()}, save_path)
        print("{} saved!".format(save_path))
        model.to(self.device)

    def get_mlm_accuracy(self, predictions, labels):
        predictions = torch.argmax(predictions, dim=-1, keepdim=False)
        mask = (labels > 0).to(self.device)
        mlm_accuracy = torch.sum((predictions == labels) * mask).float()
        mlm_accuracy /= (torch.sum(mask).float() + 1e-8)
        return mlm_accuracy.item()

    def train(self, epoch, df_path):
        self.bert_model.train()
        self.iteration(epoch, df_path=df_path)

    def iteration(self, epoch, df_path="../../user_data/models/df_log.csv"):
        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(self.train_data_loader),
                              desc="EP_train:%d" % epoch,
                              total=len(self.train_data_loader),
                              bar_format="{l_bar}{r_bar}")

        # 设置学习率warmup和decay
        num_steps = len(self.train_data_loader) * (epoch-1)
        decay_rate = config['pretrain']['decay_rate']
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda x: 1/(1+decay_rate*epoch))
        warmup_scheduler = warmup.LinearWarmup(self.optimizer, warmup_period=config['pretrain']['warmup_steps'])
        warmup_scheduler.last_step = num_steps

        # 因为有可能没有mask的样本loss则为0，不必计算
        total_loss = 0
        total_count = 0
        log_dic = defaultdict(list)
        for i, data in data_iter:
            input_ids, segment_ids, output_ids, label = data

            output_ids = output_ids.to(self.device)

            batch_size, seq_len = input_ids.shape

            # 生成位置id
            position_ids = torch.arange(0, seq_len * batch_size, dtype=torch.int32)
            position_ids = position_ids.view(batch_size, seq_len)
            position_ids = torch.fmod(position_ids, seq_len)

            input_tensor = torch.cat((input_ids, segment_ids, position_ids), 1).view(batch_size, 3, seq_len).to(self.device)
            mlm_preds = self.bert_model.forward(input_tensor)

            mlm_acc = self.get_mlm_accuracy(mlm_preds, output_ids)
            # print(mlm_acc)
            mlm_loss = self.bert_model.compute_loss(mlm_preds, output_ids, self.bert_config.vocab_size, ignore_index=0)
            loss = mlm_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            lr_scheduler.step()
            warmup_scheduler.dampen()
            if mlm_loss.item() != 0:
                total_loss += mlm_loss.item()
                total_count += 1
            log_dic['epoch'].append(epoch)
            log_dic['steps'].append(num_steps+i+1)
            log_dic['train_mlm_loss'].append(mlm_loss.item())
            log_dic['train_mlm_acc'].append(mlm_acc)

            data_iter.set_postfix(loss=total_loss/total_count,
                                  acc=mlm_acc,
                                  lr=self.optimizer.param_groups[0]['lr'])
            data_iter.update(1)

        log_dic_df = pd.DataFrame().from_dict(log_dic)
        if os.path.isfile(df_path):
            df = pd.read_csv(df_path)
            df = pd.concat([df, log_dic_df], axis=0)
        else:
            df = log_dic_df
        df.reset_index(inplace=True, drop=True)
        df.to_csv(df_path, index=False)


def load_data(corpus_path, max_len):
    """加载数据
    单条格式：(文本1 ids, 文本2 ids, 标签id)
    """
    D = []
    with open(corpus_path) as f:
        for l in f:
            l = l.strip().split('\t')
            if len(l) == 3:
                a, b, c = l[0], l[1], int(l[2])
            else:
                a, b, c = l[0], l[1], -5  # 未标注数据，标签为-5
            a = [int(i) for i in a.split(' ')]
            b = [int(i) for i in b.split(' ')]
            truncate_sequences(max_len, -1, a, b)
            D.append((a, b, c))
    return D


def truncate_sequences(max_len, index, *sequences):
    """截断总长度至不超过maxlen
    """
    sequences = [s for s in sequences if s]
    while True:
        lengths = [len(s) for s in sequences]
        if sum(lengths) > max_len:
            i = np.argmax(lengths)
            sequences[i].pop(index)
        else:
            return sequences


def get_tokens(all_data):
    # 统计词频
    tokens = {}
    for d in all_data:
        for i in d[0] + d[1]:
            tokens[i] = tokens.get(i, 0) + 1

    # 舍弃一些词频小的字
    tokens = {i: j for i, j in tokens.items() if j >= config['min_count']}
    # print(tokens)
    # 把词频高的词排在前面
    tokens = sorted(tokens.items(), key=lambda s: -s[1])
    # print(tokens)
    # 保留 0: pad, 1: unk, 2: cls, 3: sep, 4: mask
    tokens = {
        t[0]: i + 5
        for i, t in enumerate(tokens)
    }
    # print(tokens)
    return tokens


# 获取训练集、验证集、测试集的数据
def preprocess_data():
    data = load_data(
        config['train_data_path'],
        config['seq_max_len']
    )
    train_data = [d for i, d in enumerate(data) if i % 10 != 0]
    valid_data = [d for i, d in enumerate(data) if i % 10 == 0]
    test_data = load_data(
        config['test_data_path'],
        config['seq_max_len']
    )

    return train_data, valid_data, test_data


def get_data_loader(data, tokens, random, batch_size, num_workers):
    dataset = BERTDataSet(
        data=data,
        word2idx=tokens,
        random=random
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    return dataloader


def init_trainer(dynamic_lr, load_model=False):
    train_data, valid_data, test_data = preprocess_data()
    tokens = get_tokens(train_data+valid_data+test_data)
    print(len(tokens))

    train_data = train_data+test_data+valid_data
    # train_data = train_data[:2]

    bert_config = BertConfig(**config['bert_config'])
    train_data_loader = get_data_loader(train_data, tokens, random=True, batch_size=config['pretrain']['batch_size'], num_workers=2)

    # print("vacab_size: ", bert_config.vocab_size)
    trainer = PreTrainer(
        bert_config=bert_config,
        lr=dynamic_lr,
        warmup_steps=config['pretrain']['warmup_steps'],
        train_data_loader=train_data_loader,
        with_cuda=True,
    )
    if load_model:
        trainer.load_model(trainer.bert_model, dir_path=config["output_path"])
    return trainer


def train_loss_vision(df_path):
    df = pd.read_csv(df_path)
    train_loss = df.groupby('epoch')['train_mlm_loss'].mean()
    print(train_loss)
    fig, ax = plt.subplots()
    ax.plot(train_loss.index, train_loss.values, label='train loss')

    ax.set_xlabel('epoch')
    ax.set_ylabel('mlm_loss')
    ax.set_title('train loss figure')
    ax.legend()
    plt.show()


def train():
    lr = float(config['pretrain']['lr'])
    trainer = init_trainer(lr, load_model=False)
    train_epochs = config['pretrain']['epochs']
    start_epoch = trainer.load_model(trainer.bert_model, dir_path=config["output_path"]+'/bert')
    start_time = datetime.datetime.now()
    print("training start at {}, start_epoch:{} train_epochs:{}".format(start_time, start_epoch, train_epochs))
    seed = np.random.randint(1, 10000)
    for epoch in range(start_epoch, train_epochs+1):
        np.random.seed(seed+epoch)
        trainer.train(epoch, df_path=config['pre_train_log_dir'])
        # trainer.test(epoch, df_path=config['pre_train_log_dir'])
        if epoch % config['pretrain']['model_save_epoch'] == 0:
            trainer.save_state_dict(trainer.bert_model, epoch, dir_path=config["bert_model_path"],
                                    file_path="bert.model")
        # break
    end_time = datetime.datetime.now()
    print("training finished at {}, used {}".format(end_time, end_time-start_time))


if __name__ == '__main__':
    # init_trainer(dynamic_lr=1e-4, load_model=False)
    train()
    # train_loss_vision(df_path=config['pre_train_log_dir'])
