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


def load_data(corpus_path):
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
            truncate_sequences(-1, a, b)
            D.append((a, b, c))
    return D


def truncate_sequences(index, *sequences):
    """截断总长度至不超过maxlen
    """
    sequences = [s for s in sequences if s]
    while True:
        lengths = [len(s) for s in sequences]
        if sum(lengths) > config['seq_max_len'] - 3:
            i = np.argmax(lengths)
            sequences[i].pop(index)
        else:
            return sequences


def split_data():
    # 划分数据集
    data = load_data(
        config['train_data_path']
    )
    train_data = [d for i, d in enumerate(data) if i % 10 != 0]
    valid_data = [d for i, d in enumerate(data) if i % 10 == 0]
    test_data = load_data(
        config['test_data_path']
    )
    # print(train_data[0])
    # 模拟未标注
    # for d in valid_data + test_data:
    #     train_data.append((d[0], d[1], -5))
    return train_data, valid_data, test_data


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
    # 保留 0: pad, 1: unk, 2: cls, 3: sep, 4: mask, 5: no, 6: yes
    tokens = {
        t[0]: i + 5
        for i, t in enumerate(tokens)
    }
    # print(tokens)
    return tokens


class PreTrainer(object):

    def __init__(self,
                 bert_config,
                 max_seq_len,
                 batch_size,
                 lr,
                 warmup_steps,
                 train_data,
                 valid_data,
                 test_data,
                 word2idx_path=None,
                 word2idx=None,
                 with_cuda=True,
                 num_workers=1,
                 ):
        self.bert_config = bert_config
        self.batch_size = batch_size
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.curr_steps = 0

        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")

        self.max_seq_len = max_seq_len

        self.bert_model = BertForPreTraining(self.bert_config)
        self.bert_model.to(self.device)

        train_dataset = BERTDataSet(
            data=train_data,
            seq_len=self.max_seq_len,
            word2idx_path=word2idx_path,
            word2idx=word2idx,
            random=True
        )
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn
        )
        valid_dataset = BERTDataSet(data=valid_data,
                                    seq_len=self.max_seq_len,
                                    word2idx_path=word2idx_path,
                                    word2idx=word2idx,
                                    random=True
                                    )
        self.valid_dataloader = DataLoader(valid_dataset,
                                           batch_size=self.batch_size,
                                           num_workers=num_workers,
                                           collate_fn=collate_fn)

        test_dataset = BERTDataSet(data=test_data,
                                   seq_len=self.max_seq_len,
                                   word2idx_path=word2idx_path,
                                   word2idx=word2idx,
                                   random=False
                                   )
        self.test_dataloader = DataLoader(test_dataset,
                                          batch_size=self.batch_size,
                                          num_workers=num_workers,
                                          collate_fn=collate_fn)

        self.hidden_dim = bert_config.hidden_size

        self.positional_enc = self.init_positional_encoding()
        self.positional_enc = torch.unsqueeze(self.positional_enc, dim=0)

        self.optim_parameters = list(self.bert_model.parameters())
        self.optimizer = torch.optim.Adam(self.optim_parameters, lr=lr)
        print(self.bert_model)

        print("Total Parameters:", sum([p.nelement() for p in self.bert_model.parameters()]))

    def init_positional_encoding(self):
        position_enc = np.array([
            [pos / np.power(10000, 2 * i / self.hidden_dim) for i in range(self.hidden_dim)]
            if pos != 0 else np.zeros(self.hidden_dim) for pos in range(self.max_seq_len)])

        position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
        position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i+1
        denominator = np.sqrt(np.sum(position_enc ** 2, axis=1, keepdims=True))
        position_enc = position_enc / (denominator + 1e-8)
        position_enc = torch.from_numpy(position_enc).type(torch.FloatTensor)
        return position_enc

    def compute_loss(self, predictions, labels, num_class=2, ignore_index=None):
        if ignore_index is None:
            loss_func = CrossEntropyLoss()
        else:
            loss_func = CrossEntropyLoss(ignore_index=ignore_index)
        return loss_func(predictions.view(-1, num_class), labels.view(-1))

    def load_model(self, model, dir_path="./output"):
        # 加载模型
        checkpoint_dir = self.find_most_recent_state_dict(dir_path)
        if checkpoint_dir is None:
            return 0
        epoch = checkpoint_dir.split('.')[-1]
        checkpoint = torch.load(checkpoint_dir)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        torch.cuda.empty_cache()
        model.to(self.device)
        print("{} loaded for training!".format(checkpoint_dir))
        return int(epoch) + 1

    def find_most_recent_state_dict(self, dir_path):
        dic_lis = [i for i in os.listdir(dir_path)]
        if len(dic_lis) == 0:
            # raise FileNotFoundError("can not find any state dict in {}!".format(dir_path))
            print("can not find any state dict in {}!".format(dir_path))
            return None
        dic_lis = [i for i in dic_lis if "model" in i]
        dic_lis = sorted(dic_lis, key=lambda k: int(k.split(".")[-1]))
        return dir_path + "/" + dic_lis[-1]

    def save_state_dict(self, model, epoch, dir_path="./output", file_path="bert.model"):
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        save_path = os.path.join(dir_path, file_path + ".epoch.{}".format(str(epoch)))
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

    def get_nsp_accuracy(self, predictions, labels, ignore_index=None):
        """
         最简单粗暴的方法
   　　　 先排序，然后统计有多少正负样本对满足：正样本预测值>负样本预测值, 再除以总的正负样本对个数
         复杂度 O(NlogN), N为样本数
        """

        predictions = predictions.view(-1)

        if ignore_index is not None:
            labels_index = (labels != ignore_index).nonzero().view(-1).to(self.device)
            predictions = torch.index_select(predictions, dim=0, index=labels_index)
            labels = torch.index_select(labels, dim=0, index=labels_index)

        n_pos = torch.sum(labels).to(self.device)
        n_neg = (labels.shape[0] - n_pos).to(self.device)
        total_pair = (n_pos * n_neg).to(self.device)

        sorted_preds, indices = torch.sort(predictions, descending=True)
        accumulated_neg = n_pos + n_neg
        satisfied_pair = torch.tensor(0.0, dtype=torch.float32)

        sorted_preds = sorted_preds.to(self.device)
        indices = indices.to(self.device)
        accumulated_neg = accumulated_neg.to(self.device)
        satisfied_pair = satisfied_pair.to(self.device)

        for i, score in enumerate(sorted_preds):
            if labels[indices[i]] == 1:
                satisfied_pair += accumulated_neg
            accumulated_neg -= 1

        satisfied_pair -= n_pos * (n_pos + 1) / 2
        res = satisfied_pair / (total_pair + 1e-8)
        return res.item()

    def train(self, epoch, df_path="../user_data/models/df_log.pickle"):
        self.bert_model.train()
        self.iteration(epoch, self.train_dataloader, train=True, df_path=df_path)

    # def predict(self, result_path='../../prediction_result/result.txt'):
    #     if os.path.isfile(result_path):
    #         with open(result_path, 'w') as f:
    #             pass
    #
    #     data_iter = tqdm.tqdm(enumerate(self.test_dataloader),
    #                           desc="EP_test",
    #                           total=len(self.test_dataloader),
    #                           bar_format="{l_bar}{r_bar}")
    #     with open(result_path, 'a') as f:
    #         for i, data in data_iter:
    #             input_ids, attention_mask, segment_ids, output_ids, label = data
    #             input_ids = input_ids.to(self.device)
    #             segment_ids = segment_ids.to(self.device)
    #             attention_mask = attention_mask.to(self.device)
    #             seq_len = input_ids.shape[1]
    #             positional_enc = self.positional_enc[0, :seq_len, :].to(self.device)
    #
    #             # 1. forward the next_sentence_prediction and masked_lm model
    #             _, next_sen_preds = self.bert_model.forward(input_ids=input_ids,
    #                                                                 positional_enc=positional_enc,
    #                                                                 token_type_ids=segment_ids,
    #                                                                 attention_mask=attention_mask,
    #                                                                 )
    #             next_sen_preds = next_sen_preds[:, 1]
    #             for p in next_sen_preds:
    #                 f.write('%f\n' % p)
    #             # break

    def iteration(self, epoch, data_loader, train=True, df_path="../../user_data/models/df_log.csv"):
        if not os.path.isfile(df_path):
            df = pd.DataFrame(columns=[
                "epoch",
                "train_mlm_loss",
                "train_mlm_acc",
                "valid_mlm_loss",
                "valid_mlm_acc",
            ])
            df.to_csv(df_path, index=False)
            print("log DataFrame created!")

        str_code = "train" if train else "valid"

        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

        total_mlm_loss = 0
        total_mlm_acc = 0
        total_nsp_loss = 0
        total_nsp_acc = 0
        acc_count = 0
        nsp_count = 0

        for i, data in data_iter:
            self.curr_steps += 1
            input_ids, segment_ids, output_ids, label = data
            input_ids = input_ids.to(self.device)
            segment_ids = segment_ids.to(self.device)
            output_ids = output_ids.to(self.device)
            label = label.to(self.device)

            seq_len = input_ids.shape[1]

            positional_enc = self.positional_enc[0, :seq_len, :].to(self.device)

            mlm_preds, nsp_preds = self.bert_model.forward(input_ids=input_ids,
                                                           positional_enc=positional_enc,
                                                           token_type_ids=segment_ids,
                                                           )

            mlm_acc = self.get_mlm_accuracy(mlm_preds, output_ids)
            # print(mlm_acc)
            mlm_loss = self.compute_loss(mlm_preds, output_ids, self.bert_config.vocab_size, ignore_index=0)
            total_mlm_loss += mlm_loss.item()
            loss = mlm_loss

            nsp_loss = self.compute_loss(nsp_preds, label, num_class=2, ignore_index=-5)
            nsp_auc = self.get_nsp_accuracy(nsp_preds, label, ignore_index=-5)
            loss += nsp_loss
            total_nsp_loss += nsp_loss.item()
            total_nsp_acc += nsp_auc
            nsp_count += 1

            if train:
                # lr = lr_warm_up_and_decay(self.lr, self.curr_steps, self.warmup_steps)
                self.optimizer.zero_grad()
                loss.backward()
                # for param in self.model.parameters():
                #     print(param.grad.data.sum())
                self.optimizer.step()

            if torch.sum(output_ids) != 0:
                total_mlm_acc += mlm_acc
                acc_count += 1

            if train:
                log_dic = {
                    "epoch": epoch,
                    "train_mlm_loss": total_mlm_loss / (i + 1),
                    "train_mlm_acc": total_mlm_acc / (acc_count + 1),
                    "train_nsp_loss": total_nsp_loss / (nsp_count + 1),
                    "train_nsp_acc": total_nsp_acc / (nsp_count + 1),
                    "valid_mlm_loss": 0,
                    "valid_mlm_acc": 0,
                    "valid_nsp_loss": 0,
                    "valid_nsp_acc": 0,
                }

            else:
                log_dic = {
                    "epoch": epoch,
                    "valid_mlm_loss": total_mlm_loss / (i + 1),
                    "valid_mlm_acc": total_mlm_acc / (acc_count + 1),
                    "valid_nsp_loss": total_nsp_loss / (nsp_count + 1),
                    "valid_nsp_acc": total_nsp_acc / (nsp_count + 1),
                }

            if i % 10 == 0:
                data_iter.write(str(log_dic))

        if train:
            df = pd.read_csv(df_path)
            log_dic_df = pd.DataFrame([log_dic.values()], columns=log_dic.keys())
            df = pd.concat([df, log_dic_df], axis=0)
            df.reset_index(inplace=True, drop=True)
            df.to_csv(df_path, index=False)
        else:
            df = pd.read_csv(df_path)
            epoch = log_dic['epoch']
            del log_dic['epoch']
            for key in log_dic:
                df.loc[df.epoch == epoch, key] = log_dic[key]
            df.reset_index(inplace=True, drop=True)
            df.to_csv(df_path, index=False)
            return float(log_dic["test_mlm_loss"])

    def test(self, epoch, df_path="./output_wiki_bert/df_log.pickle"):
        self.bert_model.eval()
        with torch.no_grad():
            return self.iteration(epoch, self.valid_dataloader, train=False, df_path=df_path)


def train_loss_vision(df_path):
    df = pd.read_csv(df_path)
    train_loss = df.groupby('epoch')['train_mlm_loss'].mean()
    valid_loss = df.groupby('epoch')['test_mlm_loss'].mean()
    print(train_loss)
    print(valid_loss)
    fig, ax = plt.subplots()
    ax.plot(train_loss.index, train_loss.values, label='train loss')
    ax.plot(valid_loss.index, valid_loss.values, label='valid loss')

    ax.set_xlabel('epoch')
    ax.set_ylabel('mlm_loss')
    ax.set_title('train loss figure')
    ax.legend()
    plt.show()


def init_trainer(dynamic_lr, load_model=False):
    train_data, valid_data, test_data = split_data()
    word2idx = get_tokens(train_data + valid_data + test_data)

    # train_data = train_data[:100]
    # valid_data = valid_data[:100]
    train_data = train_data + test_data + valid_data
    bert_config = BertConfig(**config['bert_config'])
    # print(min(word2idx.values()))
    # print(len(word2idx))
    #
    # bert_config.vocab_size = max(word2idx.values())+1
    print("vacab_size: ", bert_config.vocab_size)
    trainer = PreTrainer(
        bert_config=bert_config,
        max_seq_len=config['seq_max_len'],
        batch_size=config['pretrain']['batch_size'],
        lr=dynamic_lr,
        warmup_steps=config['pretrain']['warmup_steps'],
        train_data=train_data,
        valid_data=valid_data,
        test_data=test_data,
        word2idx_path=None,
        word2idx=word2idx,
        with_cuda=True,
        num_workers=1,
    )
    if load_model:
        trainer.load_model(trainer.bert_model, dir_path=config["output_path"])
    return trainer


def train():
    lr = float(config['nsptrain']['lr'])
    trainer = init_trainer(lr, load_model=False)
    train_epochs = config['nsptrain']['epochs']
    # load pre-trained model
    trainer.load_model(trainer.bert_model, dir_path=config["bert_model_path"])
    # load nsp model
    start_epoch = trainer.load_model(trainer.bert_model, dir_path=config["nsp_model_path"])

    start_time = datetime.datetime.now()
    print("trainning start at {}, start_epoch:{} train_epochs:{}".format(start_time, start_epoch, train_epochs))
    for epoch in range(start_epoch, train_epochs + 1):
        trainer.train(epoch, df_path=config['nsp_train_log_dir'])
        trainer.test(epoch, df_path=config['nsp_train_log_dir'])
        if epoch % config['nsptrain']['model_save_epoch'] == 0:
            trainer.save_state_dict(trainer.bert_model, epoch, dir_path=config["nsp_model_path"],
                                    file_path="bert.model")

        # break
    end_time = datetime.datetime.now()
    print("training finished at {}, used {}".format(end_time, end_time - start_time))


# def predict():
#     lr = float(config['lr'])
#     trainer = init_trainer(lr, load_model=True)
#     trainer.predict()


if __name__ == '__main__':
    # predict()
    train()
    # train_loss_vision(df_path=config['pre_train_log_dir'])
