from torch_implement.bert_model import BertConfig
import torch
from torch_implement.my_model import SimilarSentenceModel
from torch_implement.dataset import BERTDataSet, collate_fn
import os
from torch_implement import config
from torch.utils.data import DataLoader
from sklearn import metrics
import pandas as pd
import numpy as np
import tqdm
from torch_implement import find_best_threshold
from torch_implement.pre_train import get_tokens, preprocess_data
import datetime
import matplotlib.pyplot as plt
from torch_implement.pre_train import get_data_loader


class SimilarityTrainer(object):

    def __init__(self,
                 vocab_size,
                 bert_config,
                 batch_size,
                 lr,
                 train_data_loader,
                 valid_data_loader,
                 with_cuda=True,
                 ):
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.lr = lr
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        # 判断是否有可用GPU
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")

        self.bert_model = SimilarSentenceModel(config=bert_config)
        # 将模型发送到计算设备(GPU或CPU)
        self.bert_model.to(self.device)
        print(self.bert_model)

        # 初始化位置编码
        self.hidden_dim = bert_config.hidden_size

        # 声明需要优化的参数, 并传入Adam优化器
        self.optim_parameters = list(self.bert_model.parameters())

        self._init_optimizer(lr=self.lr)

    def _init_optimizer(self, lr):
        self.optimizer = torch.optim.Adam(self.optim_parameters, lr=lr, weight_decay=1e-3)

    def load_model(self, model, dir_path="../../user_data/models/similarity"):
        checkpoint_dir = self.find_most_recent_state_dict(dir_path)
        if not checkpoint_dir:
            return 1
        checkpoint = torch.load(checkpoint_dir)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        torch.cuda.empty_cache()
        model.to(self.device)
        print("sim model: {} loaded!".format(checkpoint_dir))
        start_epoch = checkpoint_dir.split('.')[-1]
        return int(start_epoch) + 1

    def load_bert_model(self, model, dir_path="../../user_data/models/bert"):
        checkpoint_dir = self.find_most_recent_state_dict(dir_path)
        if not checkpoint_dir:
            raise FileNotFoundError("can not find any state dict in {}!".format(dir_path))
        checkpoint = torch.load(checkpoint_dir)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        torch.cuda.empty_cache()
        model.to(self.device)
        print("bert model: {} loaded!".format(checkpoint_dir))

    def train(self, epoch, df_name):
        # 一个epoch的训练
        self.bert_model.train()
        self.iteration(epoch,  self.train_data_loader, train=True, df_name=df_name)

    def valid(self, epoch, df_name):
        # 一个epoch的测试, 并返回测试集的auc
        self.bert_model.eval()
        with torch.no_grad():
            return self.iteration(epoch, self.valid_data_loader, train=False, df_name=df_name)

    def iteration(self, epoch, data_loader, train=True, df_name="df_log.csv"):
        # 初始化一个pandas DataFrame进行训练日志的存储
        if not os.path.isfile(df_name):
            df = pd.DataFrame(columns=["epoch", "train_loss", "train_auc",
                                       "test_loss", "test_auc"
                                       ])
            df.to_csv(df_name, index=False)
            print("log DataFrame created!")

        # 进度条显示
        str_code = "train" if train else "valid"
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

        total_loss = 0
        # 存储所有预测的结果和标记, 用来计算auc
        all_predictions, all_labels = [], []

        for i, data in data_iter:
            input_ids, segment_ids, output_ids, label = data
            input_ids = input_ids.to(self.device)
            label = label.to(self.device)

            batch_size, seq_len = input_ids.shape

            # 生成位置id
            position_ids = torch.arange(0, seq_len * batch_size, dtype=torch.int32)
            position_ids = position_ids.view(batch_size, seq_len)
            position_ids = torch.fmod(position_ids, seq_len)
            position_ids = position_ids.to(self.device)

            # 正向传播, 得到预测结果和loss
            predictions, loss = self.bert_model.forward(text_input=input_ids,
                                                        position_ids=position_ids,
                                                        labels=label,
                                                        )
            # 提取预测的结果和标记, 并存到all_predictions, all_labels里
            # 用来计算auc
            predictions = predictions.detach().cpu().numpy().reshape(-1).tolist()
            labels = label.cpu().numpy().reshape(-1).tolist()
            all_predictions.extend(predictions)
            all_labels.extend(labels)
            # 计算auc
            fpr, tpr, thresholds = metrics.roc_curve(y_true=all_labels,
                                                     y_score=all_predictions)
            auc = metrics.auc(fpr, tpr)

            # 反向传播
            if train:
                # 清空之前的梯度
                self.optimizer.zero_grad()
                # 反向传播, 获取新的梯度
                loss.backward()
                # 用获取的梯度更新模型参数
                self.optimizer.step()

            # 为计算当前epoch的平均loss
            total_loss += loss.item()

            if train:
                log_dic = {
                    "epoch": epoch,
                    "train_loss": total_loss/(i+1), "train_auc": auc,
                    "test_loss": 0, "test_auc": 0
                }

            else:
                log_dic = {
                    "epoch": epoch,
                    "test_loss": total_loss/(i+1), "test_auc": auc
                }
            if train:
                data_iter.set_postfix(loss=log_dic['train_loss'], auc=log_dic['train_auc'],
                                      lr=self.optimizer.param_groups[0]['lr'])
            else:
                data_iter.set_postfix(loss=log_dic['test_loss'], auc=log_dic['test_auc'],
                                      lr=self.optimizer.param_groups[0]['lr'])
            data_iter.update(1)

        # threshold_ = find_best_threshold(all_predictions, all_labels)
        # print(str_code + " best threshold: " + str(threshold_))

        # 将当前epoch的情况记录到DataFrame里
        if train:
            df = pd.read_csv(df_name)
            log_dic_df = pd.DataFrame([log_dic.values()], columns=log_dic.keys())
            df = pd.concat([df, log_dic_df], axis=0)
            df.reset_index(inplace=True, drop=True)
            df.to_csv(df_name, index=False)

        else:
            df = pd.read_csv(df_name)
            epoch = log_dic['epoch']
            del log_dic['epoch']
            for key in log_dic:
                df.loc[df.epoch == epoch, key] = log_dic[key]
            df.reset_index(inplace=True, drop=True)
            df.to_csv(df_name, index=False)
            # 返回auc, 作为early stop的衡量标准
            return auc

    def find_most_recent_state_dict(self, dir_path):
        """
        :param dir_path: 存储所有模型文件的目录
        :return: 返回最新的模型文件路径, 按模型名称最后一位数进行排序
        """
        dic_lis = [i for i in os.listdir(dir_path)]
        dic_lis = [i for i in dic_lis if "model" in i]
        if len(dic_lis) == 0:
            return None
        dic_lis = sorted(dic_lis, key=lambda k: int(k.split(".")[-1]))
        return dir_path + "/" + dic_lis[-1]

    def save_state_dict(self, model, epoch, state_dict_dir="../output", file_path="bert.model"):
        """存储当前模型参数"""
        if not os.path.exists(state_dict_dir):
            os.mkdir(state_dict_dir)
        save_path = state_dict_dir + "/" + file_path + ".epoch.{}".format(str(epoch))
        model.to("cpu")
        torch.save({"model_state_dict": model.state_dict()}, save_path)
        print("{} saved!".format(save_path))
        model.to(self.device)


def init_trainer(dynamic_lr, batch_size=32):
    train_data, valid_data, test_data = preprocess_data()
    tokens = get_tokens(train_data + valid_data + test_data)
    # train_data = train_data[:10]
    # valid_data = valid_data[:10]
    train_data_loader = get_data_loader(data=train_data, tokens=tokens, random=False, batch_size=batch_size,
                                        num_workers=2)
    valid_data_loader = get_data_loader(data=valid_data, tokens=tokens, random=False, batch_size=batch_size,
                                        num_workers=2)

    bert_config = BertConfig(**config['bert_config'])
    trainer = SimilarityTrainer(vocab_size=config['bert_config']['vocab_size'],
                                bert_config=bert_config,
                                batch_size=batch_size,
                                lr=dynamic_lr,
                                with_cuda=True,
                                train_data_loader=train_data_loader,
                                valid_data_loader=valid_data_loader
                                )
    return trainer


def train():
    lr = float(config['simtrain']['lr'])
    trainer = init_trainer(lr, batch_size=config['simtrain']['batch_size'])
    bert_model_path = os.path.join(config['output_path'], 'bert')
    sim_model_path = config['sim_model_path']
    trainer.load_bert_model(trainer.bert_model, dir_path=bert_model_path)

    start_epoch = trainer.load_model(trainer.bert_model, dir_path=sim_model_path)

    start_time = datetime.datetime.now()
    print("training start at {}".format(start_time))
    for epoch in range(start_epoch, config['simtrain']['epochs']+1):
        print("train with learning rate {}".format(lr))
        trainer.train(epoch, df_name=config['sim_train_log_dir'])
        trainer.valid(epoch, df_name=config['sim_train_log_dir'])
        if epoch % config['simtrain']['model_save_epoch'] == 0:
            trainer.save_state_dict(trainer.bert_model, epoch, state_dict_dir=config['sim_model_path'],
                                    file_path="sim.model")
        # break
    end_time = datetime.datetime.now()
    print("training finished at {} used{}".format(end_time, end_time-start_time))


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


if __name__ == '__main__':
    train()
