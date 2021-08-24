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


class SimInference(object):

    def __init__(self,
                 bert_config,
                 max_seq_len,
                 word2idx,
                 test_data,
                 num_workers=1,
                 with_cuda=True,
                 ):
        self.bert_config = bert_config
        self.max_seq_len = max_seq_len
        # 判断是否有可用GPU
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")

        test_dataset = BERTDataSet(data=test_data,
                                   word2idx=word2idx,
                                   random=False
                                   )
        self.test_dataloader = DataLoader(test_dataset,
                                          batch_size=64,
                                          num_workers=num_workers,
                                          collate_fn=collate_fn)
        self.model = SimilarSentenceModel(self.bert_config)
        print(self.model)

        # 初始化位置编码
        self.hidden_dim = bert_config.hidden_size
        self.positional_enc = self.init_positional_encoding()
        # 扩展位置编码的维度, 留出batch维度,
        # 即positional_enc: [batch_size, embedding_dimension]
        self.positional_enc = torch.unsqueeze(self.positional_enc, dim=0)

    def init_positional_encoding(self):
        position_enc = np.array([
            [pos / np.power(10000, 2 * i / self.hidden_dim) for i in range(self.hidden_dim)]
            if pos != 0 else np.zeros(self.hidden_dim) for pos in range(self.max_seq_len)])

        position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
        position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i+1
        denominator = np.sqrt(np.sum(position_enc**2, axis=1, keepdims=True))
        # 归一化
        position_enc = position_enc / (denominator + 1e-8)
        position_enc = torch.from_numpy(position_enc).type(torch.FloatTensor)
        return position_enc

    def find_most_recent_state_dict(self, dir_path):
        """
        :param dir_path: 存储所有模型文件的目录
        :return: 返回最新的模型文件路径, 按模型名称最后一位数进行排序
        """
        dic_lis = [i for i in os.listdir(dir_path)]
        if len(dic_lis) == 0:
            return None
        dic_lis = [i for i in dic_lis if "model" in i]
        dic_lis = sorted(dic_lis, key=lambda k: int(k.split(".")[-1]))
        return dir_path + "/" + dic_lis[-1]

    def load_model(self, model, dir_path="../../user_data/models/bert"):
        checkpoint_dir = self.find_most_recent_state_dict(dir_path)
        if not checkpoint_dir:
            raise RuntimeError("not found any model, please train a model first!")
        checkpoint = torch.load(checkpoint_dir)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        torch.cuda.empty_cache()
        model.to(self.device)
        print("sim model: {} loaded!".format(checkpoint_dir))

    def iteration(self):
        # 进度条显示
        data_iter = tqdm.tqdm(enumerate(self.test_dataloader),
                              desc="prediction",
                              total=len(self.test_dataloader),
                              bar_format="{l_bar}{r_bar}")

        # 存储所有预测的结果和标记, 用来计算auc
        # all_predictions = []
        for i, data in data_iter:
            input_ids, segment_ids, output_ids, label = data
            batch_size, seq_len = input_ids.shape

            input_ids = input_ids.to(self.device)

            # 生成位置id
            position_ids = torch.arange(0, seq_len * batch_size, dtype=torch.int32)
            position_ids = position_ids.view(batch_size, seq_len)
            position_ids = torch.fmod(position_ids, seq_len)
            position_ids = position_ids.to(self.device)

            # 正向传播, 得到预测结果
            predictions = self.model.forward(text_input=input_ids,
                                             position_ids=position_ids
                                             )
            # 提取预测的结果存到all_predictions
            predictions = predictions.detach().cpu().numpy().reshape(-1).tolist()
            # if i % 20 == 0:
            #     print(predictions)
            yield predictions

    def predict(self, model_path, save_to):
        self.load_model(self.model, model_path)
        all_predictions = self.iteration()
        # print(all_predictions)
        with open(save_to, 'w') as f:
            for pre in all_predictions:
                for p in pre:
                    f.write("{}\n".format(p))
        print("save prediction result to {}".format(save_to))
        return all_predictions


def predict():
    bert_config = BertConfig(**config['bert_config'])
    train_data, valid_data, test_data = preprocess_data()
    word2idx = get_tokens(train_data + valid_data + test_data)
    # test_data = test_data[:128]
    inf = SimInference(bert_config,
                       max_seq_len=config['seq_max_len'],
                       test_data=test_data,
                       word2idx=word2idx,
                       num_workers=1)

    inf.predict(model_path=config['sim_model_path'], save_to=config['prediction_result'])


if __name__ == '__main__':
    predict()
