from torch_implement.bert_model import *
import torch
from torch import nn


class BertPreTrainingHeadsMLM(nn.Module):
    """
    BERT的训练中通过隐藏层输出Masked LM的预测和Next Sentence的预测
    """

    def __init__(self, config, bert_model_embedding_weights):
        super(BertPreTrainingHeadsMLM, self).__init__()

        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)
        # 把transformer block输出的[batch_size, seq_len, embed_dim]
        # 映射为[batch_size, seq_len, vocab_size]
        # 用来进行MaskedLM的预测

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertModelMLM(BertPreTrainedModel):

    def __init__(self, config):
        super(BertModelMLM, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, position_ids=None, token_type_ids=None, attention_mask=None,
                output_all_encoded_layers=True, get_attention_matrices=False):
        if attention_mask is None:
            attention_mask = (input_ids > 0)
            # attention_mask [batch_size, length]
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, position_ids, token_type_ids)
        # 经过所有定义的transformer block之后的输出
        encoded_layers, all_attention_matrices = self.encoder(embedding_output,
                                                              extended_attention_mask,
                                                              output_all_encoded_layers=output_all_encoded_layers,
                                                              get_attention_matrices=get_attention_matrices)
        # 可输出所有层的注意力矩阵用于可视化
        if get_attention_matrices:
            return all_attention_matrices
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers


class MyBertForPreTraining(BertPreTrainedModel):

    def __init__(self, config):
        super(MyBertForPreTraining, self).__init__(config)
        self.bert = BertModelMLM(config)
        self.cls = BertPreTrainingHeadsMLM(config, self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)
        self.vocab_size = config.vocab_size
        self.mlm_loss_func = CrossEntropyLoss(ignore_index=0)

    def compute_loss(self, predictions, labels, num_class=2, ignore_index=-100):
        loss_func = CrossEntropyLoss(ignore_index=ignore_index)
        return loss_func(predictions.view(-1, num_class), labels.view(-1))

    def forward(self, input_tensor):
        """
        :param input_tensor: dim: [batch_size, 3, seq_len] 3对应input_ids,segment_ids,position_ids
        :return: mlm_preds: dim: [batch_size, seq_len]
        """
        batch_size, _, seq_len = input_tensor.shape
        input_ids = input_tensor[:, 0, :].view(batch_size, seq_len)
        token_type_ids = input_tensor[:, 1, :].view(batch_size, seq_len)
        position_ids = input_tensor[:, 2, :].view(batch_size, seq_len)
        sequence_output = self.bert(input_ids, position_ids, token_type_ids, attention_mask=None,
                                    output_all_encoded_layers=False)
        mlm_preds = self.cls(sequence_output)
        return mlm_preds


class SimilarSentenceModel(nn.Module):

    def __init__(self, config):
        super(SimilarSentenceModel, self).__init__()
        self.bert = BertModelMLM(config)
        self.dense = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.final_dense = nn.Linear(config.hidden_size, 1)
        self.activation = nn.Sigmoid()
        # self.loss_function = FocalLoss(class_num=2)

    def compute_loss(self, predictions, labels):
        # 将预测和标记的维度展平, 防止出现维度不一致
        predictions = predictions.view(-1)
        labels = labels.float().view(-1)
        epsilon = 1e-8
        # 交叉熵
        loss =\
            - labels * torch.log(predictions + epsilon) - \
            (torch.tensor(1.0) - labels) * torch.log(torch.tensor(1.0) - predictions + epsilon)
        loss = torch.mean(loss)
        return loss

    def forward(self, text_input, position_ids, labels=None):
        encoded_layers = self.bert(text_input, position_ids=position_ids, output_all_encoded_layers=True)
        sequence_output = encoded_layers[-1]
        # # sequence_output的维度是[batch_size, seq_len, embed_dim]
        avg_pooled = sequence_output.mean(1)
        max_pooled = torch.max(sequence_output, dim=1)
        pooled = torch.cat((avg_pooled, max_pooled[0]), dim=1)
        pooled = self.dense(pooled)

        # 下面是[batch_size, hidden_dim * 2] 到 [batch_size, 1]的映射
        # 我们在这里要解决的是二分类问题

        predictions = self.final_dense(pooled)

        # 用sigmoid函数做激活, 返回0-1之间的值
        predictions = self.activation(predictions)
        if labels is not None:
            # 计算loss
            loss = self.compute_loss(predictions, labels)
            return predictions, loss
        else:
            return predictions
