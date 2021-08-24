from kaggle.pre_train import preprocess_data
from transformers import BertForMaskedLM, BertConfig, TrainingArguments, Trainer
from torch import nn
import torch
import torch.nn.functional as F
from kaggle.pre_train import convert_data_to_sentence_pair
from torch.utils.data.dataset import Dataset
from transformers import BertTokenizer, PreTrainedTokenizer
import os
import torch


class LineByLineTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int):

        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        print("Creating features from dataset file at %s", file_path)

        lines = []
        labels = []
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                if len(line) <= 0 or line.isspace():
                    continue
                line = line.split('\t')
                lines.append([line[0], line[1]])
                labels.append([line[2]])

        batch_encoding = tokenizer(lines, add_special_tokens=True, truncation=True, max_length=block_size)
        self.examples = batch_encoding["input_ids"]
        token_type_ids = batch_encoding["token_type_ids"]
        self.examples = [{
            "input_ids": torch.tensor(e, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids[i], dtype=torch.long),
            "labels": labels[i]} for i, e in
                         enumerate(self.examples)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = F.gelu
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertPredictionModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, 1, bias=True)

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = F.sigmoid(hidden_states)
        return hidden_states


class PredictionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertPredictionModel(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class SimilarityModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.bert = BertForMaskedLM(config)
        self.cls = PredictionModel(config)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
    ):
        output = self.bert(input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds,
                           encoder_hidden_states, encoder_attention_mask, labels=None, output_attentions=False,
                           output_hidden_states=True, return_dict=True)
        hidden_states = output.hidden_states
        res = self.cls(hidden_states[-1, 0])
        return res


def get_trainer(model, dataset):
    training_args = TrainingArguments(
        output_dir='../../user_data/models/kaggle_sim/',
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
        train_dataset=dataset,
    )
    return trainer


def fine_tune():
    train_data, valid_data, test_data = preprocess_data()
    vocab_file = '../../user_data/vocab.txt'
    tokenizer = BertTokenizer.from_pretrained(vocab_file)
    train_file = convert_data_to_sentence_pair(train_data)
    dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=train_file,
        block_size=32  # maximum sequence length
    )
    print('No. of lines: ', len(dataset))  # No of lines in your datset

    #
    config = BertConfig(
        vocab_size=8000,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        max_position_embeddings=512
    )
    bert_model = SimilarityModel(config)
    #
    trainer = get_trainer(bert_model, dataset)
    trainer.train()
    trainer.save_model(output_dir='../../user_data/models/kaggle_sim/')


if __name__ == '__main__':
    fine_tune()


