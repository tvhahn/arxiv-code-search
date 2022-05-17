import torch
from transformers import BertModel
from torch import nn


class ArxivClassifier(nn.Module):
    def __init__(self, n_classes, pre_trained_model_name):
        super(ArxivClassifier, self).__init__()
        self.encoder = BertModel.from_pretrained(pre_trained_model_name)
        self.dense_1 = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.5)
        self.dense_2 = torch.nn.Linear(768, n_classes)

    def forward(self, input_ids, attention_mask):
        output_1 = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooled_output = hidden_state[:, 0]
        pooled_output = self.dense_1(pooled_output)
        pooled_output = torch.nn.ReLU()(pooled_output)
        # pooled_output = self.dense_1(pooled_output)
        # pooled_output = torch.nn.ReLU()(pooled_output)
        pooled_output = self.dropout(pooled_output)
        output = self.dense_2(pooled_output)
        return output
