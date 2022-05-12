import torch
from transformers import (
    BertModel,
    BertTokenizer,
    AdamW,
    get_linear_schedule_with_warmup,
    BertForSequenceClassification,
)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pathlib import Path
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch import nn, optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from torchmetrics import PrecisionRecallCurve


# create data loader -- inspired by https://curiousily.com/posts/sentiment-analysis-with-bert-and-hugging-face-using-pytorch-and-python/
class ArxivDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item]).lower()
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=self.max_len,
            return_token_type_ids=False,
            padding="max_length",
            return_attention_mask=True,
            truncation=True,
            return_tensors="pt",  # Return PyTorch tensors
        )

        return {
            "texts": text,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = ArxivDataset(
        texts=df.para.to_numpy(),
        labels=df.label.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len,
    )

    return DataLoader(ds, batch_size=batch_size, num_workers=4)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(
        self,
        patience=7,
        verbose=False,
        delta=0.005,
        path="checkpoint.pt",
        trace_func=print,
        early_stop_delay=20,
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print

        From https://github.com/Bjarten/early-stopping-pytorch
        License: MIT
        """
        self.patience = patience
        self.verbose = verbose
        self.early_stop_delay = early_stop_delay
        self.counter = 0
        self.epoch = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.epoch < self.early_stop_delay:
            self.epoch += 1
            pass
        else:
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
            elif score < self.best_score + self.delta:
                self.counter += 1
                if self.verbose:
                    self.trace_func(
                        f"EarlyStopping counter: {self.counter} out of {self.patience}"
                    )
                if self.counter >= self.patience:
                    self.early_stop = True
            elif torch.isnan(torch.tensor(score)).item():
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
                # print('########## IS NAN #######')
            else:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
                self.counter = 0
            self.epoch += 1


    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save({"model": model.state_dict()}, self.path)
        # torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
