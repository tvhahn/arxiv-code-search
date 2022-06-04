import torch
# from transformers import (
#     BertModel,
#     BertTokenizer,
#     AdamW,
#     get_linear_schedule_with_warmup,
#     BertForSequenceClassification,
# )

from torch.utils.data import Dataset, DataLoader
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


# create data loader -- inspired by https://curiousily.com/posts/sentiment-analysis-with-bert-and-hugging-face-using-pytorch-and-python/
class ArxivDataset(Dataset):
    def __init__(self, texts, labels, ids, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.ids = ids
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text_orig = str(self.texts[item])
        text = str(self.texts[item]).lower()
        id = str(self.ids[item])
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
            'texts_orig': text_orig,
            "ids": id,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_masks": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def create_data_loader(df, tokenizer, max_len, batch_size, label_column="label"):
    ds = ArxivDataset(
        texts=df.para.to_numpy(),
        labels=df[label_column].to_numpy(),
        ids=df.id.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len,
    )

    return DataLoader(ds, batch_size=batch_size, num_workers=4)


def under_over_sampler(x, y, method=None, ratio=0.5):
    """
    Returns an undersampled or oversampled data set. Implemented using imbalanced-learn package.
    ['random_over','random_under']
    """

    if method == None:
        return x, y

    # oversample methods: https://imbalanced-learn.readthedocs.io/en/stable/over_sampling.html
    elif method == "random_over":
        # print('before:',sorted(Counter(y).items()))
        ros = RandomOverSampler(sampling_strategy=ratio, random_state=0)
        x_resampled, y_resampled = ros.fit_resample(x, y)
        # print('after:',sorted(Counter(y_resampled).items()))
        return x_resampled, y_resampled

    elif method == "random_under":
        rus = RandomUnderSampler(sampling_strategy=ratio, random_state=0)
        x_resampled, y_resampled = rus.fit_resample(x, y)
        return x_resampled, y_resampled

    else:
        return x, y


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
