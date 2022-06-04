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
from transformers import AutoModel
import pickle
import argparse
from src.models.utils import create_data_loader


def set_directories(args):

    if args.proj_dir:
        proj_dir = Path(args.proj_dir)
    else:
        proj_dir = Path().cwd()

    if args.path_data_dir:
        path_data_dir = Path(args.path_data_dir)
    else:
        path_data_dir = proj_dir / "data"

    return proj_dir, path_data_dir


def main(args):

    # set directories
    proj_dir, path_data_dir = set_directories(args)
    path_label_dir = path_data_dir / "processed/labels/labels_complete"
    embedding_dir = path_data_dir / "processed/embeddings"
    embedding_dir.mkdir(parents=True, exist_ok=True)

    # load label data
    df = pd.read_csv(path_label_dir / "labels.csv", dtype={"id": str})
    df["para"] = df["para"].str.lower()
    df["label"] = df["label"].apply(lambda x: 1 if x > 0 else 0)  # binary labels

    # load bert tokenizer and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device name:", device)

    scratch_path = Path.home() / "scratch"
    if scratch_path.exists():
        tokenizer = BertTokenizer.from_pretrained(proj_dir / "bert_cache_dir")
        model = AutoModel.from_pretrained(proj_dir / "bert_cache_dir")
    else:
        tokenizer = BertTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
        model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")

    model.to(device)

    # loop through label data and create embeddings
    data_loader = create_data_loader(df, tokenizer, 512, 20)

    dfh_list = []
    for i, data in enumerate(data_loader):

        labels = data["labels"]
        with torch.no_grad():
            # from https://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/
            last_hidden_states = model(
                data["input_ids"].to(device),
                attention_mask=data["attention_masks"].to(device),
            )

            features = (
                last_hidden_states[0][:, 0, :].cpu().numpy()
            )  

            df_h = pd.DataFrame(labels, columns=["label"])
            df_h["id"] = data["ids"]
            df_h["para"] = data["texts_orig"]
            df_h["h"] = features.tolist()
            df_h["h"] = df_h["h"].apply(lambda x: np.array(x))
            dfh_list.append(df_h)

        if i % 5 == 0:
            print(i * 20)

    dfh = pd.concat(dfh_list)
    # save dfh as a pickle file
    with open(embedding_dir / "df_embeddings.pkl", "wb") as f:
        pickle.dump(dfh, f)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create embeddings from labels")

    parser.add_argument(
        "-p",
        "--proj_dir",
        dest="proj_dir",
        type=str,
        help="Location of project folder",
    )

    parser.add_argument(
        "--path_data_dir",
        dest="path_data_dir",
        type=str,
        help="Location of the data folder, containing the raw, interim, and processed folders",
    )

    args = parser.parse_args()

    main(args)
