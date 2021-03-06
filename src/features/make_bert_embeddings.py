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

    if args.path_emb_dir:
        path_emb_dir = Path(args.path_emb_dir)
    else:
        path_emb_dir = proj_dir / "data" / "processed" / "embeddings"

    path_emb_dir.mkdir(parents=True, exist_ok=True)
    
    if args.path_label_dir:
        path_label_dir = Path(args.path_label_dir)
    else:
        path_label_dir = proj_dir / "data" / "processed" / "labels" / "labels_complete"

    return proj_dir, path_emb_dir, path_label_dir


def create_single_embedding(text, model, tokenizer, device=None, max_len=512):
    """Create a single embedding for a given text.
    
    Parameters
    ----------
    text : str
        Text to create embedding for.
    model : BertModel
        Bert model to use. The model should already be sent to the appropriate device.
    tokenizer : BertTokenizer
        Bert tokenizer to use.


    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # lowercase text
    text = text.lower()

    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        max_length=512,
        return_token_type_ids=False,
        padding="max_length",
        return_attention_mask=True,
        truncation=True,
        return_tensors="pt",  # Return PyTorch tensors
    )

    input_ids = encoding["input_ids"]
    attention_masks = encoding["attention_mask"]

    with torch.no_grad():
        # from https://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/
        last_hidden_states = model(
            input_ids=input_ids.to(device),
            attention_mask=attention_masks.to(device)
        )

        # features =last_hidden_states[0][:, 0, :].cpu().numpy().flatten()
        features =last_hidden_states[0][:, 0, :].cpu().numpy()

    return features


def create_batch_embeddings(df, model, tokenizer, device=None, max_len=512, batch_size=20, label_column="label"):
    """Create batch embeddings for a given dataframe.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe with text and labels.
    model : BertModel
        Model to use for embedding.
    tokenizer : BertTokenizer
        Tokenizer to use for embedding.
    data_loader : DataLoader
        Pytorch DataLoader to get the text from the df.
    device : torch.device, optional
        Device to use for embedding. If "none" is specified, the system will look for a GPU, else will use the CPU.
    max_len : int, optional
        Maximum length of the text. The default is 512.
    batch_size : int, optional
        Batch size for embedding. The default is 20.
    label_column : str, optional
        Column name for the label. The default is "label".
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create copy of df
    df_copy = df.copy()

    df_copy["para"] = df_copy["para"].str.lower()
    df_copy["label"] = df_copy["label"].apply(lambda x: 1 if x > 0 else 0)  # binary labels

    # loop through label data and create embeddings
    data_loader = create_data_loader(df_copy, tokenizer, max_len, batch_size)

    features_list = []
    for i, data in enumerate(data_loader):
        if i % 5 == 0:
            print(f"No. para: {batch_size * i}")

        with torch.no_grad():
            # print(data["input_ids"].shape)
            # from https://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/
            last_hidden_states = model(
                data["input_ids"].to(device),
                attention_mask=data["attention_masks"].to(device),
            )

            features = (
                last_hidden_states[0][:, 0, :].cpu().numpy()
            )

            features_list.append(features)  

    return np.vstack(features_list)


def main(args):

    # set directories
    proj_dir, path_emb_dir, path_label_dir = set_directories(args)

    # set file names
    label_file_name = args.label_file_name
    emb_file_name = args.emb_file_name

    # load label data
    # if csv file
    if label_file_name.endswith(".csv"):
        df = pd.read_csv(path_label_dir / label_file_name, dtype={"id": str})
    elif label_file_name.endswith(".ods"):
        df = pd.read_excel(
            path_label_dir / label_file_name,
            parse_dates=["update_date"],
            engine="odf",
            dtype={"id": str},
            )
    else:
        raise ValueError("label file name must end with .csv or .ods")
        
    # df["para"] = df["para"].str.lower()
    df["label"] = df["label"].apply(lambda x: 1 if x > 0 else 0)  # binary labels

    # load bert tokenizer and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    print("device name:", device)

    scratch_path = Path.home() / "scratch"
    if scratch_path.exists():
        tokenizer = BertTokenizer.from_pretrained(proj_dir / "bert_cache_dir")
        model = AutoModel.from_pretrained(proj_dir / "bert_cache_dir")
    else:
        tokenizer = BertTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
        model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")

    model.to(device)

    features_array = create_batch_embeddings(df, model, tokenizer, device=device, max_len=512, batch_size=20, label_column="label")

    df["h"] = features_array.tolist()

    df = df[["id", "label", "para", "h"]]

    # save dfh as a pickle file
    with open(path_emb_dir / emb_file_name, "wb") as f:
        pickle.dump(df, f)


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
        "--path_emb_dir",
        type=str,
        help="Path to the folder that contains all the embedding pickle files",
    )

    parser.add_argument(
        "--path_label_dir",
        type=str,
        help="Path to the folder that contains all the individual label files (csv or ods)",
    )

    parser.add_argument(
        "--emb_file_name",
        type=str,
        default="df_embeddings.pkl",
        help="Name of the embedding file to save",
    )

    parser.add_argument(
        "--label_file_name",
        type=str,
        help="Name of the label file (containing paragrapsh) used to create the embeddings",
    )

    args = parser.parse_args()

    main(args)
