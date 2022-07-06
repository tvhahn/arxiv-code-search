import pandas as pd
from pathlib import Path
import numpy as np
import re
import os
import argparse
import logging
from datetime import datetime
import nltk
import shutil
import torch
from transformers import (
    BertModel,
    BertTokenizer,
    AutoModel)
from src.features.make_bert_embeddings import (create_batch_embeddings)
import pickle


def set_directories(args):

    if args.proj_dir:
        proj_dir = Path(args.proj_dir)
    else:
        proj_dir = Path().cwd()

    if args.path_trained_model_dir:
        path_trained_model_dir = Path(args.path_trained_model_dir)
    else:
        path_trained_model_dir = proj_dir / "models" /" final_results_classical" / "model_files"

    if args.path_label_dir:
        path_label_dir = Path(args.path_label_dir)
    else:
        path_label_dir = proj_dir / "data" / "interim"

    return proj_dir, path_trained_model_dir, path_label_dir




def main(args):
    """Add predictions and probabilities to a label file. This uses an already
    trained sklearn model.
    """
    logger = logging.getLogger(__name__)
    logger.info("Add probabilities to label file.")

    proj_dir, path_trained_model_dir, path_label_dir = set_directories(args)

    # label file name
    label_file_name = args.label_file_name

    # load label data
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


    # load bert tokenizer and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device name:", device)

    scratch_path = Path.home() / "scratch"
    if scratch_path.exists():
        tokenizer = BertTokenizer.from_pretrained(proj_dir / "bert_cache_dir")
        bert_model = AutoModel.from_pretrained(proj_dir / "bert_cache_dir")
    else:
        tokenizer = BertTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
        bert_model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")

    bert_model.to(device)

    features = create_batch_embeddings(df, bert_model, tokenizer, device)

    # scale and get predictions/probabilities
    model_name = args.model_name
    scaler_name = args.scaler_name

    # load sklearn scaler from scaler file
    with open(path_trained_model_dir / scaler_name, "rb") as f:
        scaler = pickle.load(f)

    # load the model
    with open(path_trained_model_dir / model_name, "rb") as f:
        model = pickle.load(f)

    features = scaler.transform(features)
    probabilities = model.predict_proba(features)
    predictions = model.predict(features)

    # create new column with predictions
    df["pred"] = predictions

    # create new columns with probabilities, one for each probability
    for i, col in enumerate(model.classes_):
        df[f"proba_{col}"] = probabilities[:, i]

    # save the dataframe as either a csv or ods file
    # depending on the file name extension
    if args.label_file_save_name:
        label_file_save_name = args.label_file_save_name
    else:
        label_file_save_name = label_file_name

    if label_file_name.endswith(".csv"):
        df.to_csv(path_label_dir / label_file_save_name, index=False)
    else:
        # save df as an .ods file
        df.to_excel(
            path_label_dir / label_file_save_name,
            engine="odf",
            index=False,
            )




if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(description="Create .txt files from .pdf files")

    parser.add_argument(
        "-p",
        "--proj_dir",
        dest="proj_dir",
        type=str,
        help="Location of project folder",
    )

    # argument for txt_dir_path
    parser.add_argument(
        "--path_trained_model_dir",
        type=str,
        help="Path to the folder that contains all the trained classical models.",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        help="File name of the sklearn model to be used for prediction.",
    )

    parser.add_argument(
        "--scaler_name",
        type=str,
        help="File name of the sklearn scaler to be used for scaling.",
    )

    parser.add_argument(
        "--path_label_dir",
        type=str,
        help="Path to the folder that contains all the individual label files (csv or ods)",
    )


    parser.add_argument(
        "--label_file_name",
        type=str,
        help="Name of the label file (containing paragrapsh) used to create the embeddings",
    )

    parser.add_argument(
        "--label_file_save_name",
        type=str,
        help="New name of the label file with the predictions and probabilities added",
    )

    args = parser.parse_args()

    main(args)
