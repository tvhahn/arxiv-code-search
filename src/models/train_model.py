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
import argparse
import shutil
import logging
import re
import argparse
import datetime


def find_most_recent_checkpoint(path_prev_checkpoint):
    """Finds the most recent checkpoint in a checkpoint folder
    and returns the path to that .pt file.
    """

    ckpt_list = list(path_prev_checkpoint.rglob("*.pt"))
    max_epoch = sorted(list(int(re.findall("[0-9]+", str(i))[-1]) for i in ckpt_list))[
        -1
    ]
    return Path(path_prev_checkpoint / f"train_{max_epoch}.pt")


def set_directories(args):

    if args.proj_dir:
        proj_dir = Path(args.proj_dir)
    else:
        # proj_dir assumed to be cwd
        proj_dir = Path.cwd()

    # check if "scratch" path exists in the home directory
    # if it does, assume we are on HPC
    scratch_path = Path.home() / "scratch"

    if args.path_data_dir:
        path_data_dir = Path(args.path_data_dir)
    elif scratch_path.exists():
        path_data_dir = scratch_path / "arxiv-code-search/data"
    else:
        path_data_dir = proj_dir / "data"

    if args.path_model_dir:
        path_model_dir = Path(args.path_model_dir)
    elif scratch_path.exists():
        path_model_dir = scratch_path / "arxiv-code-search/models"
    else:
        path_model_dir = proj_dir / "models"

    Path(path_model_dir).mkdir(parents=True, exist_ok=True)

    if args.path_label_dir:
        path_label_dir = Path(args.path_label_dir)
    else:  # default to proj_dir
        path_label_dir = proj_dir / "processed" / "labels" / "labels_complete"

    Path(path_label_dir).mkdir(parents=True, exist_ok=True)

    # set time
    if args.model_time_suffix:
        model_start_time = (
            datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S")
            + "_"
            + args.model_time_suffix
        )
    else:
        model_start_time = datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S")

    # set checkpoint directory
    # if loading the model from a checkpoint, a checkpoint folder name
    # should be passed as an argument, like: -c 2021_07_14_185903
    # the various .pt files will be inside the checkpoint folder
    # and the latest one will be loaded
    if args.ckpt_name:
        prev_checkpoint_dir_name = args.ckpt_name
        path_prev_checkpoint_dir = (
            path_model_dir / "interim/checkpoints" / prev_checkpoint_dir_name
        )
        if Path(path_prev_checkpoint_dir).exists():
            print("Previous checkpoints exist. Training from most recent checkpoint.")

            path_prev_checkpoint_dir = find_most_recent_checkpoint(
                path_prev_checkpoint_dir
            )

        else:
            print("Could not find previous checkpoint folder. Training from beginning.")
    else:
        # set dummy name for path_prev_checkpoint
        path_prev_checkpoint_dir = Path("no_prev_checkpoint_needed")

    path_checkpoint_dir = path_model_dir / "interim/checkpoints" / model_start_time

    Path(path_checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # save src directory as a zip into the checkpoint folder
    shutil.make_archive(
        path_checkpoint_dir / f"src_files_{model_start_time}",
        "zip",
        proj_dir / "src",
    )

    return (
        path_data_dir,
        path_model_dir,
        path_label_dir,
        path_checkpoint_dir,
        path_prev_checkpoint_dir,
        model_start_time,
    )


def save_checkpoint(epoch, path_checkpoint_folder, gen, critic, opt_gen, opt_critic):
    torch.save(
        {
            "gen": gen.state_dict(),
            "critic": critic.state_dict(),
            "opt_gen": opt_gen.state_dict(),
            "opt_critic": opt_critic.state_dict(),
            "epoch": epoch,
        },
        path_checkpoint_folder / f"train_{epoch}.pt",
    )


def main(args):
    # set directories
    (
        path_data_dir,
        path_model_dir,
        path_label_dir,
        path_checkpoint_dir,
        path_prev_checkpoint_dir,
        model_start_time,
    ) = set_directories(args)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--path_data_dir", dest="path_data_dir", type=str, help="Path to the data (contains the raw/interim/processed folders)"
    )

    parser.add_argument(
        "-p",
        "--proj_dir",
        dest="proj_dir",
        type=str,
        help="Location of project folder",
    )

    parser.add_argument(
        "--path_model_dir",
        dest="path_model_dir",
        type=str,
        help="Location of models folder (where interim/final models are kept).",
    )

    parser.add_argument(
        "--path_label_dir",
        dest="path_label_dir",
        type=str,
        help="Location of label folder (where final labels.csv is kept).",
    )

    parser.add_argument(
        "-c",
        "--checkpoint",
        dest="ckpt_name",
        type=str,
        help="Name of chekpoint folder to load previous checkpoint from",
    )

    parser.add_argument(
        "--model_time_suffix",
        dest="model_time_suffix",
        type=str,
        help="Optional suffix string to append at the end of the model start time identifier",
    )

    parser.add_argument(
        "--batch_size",
        dest="batch_size",
        type=int,
        default=1,
        help="Mini-batch size for each GPU",
    )

    parser.add_argument(
        "--learning_rate",
        dest="learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate for optimizer",
    )

    parser.add_argument(
        "--n_epochs",
        dest="n_epochs",
        type=int,
        default=500,
        help="Number of epochs",
    )

    args = parser.parse_args()

    main(args)
