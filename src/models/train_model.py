import torch
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup, BertForSequenceClassification
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


def find_most_recent_checkpoint(path_prev_checkpoint):
    """Finds the most recent checkpoint in a checkpoint folder
    and returns the path to that .pt file.
    """

    ckpt_list = list(path_prev_checkpoint.rglob("*.pt"))
    max_epoch = sorted(list(int(re.findall("[0-9]+", str(i))[-1]) for i in ckpt_list))[
        -1
    ]
    return Path(path_prev_checkpoint / f"train_{max_epoch}.pt")


def find_most_recent_checkpoint(path_prev_checkpoint):
    """Finds the most recent checkpoint in a checkpoint folder
    and returns the path to that .pt file.
    """

    ckpt_list = list(path_prev_checkpoint.rglob("*.pt"))
    max_epoch = sorted(list(int(re.findall("[0-9]+", str(i))[-1]) for i in ckpt_list))[
        -1
    ]
    return Path(path_prev_checkpoint / f"train_{max_epoch}.pt")


def set_directories():
    """Sets the directory paths used for data, checkpoints, etc."""

    # check if "scratch" path exists in the home directory
    # if it does, assume we are on HPC
    scratch_path = Path.home() / "scratch"
    if scratch_path.exists():
        print("Assume on HPC")
    else:
        print("Assume on local compute")

    path_processed_data = Path(args.path_data)

    # if loading the model from a checkpoint, a checkpoint folder name
    # should be passed as an argument, like: -c 2021_07_14_185903
    # the various .pt files will be inside the checkpoint folder
    if args.ckpt_name:
        prev_checkpoint_folder_name = args.ckpt_name
    else:
        # set dummy name for path_prev_checkpoint
        path_prev_checkpoint = Path("no_prev_checkpoint_needed")

    if args.proj_dir:
        proj_dir = Path(args.proj_dir)
    else:
        # proj_dir assumed to be cwd
        proj_dir = Path.cwd()

    # set time
    if args.model_time_suffix:
        model_start_time = datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S") + "_" + args.model_time_suffix
    else:
        model_start_time = datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S")

    if scratch_path.exists():
        # for HPC
        root_dir = scratch_path / "earth-mantle-surrogate"
        print(root_dir)

        if args.ckpt_name:
            path_prev_checkpoint = (
                root_dir / "models/interim/checkpoints" / prev_checkpoint_folder_name
            )
            if Path(path_prev_checkpoint).exists():
                print(
                    "Previous checkpoints exist. Training from most recent checkpoint."
                )

                path_prev_checkpoint = find_most_recent_checkpoint(path_prev_checkpoint)

            else:
                print(
                    "Could not find previous checkpoint folder. Training from beginning."
                )

        path_input_folder = path_processed_data / "input"
        path_truth_folder = path_processed_data / "truth"
        path_checkpoint_folder = (
            root_dir / "models/interim/checkpoints" / model_start_time
        )
        Path(path_checkpoint_folder).mkdir(parents=True, exist_ok=True)

    else:

        # for local compute
        root_dir = Path.cwd()  # set the root directory as a Pathlib path
        print(root_dir)

        if args.ckpt_name:
            path_prev_checkpoint = (
                root_dir / "models/interim/checkpoints" / prev_checkpoint_folder_name
            )
            if Path(path_prev_checkpoint).exists():
                print(
                    "Previous checkpoints exist. Training from most recent checkpoint."
                )

                path_prev_checkpoint = find_most_recent_checkpoint(path_prev_checkpoint)

            else:
                print(
                    "Could not find previous checkpoint folder. Training from beginning."
                )

        path_input_folder = path_processed_data / "input"
        path_truth_folder = path_processed_data / "truth"
        path_checkpoint_folder = (
            root_dir / "models/interim/checkpoints" / model_start_time
        )
        Path(path_checkpoint_folder).mkdir(parents=True, exist_ok=True)

    # save src directory as a zip into the checkpoint folder
    shutil.make_archive(
        path_checkpoint_folder / f"src_files_{model_start_time}",
        "zip",
        proj_dir / "src",
    )
    shutil.copy(
        proj_dir / "bash_scripts/train_model_hpc.sh",
        path_checkpoint_folder / "train_model_hpc.sh",
    )

    return (
        root_dir,
        path_input_folder,
        path_truth_folder,
        path_checkpoint_folder,
        path_prev_checkpoint,
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
    root_dir, path_input_folder, path_truth_folder, path_checkpoint_folder, path_prev_checkpoint, model_start_time = set_directories()


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--path_data", dest="path_data", type=str, help="Path to processed data"
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
        "-p",
        "--proj_dir",
        dest="proj_dir",
        type=str,
        help="Location of project folder",
    )


    parser.add_argument(
        "--var_to_include",
        dest="var_to_include",
        type=int,
        default=1,
        help="Number of variables to be trained on. var_to_include=1 \
            is only the temperature data. \
            var_to_include=4 is the temperature, vx, vy, and vz.",
    )

    parser.add_argument(
        "--batch_size",
        dest="batch_size",
        type=int,
        default=1,
        help="Mini-batch size for each GPU",
    )

    parser.add_argument(
        "--cat_noise",
        action="store_true",
        help="Will concatenate noise if argument used (sets cat_noise=True).",
    )

    parser.add_argument(
        "--learning_rate",
        dest="learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate for optimizer",
    )

    parser.add_argument(
        "--critic_iterations",
        dest="critic_iterations",
        type=int,
        default=5,
        help="Number of critic iterations for every 1 generator iteration",
    )

    parser.add_argument(
        "--num_epochs",
        dest="num_epochs",
        type=int,
        default=500,
        help="Number of epochs",
    )

    parser.add_argument(
        "--lambda_gp",
        dest="lambda_gp",
        type=int,
        default=10,
        help="Lambda modifier for gradient penalty",
    )

    parser.add_argument(
        "--gen_pretrain_epochs",
        dest="gen_pretrain_epochs",
        type=int,
        default=5,
        help="Epochs to train generator alone at the beginning",
    )

    args = parser.parse_args()


    (
        root_dir,
        path_input_folder,
        path_truth_folder,
        path_checkpoint_folder,
        path_prev_checkpoint,
        model_start_time,
    ) = set_directories()

    main(args)