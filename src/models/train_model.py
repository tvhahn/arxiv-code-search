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
from torch.utils.tensorboard import SummaryWriter
import argparse
import shutil
import logging
import re
import argparse
import datetime
from src.models.utils import (create_data_loader, EarlyStopping, under_over_sampler)
from src.models.model import ArxivClassifier


def find_most_recent_checkpoint(path_prev_checkpoint_dir):
    """Finds the most recent checkpoint in a checkpoint folder
    and returns the path to that .pt file.
    """

    ckpt_list = list(path_prev_checkpoint_dir.rglob("*.pt"))
    if len(ckpt_list) == 1:
        print("Previous checkpoints exist. Training from most recent checkpoint.")
        return Path(path_prev_checkpoint_dir / ckpt_list[0])
    elif len(ckpt_list) > 1:
        print("Previous checkpoints exist. Training from most recent checkpoint.")
        try:
            max_epoch = sorted(list(int(re.findall("[0-9]+", str(i))[-1]) for i in ckpt_list))[
                -1
            ]
            return Path(path_prev_checkpoint_dir / f"train_{max_epoch}.pt")

        except:
            pass
    else:
        print("No checkpoints found in the specified checkpoint directory.")
        return Path("no_prev_checkpoint_found")


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
    elif scratch_path.exists():
        path_label_dir = scratch_path / "arxiv-code-search" / "data" / "processed" / "labels" / "labels_complete"
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
        
            path_prev_checkpoint = find_most_recent_checkpoint(
                path_prev_checkpoint_dir
            )

        else:
            print("Could not find previous checkpoint folder. Training from beginning.")
            path_prev_checkpoint = Path("prev_checkpoint_not_found")
    else:
        # set dummy name for path_prev_checkpoint
        path_prev_checkpoint = Path("no_prev_checkpoint_needed")

    path_checkpoint_dir = path_model_dir / "interim/checkpoints" / model_start_time

    Path(path_checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # save src directory as a zip into the checkpoint folder
    shutil.make_archive(
        path_checkpoint_dir / f"src_files_{model_start_time}",
        "zip",
        proj_dir / "src",
    )

    return (
        proj_dir,
        path_data_dir,
        path_model_dir,
        path_label_dir,
        path_checkpoint_dir,
        path_prev_checkpoint,
        model_start_time,
    )


def save_checkpoint(model, path_checkpoint_dir, model_name="model_checkpoint.pt"):
    torch.save(
        {
            "model": model.state_dict(),
        },
        path_checkpoint_dir / model_name,
    )


# Function to calculate the accuracy of our predictions vs labels
# from https://mccormickml.com/2019/07/22/BERT-fine-tuning/#41-bertforsequenceclassification
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def train_epoch(
  model, 
  data_loader, 
  loss_fn, 
  optimizer, 
  device, 
  scheduler, 
  n_examples,
):
    model = model.train()

    # losses = []
    # correct_predictions = 0
    total_train_loss = 0
    total_train_accuracy = 0
  
    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_masks = d["attention_masks"].to(device)
        labels = d["labels"].to(device)

        model.zero_grad()    

        loss, logits = model(
            input_ids=input_ids,
            attention_mask=attention_masks,
            labels=labels
        )

        total_train_loss += loss.item()

        # _, preds = torch.max(outputs, dim=1)
        # loss = loss_fn(outputs, labels)

        # correct_predictions += torch.sum(preds == labels)
        # losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        logits = logits.detach().cpu().numpy()
        labels = labels.to('cpu').numpy()

        total_train_accuracy += flat_accuracy(logits, labels)
        
    return np.mean(total_train_accuracy), np.mean(total_train_loss)


def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()

    total_val_loss = 0
    total_val_accuracy = 0

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_masks = d["attention_masks"].to(device)
            labels = d["labels"].to(device)

            loss, logits = model(
                input_ids=input_ids,
                attention_mask=attention_masks,
                labels=labels
            )

            total_val_loss += loss.item()

            logits = logits.detach().cpu().numpy()
            labels = labels.to('cpu').numpy()

            total_val_accuracy += flat_accuracy(logits, labels)

    return np.mean(total_val_accuracy), np.mean(total_val_loss)


def main(args):

    # modifiable parameters not defined with argparse
    PRE_TRAINED_MODEL_NAME = 'allenai/scibert_scivocab_uncased'
    MAX_LEN = 512 # maximum number of tokens
    N_LABELS = 4 # number of labels
    EARLY_STOP_DELAY = 5 # number of epochs to wait before monitoring for early stopping
    PATIENCE = 100 # number of epochs to wait before early stopping to see if the model is improving

    # other args
    batch_size = args.batch_size
    n_epochs = args.n_epochs
    n_classes = args.n_classes

    # set directories
    (
        proj_dir,
        path_data_dir,
        path_model_dir,
        path_label_dir,
        path_checkpoint_dir,
        path_prev_checkpoint,
        model_start_time,
    ) = set_directories(args)

    print("path_label_dir:", path_label_dir)

    # setup TensorBoard
    # writer_train = SummaryWriter(path_model_dir / "logs" / model_start_time)
    writer = SummaryWriter(path_model_dir / "logs" / model_start_time)
    writer_acc = SummaryWriter(path_model_dir / "logs" / model_start_time) 

    # prepare data
    df = pd.read_csv(path_label_dir / "labels.csv", dtype={"id": str})
    df["y"] = df["label"].apply(lambda x: 1 if x > 0 else 0) # binary labels
    ids = df["id"].unique() # get all the arxiv paper ids
    
    train_ids, val_ids = train_test_split(ids, test_size=0.4, random_state=13,)

    df_train = df[df['id'].isin(train_ids)]
    df_val = df[df['id'].isin(val_ids)]
    
    # build model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device name:", device)

    tokenizer = BertTokenizer.from_pretrained(proj_dir / "bert_cache_dir")

    if n_classes > 2:
        label_column = "label"
    else:
        label_column = "y" # binary label

    # over/under sampling (not performed on validation set)
    df_train, _ = under_over_sampler(df_train, df_train['y'], method="random_over", ratio=0.3)
    counts = df_train["y"].value_counts()
    print(f"df_train counts after over sampling: {counts[0]}, {counts[1]}")
    df_train, _ = under_over_sampler(df_train, df_train['y'], method="random_under", ratio=0.6)
    counts = df_train["y"].value_counts()
    print(f"df_train counts afer under sampling: {counts[0]}, {counts[1]}")

    # get percentages of each class in each split
    print(df_train["y"].value_counts() / len(df_train) * 100)
    print(df_val["y"].value_counts() / len(df_val) * 100)
    
    train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, batch_size, label_column)
    val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, batch_size, label_column)

    # model and model parameters
    # model = ArxivClassifier(n_classes, proj_dir / "bert_cache_dir")

    
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
        num_labels = 2, # The number of output labels--2 for binary classification.
                        # You can increase this for multi-class tasks.   
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = False, # Whether the model returns all hidden-states.
    )

    model = model.to(device)


    optimizer = AdamW(model.parameters(), lr=2e-5, eps = 1e-8)
    total_steps = len(train_data_loader) * n_epochs

    scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=0,
                num_training_steps=total_steps
            )

    loss_fn = nn.CrossEntropyLoss().to(device)

    # load from checkpoint if wanted
    if path_prev_checkpoint.exists():
        print("Loading from previous checkpoint")
        checkpoint = torch.load(path_prev_checkpoint)
        model.load_state_dict(checkpoint["model"])
        epoch_start = checkpoint["epoch"] + 1
    else:
        epoch_start = 0

    #################
    # TRAINING LOOP #
    #################
    history = defaultdict(list)
    best_accuracy = 0

    # initialize the early_stopping object
    early_stopping = EarlyStopping(
        patience=PATIENCE,
        verbose=False,
        early_stop_delay=EARLY_STOP_DELAY,
        path=path_checkpoint_dir / "checkpoint.pt",
        delta=0.0001,
        )

    for epoch in range(epoch_start, epoch_start + n_epochs):
        print(f'Epoch {epoch + 1}/{n_epochs}')
        print('-' * 10)
        train_acc, train_loss = train_epoch(
                                    model,
                                    train_data_loader,    
                                    loss_fn, 
                                    optimizer, 
                                    device, 
                                    scheduler, 
                                    len(df_train)
                                )

        print(f'Train loss {train_loss} accuracy {train_acc}')
        # writer_train.add_scalar("Loss/train", train_loss, epoch)

        val_acc, val_loss = eval_model(
                                model,
                                val_data_loader,
                                loss_fn, 
                                device, 
                                len(df_val)
                            )



        print(f'Val   loss {val_loss} accuracy {val_acc}')
        print()
        writer.add_scalars(model_start_time, {"train_loss": train_loss, "val_loss": val_loss}, epoch)
        writer_acc.add_scalars(model_start_time, {"train_acc": train_acc, "val_acc": val_acc}, epoch)

        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break


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
        default=4,
        help="Mini-batch size to send to GPU",
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
        default=10,
        help="Number of epochs",
    )

    parser.add_argument(
        "--n_classes",
        dest="n_classes",
        type=int,
        default=2,
        help="Number of classes",
    )


    args = parser.parse_args()

    main(args)
