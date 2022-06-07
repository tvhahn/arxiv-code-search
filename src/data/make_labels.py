"""
Combine multiple csv or ods files, that contain labels, into one. Save into the data/processed/labels/labels_complete directory.
"""

from pathlib import Path
import pandas as pd
from multiprocessing import Pool
import os
import logging
import argparse


def read_csv(filename):
    "converts a filename to a pandas dataframe"
    return pd.read_csv(filename)


def read_ods(filename):
    "converts a filename to a pandas dataframe"

    df = pd.read_excel(
        filename,
        parse_dates=["update_date"],
        engine="odf",
        )
    return df[["id", "pattern", "token_count", "update_date", "label", "para"]]


def set_directories(args):

    if args.path_data_dir:
        path_data_dir = Path(args.path_data_dir)
    else:
        path_data_dir = Path().cwd() / "data"


    if args.path_label_dir:
        path_label_dir = Path(args.path_label_dir)
    else:
        path_label_dir = path_data_dir / "interim"


    path_save_dir = path_data_dir / "processed" / "labels" / "labels_complete"

    Path(path_save_dir).mkdir(parents=True, exist_ok=True)

    return path_data_dir, path_label_dir, path_save_dir


def main(args):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    path_data_dir, path_label_dir, path_save_dir = set_directories(args)

    # get a list of file names
    if args.file_type == "ods":
        files = os.listdir(path_label_dir)
        file_list = [
            Path(path_label_dir) / filename
            for filename in files
            if filename.endswith(".ods")
        ]

        reader_func = read_ods

    elif args.file_type == "csv":
        files = os.listdir(path_label_dir)
        file_list = [
            Path(path_label_dir) / filename
            for filename in files
            if filename.endswith(".csv")
        ]

        reader_func = read_csv
    else:
        raise ValueError("file_type must be either ods or csv")


    # set up your pool
    with Pool(
        processes=args.n_cores
    ) as pool:  # or whatever your hardware can support

        # have your pool map the file names to dataframes
        df_list = pool.map(reader_func, file_list)

        # reduce the list of dataframes to a single dataframe
        combined_df = pd.concat(df_list, ignore_index=True)

        return combined_df


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.info("combine multiple label csv or ods files into one")

    parser = argparse.ArgumentParser(description="Create final labels.")

    parser.add_argument(
        "--path_data_dir",
        type=str,
        help="Path to data folder that contains raw/interim/processed data folders",
    )

    parser.add_argument(
        "--path_label_dir",
        type=str,
        help="Path to the folder that contains all the individual label files (csv or ods)",
    )


    parser.add_argument(
        "--n_cores",
        type=int,
        default=2,
        help="Number of cores to use for multiprocessing",
    )

    parser.add_argument(
        "--file_type",
        type=str,
        default="ods",
        help="Combine either the csv or ods files",
    )

    parser.add_argument(
        "--save_name",
        type=str,
        default="labels.csv",
        help="Name of the final compiled label csv file",
    )

    args = parser.parse_args()


    df = main(args)
    df = df.dropna(subset=['label']).astype({'label': int})
    print("Final df shape:", df.shape)

    path_data_dir, path_label_dir, path_save_dir = set_directories(args)
    df.to_csv(
        path_save_dir / args.save_name,
        index=False,
    )
