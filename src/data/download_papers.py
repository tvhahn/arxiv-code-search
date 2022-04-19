# -*- coding: utf-8 -*-
import logging
from pathlib import Path
import json
import pandas as pd
from src.data.utils import load_metadata_csv
import argparse
import ast
from time import sleep

from urllib.request import urlretrieve


def download_papers_from_index(pdf_root_dir, index_file_path=None, index_file_no=None, index_file_dir=None, n_papers=None):
    """
    Download papers from an index file.

    Index files will have format like: "index_of_papers_for_labels_{index_file_no}.csv"
    """

    if index_file_no is not None and index_file_dir is not None:
        index_file_path = index_file_dir / f"index_of_papers_for_labels_{index_file_no}.csv"
    if index_file_no is not None and index_file_dir is None:
        index_file_dir = project_dir / "data/processed/labels/index_files"
        index_file_path = index_file_dir / f"index_of_papers_for_labels_{index_file_no}.csv"
    if index_file_path is not None and index_file_dir is None:
        # get index_file_no from index_file_path
        index_file_no = index_file_path.stem.split("_")[-1]
    
    # assert that index_file_path exists
    assert index_file_path.exists(), f"{index_file_path} does not exist or not specified."



    pdf_root_dir.mkdir(parents=True, exist_ok=True)

    pdf_save_dir = pdf_root_dir / str(index_file_no)

    pdf_save_dir.mkdir(parents=True, exist_ok=True)

    df = load_metadata_csv(index_file_path)

    for i, row in df.iterrows():
        id = row['id']
        title = row['title']
        print(f"{i}: {id} - {title}")
        url = f"https://arxiv.org/pdf/{id}.pdf"
        urlretrieve(url, pdf_save_dir / f"{id}.pdf")
        sleep_time = 3
        # print('sleep time: ', sleep_time)
        sleep(sleep_time)
        if n_papers is not None and i >= n_papers-1:
            break


def main():
    """
    Download papers from an index file.
    """
    logger = logging.getLogger(__name__)
    logger.info("Download arxiv papers")

    pdf_root_dir = project_dir / "data/raw/pdfs"
    index_file_dir = project_dir / "data/processed/labels/index_files"

    if args.n_papers:
        n_papers = args.n_papers
    else:
        n_papers = None

    download_papers_from_index(pdf_root_dir, index_file_no=args.index_file_no, index_file_dir=index_file_dir, n_papers=n_papers)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    parser = argparse.ArgumentParser(description="Build data sets for analysis")

    parser.add_argument(
        "--index_file_no",
        type=int,
        help="Index number of the index file to use.",
    )

    parser.add_argument(
        "--n_papers",
        type=int,
        help="Number of papers to download. If not specified, download all papers.",
    )

    args = parser.parse_args()

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()
