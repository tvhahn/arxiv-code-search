import logging
from pathlib import Path
import pandas as pd
from multiprocessing import Pool
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO
import os
import numpy as np
import argparse
import tqdm
import re
import unicodedata
from pdfminer import high_level


# from https://stackoverflow.com/a/26495057
def convert_pdf_to_txt(path):

    text = high_level.extract_text(path)
    text = text.replace("-\n", "")
    text = unicodedata.normalize("NFC", text)

    text_lower = text.lower()

 

    # get the .txt save name
    save_name = path.stem + ".txt"


    return {save_name: text}


def main(pdf_dir_path):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    # logger = logging.getLogger(__name__)
    # logger.info(
    #     "making the final data set with geo data, but nothing extra (e.g APGAR data)"
    # )

    # get a list of file names
    files = os.listdir(pdf_dir_path)

    file_list = [
        Path(pdf_dir_path) / filename
        for filename in files
        if filename.endswith(".pdf")
    ]

    # set up your pool
    with Pool(processes=args.n_cores) as pool:  # or whatever your hardware can support

        # have your pool map the file names to the converter
        # use tqdm for a progress bar - from https://stackoverflow.com/a/45276885
        txt_list = list(tqdm.tqdm(pool.imap(convert_pdf_to_txt, file_list), total=len(file_list)))

        # txt_list = pool.map(convert_pdf_to_txt, file_list)

        txt_dict = {}
        for txt in txt_list:
            txt_dict.update(txt)

        return txt_dict


if __name__ == "__main__":
    # log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    # logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(description="Create .txt files from .pdf files")

    parser.add_argument(
        "--n_cores",
        type=int,
        default=6,
        help="Number of cores to use for multiprocessing",
    )

    # argument for txt_dir_path
    parser.add_argument(
        "--raw_data_dir",
        type=str,
        help="Path to the raw data directory (as a str).",
    )


    args = parser.parse_args()

    if args.raw_data_dir:
        raw_data_dir = Path(args.raw_data_dir)
    else:
        raw_data_dir = project_dir / "data/raw"

    ######
    # journal articles
    ######
    pdf_dir_path = raw_data_dir / "pdfs"
    txt_dir_path = raw_data_dir / "txts"

    # make the txt directory if it doesn't exist
    txt_dir_path.mkdir(parents=True, exist_ok=True)

    txt_dict = main(pdf_dir_path)
    
    for save_name in txt_dict:
        with open(txt_dir_path / save_name, "w") as f:
            f.write(txt_dict[save_name])
