import pandas as pd
from pathlib import Path
import numpy as np
import re
import os
import argparse
import logging


def get_txt_file_list(txt_root_dir):
    if args.index_file_no:
        # get a list of file names
        txt_dir = txt_root_dir / str(args.index_file_no)

        files = os.listdir(txt_dir)

        file_list = [
            Path(txt_dir) / filename
            for filename in files
            if filename.endswith(".txt")
        ]

        file_dict = {args.index_file_no: file_list}
    else:
        # find all the sub folders in the txt_root_dir
        txt_dir_list = [txt_root_dir / dir_name for dir_name in os.listdir(txt_root_dir)]

        file_dict = {}
        for txt_dir in txt_dir_list:
            # get index_no from txt_dir
            index_no = int(txt_dir.stem)
            files = os.listdir(txt_dir)

            file_list = [
                Path(txt_dir) / filename
                for filename in files
                if filename.endswith(".txt")
            ]

            file_dict[index_no] = file_list
            
    return file_dict


def get_paragraph(sub_str, rel_ind):
    counter = 0

    for i, s in enumerate(sub_str.split("\n\n")):
        counter += len(s) + 2
        if counter > rel_ind:
            s = s.replace("\n", " ")
            s = s.replace("\t", " ")
            # s = s.replace("\r", " ")
            # s = s.replace('"', " ")
            s = s.replace("  ", " ")
            return s.strip()


def extract_matches_as_paragraphs(match_indices, text, save_str_width=4000):
    str_list = []

    for i in match_indices:

        # in case the match is too close to the beginning or
        # end of the text
        if i - save_str_width <= 0:
            start = 0
            rel_ind = i
        else:
            start = i - save_str_width
            rel_ind = save_str_width
        if i + save_str_width > len(text):
            end = len(text)
        else:
            end = i + save_str_width

        matched_para = get_paragraph(text[start:end], rel_ind)

        str_list.append(matched_para)
    return str_list


def main(file_list, index_no):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("search the txt files for the keywords and save results in a csv")

    # get a list of file names
    pattern_dict = {
        "open-source": re.compile(r"\b(open-source|open source)\b"),
        "open-source data": re.compile(
            r"\b(open-source|open source)(?:\W+\w+){0,9}?\W+(data|dataset|data set)\b"
        ),
        "data open-source ": re.compile(
            r"\b(data|dataset|data set)(?:\W+\w+){0,9}?\W+(open-source|open source)\b"
        ),
        # "open-source data": re.compile(r'\b(?:open-source|open source\W+(?:\w+\W+){1,6}?data|dataset|data set|(data|dataset|data set)\W+(?:\w+\W+){1,6}?(open-source|open source))\b'),
        "open-source code": re.compile(
            r"\b(open-source|open source)(?:\W+\w+){0,9}?\W+(code)\b"
        ),
        "provide implementation": re.compile(
            r"\b(provide|provided)(?:\W+\w+){0,9}?\W+(implementation|implementations)\b"
        ),
        "code open-source": re.compile(
            r"\b(code)(?:\W+\w+){0,9}?\W+(open-source|open source)\b"
        ),
        "github": re.compile(r"(github|gitlab)"),
        "data repos": re.compile(r"(mendeley|phm data challenge|nasa ames)"),
        "data repository": re.compile(
            r"\b(data|dataset|data set)(?:\W+\w+){0,9}?\W+(repository|repo)\b"
        ),
        "used dataset": re.compile(
            r"\b(used|use)(?:\W+\w+){0,5}?\W+(dataset|data set)\b"
        ),
        "dataset": re.compile(r"\b(dataset|data set|datasets|data-sets)\b"),
        "download": re.compile(r"\b(download|download)\b"),
        "data": re.compile(r"\b(data)\b"),
        "database": re.compile(r"\b(database)\b"),
        "python": "python",
        "package": re.compile(r"\b(package)\b"),
        "code": re.compile(r"\b(code)\b"),
        "code package": re.compile(
            r"\b(code|created|create|software)(?:\W+\w+){0,9}?\W+(package)\b"
        ),
        "supplementary data": re.compile(
            r"\b(supplementary|supplement)(?:\W+\w+){0,9}?\W+(data|dataset|data set)\b"
        ),
        "supplementary code": re.compile(
            r"\b(supplementary|supplement)(?:\W+\w+){0,9}?\W+(code)\b"
        ),
        "data available": re.compile(
            r"\b(data|dataset|data set)(?:\W+\w+){0,9}?\W+(available|access|found)\b"
        ),  # https://www.regular-expressions.info/near.html
        "code available": re.compile(
            r"\b(code)(?:\W+\w+){0,9}?\W+(available|access|download|package)\b"
        ),
        "data https": re.compile(
            r"\b(data|dataset|data set)(?:\W+\w+){0,9}?\W+(https|http|online)"
        ),
        "dataset provided": re.compile(
            r"\b(data|dataset|data set)(?:\W+\w+){0,9}?\W+(provide|provided|supplied)\b"
        ),
        "publicly available": re.compile(
            r"\b(publicly|public)(?:\W+\w+){0,9}?\W+(available|access|download|package|accessible|dataset|data set|data)\b"
        ),
    }

    pattern_names = list(pattern_dict.keys())


    df_list = []

    save_str_width = 4000

    # load the txt file as a string

    for txt_path in file_list:
        doi = txt_path.stem
        doi_list = []
        para_list = []
        pattern_name_list = []

        with open(txt_path, "r") as f:
            txt = f.read()
            txt_lower = txt.lower()

        for pattern in pattern_names:
            match_index = [
                match.start() for match in re.finditer(pattern_dict[pattern], txt_lower)
            ]

            match_count = len(match_index)
            # print(match_count)

            if match_count == 0:
                continue
            else:

                temp_para_list = extract_matches_as_paragraphs(
                    match_index, txt, save_str_width=save_str_width
                )

                para_list.extend(temp_para_list)
                pattern_name_list.extend([pattern] * match_count)
                doi_list.extend([doi] * match_count)

        df = pd.DataFrame(
            [doi_list, pattern_name_list, para_list],
        ).T
        df.columns = ["doi", "pattern_name", "para"]
        df_list.append(df)

    # concatenate the dataframes
    df = pd.concat(df_list)

    def unique_vals(cols):
        l = cols[0]
        # return a string of l, with each element separated by a comma
        return ", ".join(list(set(l)))

    # groupby and create list: https://stackoverflow.com/a/22221675
    # to load df with a list in it: https://stackoverflow.com/a/57373513 - use pd.eval
    df = (
        df.groupby(["doi", "para"])["pattern_name"]
        .apply(list)
        .reset_index(name="pattern")
    )
    df["pattern"] = df[["pattern"]].apply(unique_vals, axis=1)

    # add an empty 'label' column to the df
    df["label"] = ""

    # replace the '/' with '_' in the 'doi' column
    df["save_name"] = df["doi"].str.replace("/", "_")

    df = df[["save_name", "doi", "pattern", "label", "para"]]

    save_dir = project_dir / "data/interim"

    # make the save_dir directory if it doesn't exist
    save_dir.mkdir(parents=True, exist_ok=True)

    # save df_all to a csv
    df.to_csv(save_dir / f"labels_{str(index_no)}.csv", index=False)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(description="Create .txt files from .pdf files")


    # argument for txt_dir_path
    parser.add_argument(
        "--txt_root_dir",
        type=str,
        help="Path to the folder that contains all the txt files.",
    )

    parser.add_argument(
        "--index_file_no",
        type=int,
        help="Index number of the index file to use. Will only search in this folder for txts.",
    )


    args = parser.parse_args()

    if args.txt_root_dir:
        txt_root_dir = Path(args.txt_root_dir)
    else:
        txt_root_dir = project_dir / "data/raw/txts"


    file_dict = get_txt_file_list(txt_root_dir)

    for index_no, file_list in file_dict.items():
        main(file_list, index_no)


    
