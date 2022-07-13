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


def get_txt_file_list(txt_root_dir):
    if args.index_file_no:
        # get a list of file names
        txt_dir = txt_root_dir / str(args.index_file_no)

        files = os.listdir(txt_dir)

        file_list = [
            Path(txt_dir) / filename for filename in files if filename.endswith(".txt")
        ]

        file_dict = {args.index_file_no: file_list}
    else:
        # find all the sub folders in the txt_root_dir
        txt_dir_list = [
            txt_root_dir / dir_name for dir_name in os.listdir(txt_root_dir)
        ]

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


def create_chunks_of_text(text, init_token_len, max_token_len):
    token_count = len(nltk.word_tokenize(text))

    if init_token_len is None:
        init_token_len = int(max_token_len * 0.85)

    n_splits = int(np.ceil(token_count / init_token_len))

    sent = np.array(nltk.sent_tokenize(text))

    # create an array of sentence token lengths
    sent_token_lengths = np.array([len(nltk.word_tokenize(s)) for s in sent])
    sent_split_lengths = np.array_split(sent_token_lengths, n_splits)
    sent_split_indices = np.array_split(np.arange(0, len(sent_token_lengths)), n_splits)

    split_dict = {}
    for i in range(n_splits):
        chunk_lengths = int(np.sum(sent_split_lengths[i]))
        chunk_indices = list(sent_split_indices[i])

        if i == 0:
            forwards = list(
                zip(np.hstack(np.array(sent_split_lengths[i+1:], dtype=object)), 
                    np.hstack(np.array(sent_split_indices[i+1:], dtype=object)))
                    )
            # f_l: forward lenghts, f_i: forward indices
            for f_l, f_i in forwards:
                if chunk_lengths + f_l <= max_token_len:
                    chunk_lengths += f_l
                    chunk_indices = chunk_indices + [f_i]
                else:
                    break
            
            split_text = " ".join([s for s in sent[chunk_indices]])
            split_dict[i] = [split_text, chunk_lengths]

        elif i > 0 and i <= n_splits - 2:
            
            forwards = list(
                zip(np.hstack(np.array(sent_split_lengths[i+1:], dtype=object)), 
                    np.hstack(np.array(sent_split_indices[i+1:], dtype=object))))

            backwards = list(
                zip(np.hstack(np.array(sent_split_lengths[:i], dtype=object))[::-1], 
                    np.hstack(np.array(sent_split_indices[:i], dtype=object))[::-1]))

            for k in range(max_token_len):
                if k % 2 == 0:
                    f_l = forwards[0][0]
                    f_i = forwards[0][1]
                    if chunk_lengths + f_l <= max_token_len:
                        chunk_lengths += f_l
                        chunk_indices = chunk_indices + [f_i]
                        forwards.pop(0)
                    else:
                        break
                else:
                    b_l = backwards[0][0]
                    b_i = backwards[0][1]
                    if chunk_lengths + b_l <= max_token_len:
                        chunk_lengths += b_l
                        chunk_indices = [b_i] + chunk_indices
                        backwards.pop(0)
                    else:
                        break

            split_text = " ".join([s for s in sent[chunk_indices]])
            split_dict[i] = [split_text, chunk_lengths]
            
        else:

            backwards = list(
                zip(np.hstack(np.array(sent_split_lengths[:i], dtype=object))[::-1], 
                    np.hstack(np.array(sent_split_indices[:i], dtype=object))[::-1]))

            for b_l, b_i in backwards:
                if chunk_lengths + b_l <= max_token_len:
                    chunk_lengths += b_l
                    chunk_indices = [b_i] + chunk_indices
                else:
                    break

            split_text = " ".join([s for s in sent[chunk_indices]])
            split_dict[i] = [split_text, chunk_lengths]

    return split_dict


def extract_matches_as_paragraphs(match_indices, text, id, init_token_len=None, max_token_len=400, save_str_width=4000):
    para_list = []
    token_count_list = []

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
        token_count = len(nltk.word_tokenize(matched_para))

        if token_count > max_token_len:
            try:
                split_dict = create_chunks_of_text(matched_para, init_token_len, max_token_len)
                for k, v in split_dict.items():
                    para_list.append(v[0])
                    token_count_list.append(v[1])
            except Exception as e:
                print(f"{id}: Error in creating chunks of text:", e)

        else:
            para_list.append(matched_para)
            token_count_list.append(token_count)

    return para_list, token_count_list


def merge_existing_new_labels(df_existing, df_new):
    df_existing = df_existing.merge(
        df_new[["para", "id", "pattern", "token_count"]], on=["para", "id", "token_count"], how="outer"
    )

    # if pattern_y is NaN, then copy pattern_x to pattern_y
    df_existing["pattern"] = df_existing["pattern_y"].fillna(df_existing["pattern_x"])

    # drop columns that are not needed, pattern_x and pattern_y
    df_existing = df_existing.drop(["pattern_x", "pattern_y"], axis=1)
    return df_existing[["id", "pattern", "token_count", "update_date", "label", "para"]]


def main(file_list, index_no):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("search the txt files for the keywords and save results in a csv")

    # get a list of file names
    pattern_dict = {
        "open-source": re.compile(r"\b(open-source|open source|open-sourced|open sourced)\b"),
        "open-source data": re.compile(
            r"\b(open-source|open source|open)(?:\W+\w+){0,9}?\W+(data|dataset|data set)\b"
        ),
        "data open-source ": re.compile(
            r"\b(data|dataset|data set)(?:\W+\w+){0,9}?\W+(open-source|open source)\b"
        ),
        "reproducibility": re.compile(r"\b(reproducibility|reproducible|reproducible data|reproduce)\b"),
        # "open-source data": re.compile(r'\b(?:open-source|open source\W+(?:\w+\W+){1,6}?data|dataset|data set|(data|dataset|data set)\W+(?:\w+\W+){1,6}?(open-source|open source))\b'),
        "open-source code": re.compile(
            r"\b(open-source|open source|open-sourced|open sourced)(?:\W+\w+){0,9}?\W+(code)\b"
        ),
        "provide implementation": re.compile(
            r"\b(provide|provided)(?:\W+\w+){0,9}?\W+(implementation|implementations)\b"
        ),
        "implementation available": re.compile(r"\b(implementation)(?:\W+\w+){0,9}?\W+(available)\b"),
        "code open-source": re.compile(
            r"\b(code)(?:\W+\w+){0,9}?\W+(open-source|open source)\b"
        ),
        "source code": re.compile(
            r"\b(source code|source-code|source codes|source-codes)\b"
        ),
        "github": re.compile(r"\b(github|gitlab)\b"),
        "retrieve": re.compile(r"(retreived|retrieved)"),
        "data repos": re.compile(r"(mendeley|phm data challenge|nasa ames)"),
        "repository": re.compile(r"\b(repository|repositories)\b"),
        "data repository": re.compile(
            r"\b(data|dataset|data set)(?:\W+\w+){0,9}?\W+(repository|repo|catalogue)\b"
        ),
        "repo": re.compile(r"\b(repo)\b"),
        "benchmark": re.compile(r"\b(benchmark)\b"),
        "used dataset": re.compile(
            r"\b(used|use)(?:\W+\w+){0,5}?\W+(dataset|data set)\b"
        ),
        "case study data": re.compile(
            r"\b(case study|benchmark)(?:\W+\w+){0,9}?\W+(data|data set|dataset)\b"
        ),
        "public": re.compile(r"\b(public)\b"),
        "public instance": re.compile(r"\b(public instance)\b"),
        "dataset": re.compile(r"\b(dataset|data set|datasets|data-sets)\b"),
        "corpus": re.compile(r"\b(corpus)\b"),
        "download": re.compile(r"\b(download|download)\b"),
        "data": re.compile(r"\b(data)\b"),
        "database": re.compile(r"\b(database)\b"),
        "baseline": re.compile(r"\b(baseline|baselines)\b"),
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
            r"\b(data|dataset|data set|freely)(?:\W+\w+){0,9}?\W+(available|access|found)\b"
        ),  # https://www.regular-expressions.info/near.html
        "code available": re.compile(
            r"\b(code)(?:\W+\w+){0,9}?\W+(available|access|download|package)\b"
        ),
        "data https": re.compile(
            r"\b(data|dataset|data set)(?:\W+\w+){0,9}?\W+(https|http|online)"
        ),
        "dataset provided": re.compile(
            r"\b(data|dataset|data set)(?:\W+\w+){0,9}?\W+(provide|provided|supplied|public)\b"
        ),
        "publicly available": re.compile(
            r"\b(publicly|public|available)(?:\W+\w+){0,9}?\W+(publicly|available|access|download|package|accessible|dataset|data set|data)\b"
        ),
    }

    pattern_names = list(pattern_dict.keys())

    df_list = []

    save_str_width = 4000

    # load the txt file as a string
    for txt_path in file_list:
        id = txt_path.stem
        id_list = []
        para_list = []
        pattern_name_list = []
        token_count_list = []
        
        with open(txt_path, "r") as f:
            txt = f.read()
            txt_lower = txt.lower()

        for pattern in pattern_names:
            match_index = [
                match.start() for match in re.finditer(pattern_dict[pattern], txt_lower)
            ]

            match_count = len(match_index)

            if match_count == 0:
                continue
            else:

                temp_para_list, temp_token_count_list = extract_matches_as_paragraphs(
                    match_index, txt, id, save_str_width=save_str_width, max_token_len=args.max_token_len
                )

                para_list.extend(temp_para_list)
                token_count_list.extend(temp_token_count_list)
                pattern_name_list.extend([pattern] * len(temp_para_list))
                id_list.extend([id] * len(temp_para_list))

        df = pd.DataFrame(
            [id_list, pattern_name_list, token_count_list, para_list],
        ).T
        df.columns = ["id", "pattern_name", "token_count", "para"]
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
        df.groupby(["id", "token_count", "para"])["pattern_name"]
        .apply(list)
        .reset_index(name="pattern")
    )

    df["pattern"] = df[["pattern"]].apply(unique_vals, axis=1)

    # add an empty 'label' column to the df
    df["label"] = ""

    # add an empty 'update_date' column to the df
    df["update_date"] = ""

    df = df[["id", "pattern", "token_count", "update_date", "label", "para"]].sort_values(by=["id", "pattern", "token_count"])

    df = df.astype(
        {"id": str, "pattern": str, "token_count": str, "update_date": str, "label": str, "para": str}
    )

    save_dir = project_dir / "data/interim"

    # make the save_dir directory if it doesn't exist
    save_dir.mkdir(parents=True, exist_ok=True)
    save_name = f"labels_{str(index_no)}.csv"

    if args.keep_old_files:
        # get current date and time and store as nice format
        now = datetime.now()
        now_str = now.strftime("%Y-%m-%d_%H.%M")

        # save the old version of the file
        old_save_name = f"{now_str}_labels_{str(index_no)}.csv"
        old_save_path = save_dir / old_save_name

        # check if old_save_path exists
        if (save_dir / save_name).exists():
            # use shutil to copy the file and rename it to old_save_name
            shutil.copy(save_dir / save_name, old_save_path)
        

    if args.overwrite:
        pass
    else:
        existing_labels_path = save_dir / save_name
        if existing_labels_path.exists():
            df_existing = pd.read_csv(existing_labels_path, dtype=str)
            df = merge_existing_new_labels(df_existing, df)
        else:
            pass

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

    parser.add_argument(
        "--max_token_len",
        type=int,
        default=400,
        help="Maximum token length (approximate).",
    )


    parser.add_argument(
        "--overwrite",
        default=False,
        action="store_true",
        help="Overwrite existing files, otherwise will be merged into existing file.",
    )

    parser.add_argument(
        "--keep_old_files",
        default=False,
        action="store_true",
        help="Keep a copy of the old label file.",
    )

    args = parser.parse_args()

    if args.txt_root_dir:
        txt_root_dir = Path(args.txt_root_dir)
    else:
        txt_root_dir = project_dir / "data/raw/txts"

    file_dict = get_txt_file_list(txt_root_dir)

    for index_no, file_list in file_dict.items():
        main(file_list, index_no)
