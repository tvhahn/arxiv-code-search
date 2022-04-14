import pandas as pd
import numpy as np
from pathlib import Path
import os
import json


def parse_json(arxiv_json_path):
    """Parse the kaggle json file and return as dataframe"""

    metadata  = []
    with open(arxiv_json_path, 'r') as f:
        for line in f: 
            metadata.append(json.loads(line))

    return pd.DataFrame(metadata)


def load_metadata_csv(metadata_file_path):
    """
    Load arxiv metadata csv file.
    Originally extracted from arxiv-metadata-oai-snapshot.json
    """

    dtypes_dict = {
        'id': str, 
        'submitter': str, 
        'authors': str, 
        'title': str, 
        'comments': str, 
        'journal-ref': str, 
        'doi': str,
        'report-no': str, 
        'categories': str, 
        'license': str, 
        'abstract': str, 
        'versions': str,
        'authors_parsed': str,
    }

    # if metadata_file_path ends in .gz, use gzip in pandas
    if metadata_file_path.suffix == '.gz':
        df = pd.read_csv(metadata_file_path, dtype=dtypes_dict, parse_dates=['update_date'], compression='gzip')
    else:
        df = pd.read_csv(metadata_file_path, dtype=dtypes_dict, parse_dates=['update_date'])

    # parse "versions" and "authors_parsed" columns with eval
    df["versions"] = df["versions"].apply(eval)
    df["authors_parsed"] = df["authors_parsed"].apply(eval)

    return df

def filter_by_category(df, regex_pattern_cat=None):
    """
    Filter out papers based on categories using regex

    e.g regex_pattern_cat = "cs|eess"
    """
    # regex pattern to match categories (is a string)
    if regex_pattern_cat is None:
        return df
    else:
        # filter out papers based on categories using regex
        return df[df["categories"].str.contains(regex_pattern_cat, regex=True)]

def filter_by_date(df, start_date=None, end_date=None):
    """
    Filter out papers based on start and end dates
    """
    # filter out papers based on start and end dates
    if start_date is None and end_date is None:
        return df
    elif start_date is None:
        return df[(df["update_date"] <= end_date)]
    elif end_date is None:
        return df[(df["update_date"] >= start_date)]
    else:
        return df[(df["update_date"] >= start_date) & (df["update_date"] <= end_date)]

def filter_by_license(df, license_filter_list=None):
    """
    Filter by license.
    """

    # dict to match license in df
    license_type_dict = {
        "ARXIV": "http://arxiv.org/licenses/nonexclusive-distrib/1.0/",
        "CC BY 4.0": "http://creativecommons.org/licenses/by/4.0/",
        "CC BY-NC-SA 4.0": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
        "CC BY 3.0": "https://creativecommons.org/licenses/by/3.0/",
        "CC BY-NC-ND 4.0": "https://creativecommons.org/licenses/by-nc-nd/4.0/",
        "CC0 1.0": "https://creativecommons.org/publicdomain/zero/1.0/",
        "CC BY-NC-SA 3.0": "https://creativecommons.org/licenses/by-nc-sa/3.0/",
        "CC BY-SA 4.0": "https://creativecommons.org/licenses/by-sa/4.0/",
        "CC": "https://creativecommons.org/publicdomain/",
    }

    if license_filter_list is None:
        return df
    else:

        # make all items in license_filter_list upper case
        license_filter_list = [x.upper() for x in license_filter_list]

        # match license in license_filter_list to the name of the license in the df
        license_filter_list = [license_type_dict[x] for x in license_filter_list]

        # filter by license
        return df[df["license"].isin(license_filter_list)]

def select_random_articles(df, index_file_dir, check_duplicates=True, save_csv=True, save_name=None, n_articles=10):
    """
    Select n random articles from df, and ensure
    that they are not duplicated in other 'index_of_labels' csvs.
    """

    # get a list of the csv file names
    files = os.listdir(index_file_dir)

    file_list = [
        Path(index_file_dir) / filename
        for filename in files
        if filename.endswith(".csv")
        ]

    no_exisiting_index_files = len(file_list)

    if check_duplicates and no_exisiting_index_files > 0:
        # load index files with pandas and append to index_data_list
        index_data_list = []
        for file in file_list:
            index_data_list.append(load_metadata_csv(file))

        # concatenate index_data_list into one dataframe
        df_used = pd.concat(index_data_list).reset_index(drop=True)

        # concatenate df and df_used
        df_unique = pd.concat([df, df_used], sort=False).drop_duplicates(["id"], keep=False)
    else:
        df_unique = df

    # select random articles
    # check for edge cases where n_articles > df_unique.shape[0]
    # or when df_unique.shape[0] is 0
    if df_unique.shape[0] == 0:
        print("No unique articles to select from.")
        return None
    elif n_articles > df_unique.shape[0]:
        pass
    else:
        df_unique = df_unique.sample(n_articles)

    # save df_unique to csv if save_csv is True
    if save_csv:
        if save_name is None:
            save_name = f"index_of_articles_for_lables_{no_exisiting_index_files+1}.csv"
        save_path = index_file_dir / save_name
        df_unique.to_csv(save_path, index=False)

    return df_unique