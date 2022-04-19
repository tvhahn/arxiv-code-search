# -*- coding: utf-8 -*-
import logging
from pathlib import Path
import json
import pandas as pd
from src.data.utils import (
    parse_json,
    load_metadata_csv,
    filter_by_category,
    filter_by_date,
    filter_by_license,
    select_random_papers,
)
import argparse
import ast


def save_index_parameters(save_name, args, index_file_dir):
    index_parameters_file_path = index_file_dir.parent / "index_selection_params.csv"
    if index_parameters_file_path.exists():
        # load index_parameters_file_path and have all columns as strings

        df_index_parameters = pd.read_csv(
            index_parameters_file_path,
            dtype=str,
        )

        df_new = pd.DataFrame.from_dict(
            {
                "save_name": save_name,
                "regex_pattern_cat": str(args.regex_pattern_cat),
                "start_date": args.start_date,
                "end_date": args.end_date,
                "license_filter_list": args.license_filter_list,
                "n_papers": args.n_papers,
            },
            orient="index",
        ).T

        df_index_parameters = pd.concat([df_index_parameters, df_new], sort=False)
        df_index_parameters.to_csv(index_parameters_file_path, index=False)
    else:
        df_index_parameters = pd.DataFrame.from_dict(
            {
                "save_name": save_name,
                "regex_pattern_cat": str(args.regex_pattern_cat),
                "start_date": args.start_date,
                "end_date": args.end_date,
                "license_filter_list": args.license_filter_list,
                "n_papers": args.n_papers,
            },
            orient="index",
        ).T
        df_index_parameters.to_csv(index_parameters_file_path, index=False)


def main():
    """
    Make an index of random papers in the arxiv dataset. This select a subset of the
    arxiv dataset (based on some filtering criteria) and saves it as a csv file.
    Saved csv files have naming convention: 'index_of_papers_for_lables_{index_no}.csv'
    """
    logger = logging.getLogger(__name__)
    logger.info("Make an index of random papers in the arxiv dataset.")

    raw_data_dir = project_dir / "data/raw"
    metadata_file_path = raw_data_dir / args.metadata_name

    # assert that arxiv_data_path exists
    assert metadata_file_path.exists(), f"{metadata_file_path} does not exist."

    # assert that arxiv_data_path is either a csv or a json file
    assert metadata_file_path.suffix in [
        ".csv",
        ".gz",
        ".json",
    ], f"{metadata_file_path} is not a csv, gz, or json file."

    if metadata_file_path.suffix == ".json":
        df = parse_json(metadata_file_path)
    else:
        df = load_metadata_csv(metadata_file_path)

    # filter by category
    if args.regex_pattern_cat:
        print(args.regex_pattern_cat)
        df = filter_by_category(df, regex_pattern_cat=args.regex_pattern_cat)

    # filter by date
    if args.start_date:
        start_date = args.start_date
    else:
        start_date = None

    if args.end_date:
        end_date = args.end_date
    else:
        end_date = None

    df = filter_by_date(df, start_date, end_date)

    # filter by license type
    if args.license_filter_list:
        license_filter_list = ast.literal_eval(args.license_filter_list)
        df = filter_by_license(df, license_filter_list)
    else:
        pass

    index_file_dir = project_dir / "data/processed/labels/index_files"
    index_file_dir.mkdir(parents=True, exist_ok=True)

    df_unique, save_name = select_random_papers(
        df,
        index_file_dir,
        check_duplicates=True,
        save_csv=True,
        save_name=None,
        n_papers=args.n_papers,
    )

    save_index_parameters(save_name, args, index_file_dir)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    parser = argparse.ArgumentParser(description="Build data sets for analysis")

    parser.add_argument(
        "--metadata_name",
        type=str,
        default="arxiv-metadata-oai-snapshot.csv.gz",
        help="Name of the metadata file for the arxiv dataset",
    )

    parser.add_argument(
        "--regex_pattern_cat",
        type=str,
        help="Regex pattern for filtering by category. e.g. '\beess|\bcs'",
    )

    # argument for start_date and end_date
    parser.add_argument(
        "--start_date",
        type=str,
        help="Start date for filtering by date. e.g. '2020-08-01'",
    )

    parser.add_argument(
        "--end_date",
        type=str,
        help="End date for filtering by date. e.g. '2021-04-01'",
    )

    # argment for license_filter_list
    parser.add_argument(
        "--license_filter_list",
        type=str,
        help="Input as list of license types. e.g. ['cc by 4.0', 'cc0 1.0']",
    )

    # argument for n_papers to select and save in index_file_dir
    parser.add_argument(
        "--n_papers",
        type=int,
        default=10,
        help="Number of papers to select and save as an index file",
    )

    args = parser.parse_args()

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()
