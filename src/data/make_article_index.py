# -*- coding: utf-8 -*-
import logging
from pathlib import Path
import json
import pandas as pd
from src.data.utils import parse_json


def main():
    """
    Parse the json file downloaded from kaggle and save it as a csv file.
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')


    raw_data_dir = project_dir / 'data/raw'
    arxiv_json_path = raw_data_dir / "arxiv-metadata-oai-snapshot.json"

    df = parse_json(arxiv_json_path)

    # save the dataframe as a csv file and compress
    df.to_csv(raw_data_dir / 'arxiv-metadata-oai-snapshot.csv.gz', index=False, compression='gzip')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()