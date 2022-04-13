# -*- coding: utf-8 -*-
import logging
from pathlib import Path
import kaggle



def main():
    """
    Download the arxiv dataset from kaggle. Uses kaggle api.
    Refer to kaggle api github for setup: https://github.com/Kaggle/kaggle-api
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # download using kaggle api https://stackoverflow.com/a/54869077
    save_dir = project_dir / 'data/raw'
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files('Cornell-University/arxiv', path=save_dir, unzip=True)



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()