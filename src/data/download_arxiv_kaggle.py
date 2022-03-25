# -*- coding: utf-8 -*-
import logging
from pathlib import Path
import kaggle

# download using kaggle api https://stackoverflow.com/a/54869077

def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    save_dir = project_dir / 'data/raw'
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files('Cornell-University/arxiv', path=save_dir, unzip=False)



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()