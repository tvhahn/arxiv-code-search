#!/bin/bash

PROJECT_DIR=$1

python src/data/make_article_index.py \
    --metadata_name "metadata_subsample_50k.csv" \
    --n_articles 100 \
    --license_filter_list "['CC BY 4.0', 'CC0 1.0', 'CC']" \
    --regex_pattern_cat "\beess|\bcs" \
    --start_date "2020-01-01" \
    --end_date "2020-12-31"
