#!/bin/bash

PROJECT_DIR=$1

python src/data/make_paper_index.py \
    --metadata_name "metadata_subsample_50k.csv" \
    --n_papers 500 \
    --license_filter_list "['CC BY 4.0', 'CC0 1.0', 'CC']" \
    --regex_pattern_cat "\beess|\bcs|\bastro-ph|\bstat|\bphysics" \
    # --start_date "2020-01-01" \
    # --end_date "2020-12-31"
