#!/bin/bash

python src/data/make_article_index.py \
    --metadata_name "arxiv-metadata-oai-snapshot.csv" \
    --regex_pattern_cat "q-fin.TR"