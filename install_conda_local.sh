#!/bin/bash
conda install -n base -c conda-forge mamba
mamba env create -f envdoiscrape.yml
eval "$(conda shell.bash hook)"
conda activate doiscrape
pip install git+https://github.com/neuml/txtai#egg=txtai[pipeline]
pip install -e .