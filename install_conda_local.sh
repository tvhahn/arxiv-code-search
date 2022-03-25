#!/bin/bash
conda install -n base -c conda-forge mamba
mamba env create -f envarxiv.yml
eval "$(conda shell.bash hook)"
conda activate arxiv
pip install -e .