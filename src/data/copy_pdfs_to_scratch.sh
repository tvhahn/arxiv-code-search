#!/bin/bash
DIR="~/scratch/arxiv-code-search"
if [ ! -d "$DIR" ]; then
    echo "arxiv-code-search folder in scratch does not exist"
    mkdir ~/scratch/arxiv-code-search
fi

cd ..
cd ..

cp -r ./data/raw ~/scratch/arxiv-code-search