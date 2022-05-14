#!/bin/bash
if [ ! -d ~/scratch/arxiv-code-search ]; then
    echo "arxiv-code-search folder in scratch does not exist - making"
    mkdir -p ~/scratch/arxiv-code-search
fi

echo "copying data/processed to scratch"
mkdir -p ~/scratch/arxiv-code-search/data/processed
cp -r ./data/processed ~/scratch/arxiv-code-search/data