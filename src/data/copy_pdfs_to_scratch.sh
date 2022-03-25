#!/bin/bash
DIR="~/scratch/doi-scrape"
if [ ! -d "$DIR" ]; then
    echo "doi-scrape folder in scratch does not exist"
    mkdir ~/scratch/doi-scrape
fi

cd ..
cd ..

cp -r ./data/raw ~/scratch/doi-scrape