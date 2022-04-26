#!/bin/bash
module load python/3.8
virtualenv ~/arxiv
source ~/arxiv/bin/activate
pip install --upgrade pip setuptools wheel
pip install --no-index pandas scipy scikit_learn matplotlib seaborn torch torchvision torchtext torchaudio transformers
pip install --no-index jupyterlab requests beautifulsoup4 lxml black tqdm nltk ipywidgets
pip install pdfminer.six kaggle
pip install -e .

# create bash script for opening jupyter notebooks https://stackoverflow.com/a/4879146/9214620
cat << EOF >$VIRTUAL_ENV/bin/notebook.sh
#!/bin/bash
unset XDG_RUNTIME_DIR
jupyter-lab --ip \$(hostname -f) --no-browser
EOF

chmod u+x $VIRTUAL_ENV/bin/notebook.sh