#!/bin/bash

set -x
# PIP
pip install mkl
pip install faiss-cpu
pip install fasttext==0.8.3

# CONDA
conda update --yes -n base conda

conda install --yes matplotlib pandas numpy scikit-learn nltk jupyternotify wordcloud gensim=3.8.1
conda install --yes -c ingeotec microtc

conda update --yes --all

set +x