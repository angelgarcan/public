#!/bin/bash

set -x
# PIP
pip install mkl
pip install faiss-cpu

# CONDA
conda update --yes -n base conda

conda install --yes matplotlib pandas numpy scikit-learn nltk jupyternotify wordcloud gensim
conda install --yes -c ingeotec microtc

conda update --yes --all

set +x