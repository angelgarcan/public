#!/bin/bash

# PIP
pip install mkl
pip install faiss-cpu

# CONDA
conda update -n base conda
conda install --yes matplotlib pandas numpy scikit-learn nltk jupyternotify
conda install --yes -c ingeotec microtc

conda update --all