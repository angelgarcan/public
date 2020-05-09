#!/usr/bin/env python
# coding: utf-8

from collections import defaultdict
import datetime
import os
import pickle
import random
import re

from microtc.utils import tweet_iterator
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

import numpy as np
import scipy.sparse as sp

# import matplotlib as plt
# import pandas as pd


#################
# MISC
#################
# Imprime el progreso dentro de un ciclo for.
def show_progress(showProgressEach, tx, idx, legend='', ret_string=False):
    if showProgressEach is not None and idx % showProgressEach == 0:
        tx1 = datetime.datetime.now()
        dx = (tx1 - tx).total_seconds()
        label=f"{tx1} :: {dx} - Processing item #{idx} {legend}"
        if ret_string: 
            return tx1,label
        else:
            print(label)
            return tx1,None
    elif ret_string: return tx,None

# Convierte una matriz dispersa en un numpy.array.
def mtxvec2array(mtxvec):
    return np.squeeze(np.asarray(mtxvec.todense()))

# Cuenta el total de elementos no cero en una matriz dispersa triangular.
def count_nonzero_triangular(triangular_mtx):
    nprocs=triangular_mtx.count_nonzero()
    nprocs_diag=np.array([triangular_mtx[i,i] for i in range(triangular_mtx.shape[0])]).nonzero()[0].shape[0]
    nprocs-=nprocs_diag
    nprocs/=2
    nprocs+=nprocs_diag
    return int(nprocs)

def cleanStr(line):
    # Limpiando texto.
    text = line.lower()
    text = re.sub(r'\n', ' ', text)
    # Acentos.
    text = re.sub(r'á', 'a', text)
    text = re.sub(r'é', 'e', text)
    text = re.sub(r'í', 'i', text)
    text = re.sub(r'ó', 'o', text)
    text = re.sub(r'ú', 'u', text)
    # Caracteres raros.
    text = re.sub(r'ü', 'u', text)
    text = re.sub(r'ñ', 'n', text)
    text = re.sub(r'http[^ ]+', '', text)  # URLs.
#     text = re.sub(r'@[^ ]+', '', text)  # Citas.
#     text = re.sub(r'#[^ ]+', '', text)  # Tags.
    text = re.sub(r'[^a-z ]', ' ', text).strip()  # Signos de puntuación.
    text = re.sub(r'(.)\1\1+', r'\1\1', text)  # Caracteres repetidos 3 veces o más.
    text = re.sub(r' [^ ] ', ' ', text)  # Caracteres huérfanos.
    return text

stopwords = stopwords.words('spanish')
def removeStopWords(txtLst):
    new_txtLst = txtLst[:]
    idx = 0
    while idx < len(new_txtLst):
        w = new_txtLst[idx]
        if w in stopwords:
            new_txtLst.remove(w)
        else:
            idx += 1
    return new_txtLst

stemmer = SnowballStemmer('spanish')
def stemming(txtLst):
    new_txtLst = txtLst[:]
    for idx, w in enumerate(new_txtLst):
        new_txtLst[idx] = stemmer.stem(w)            
    return new_txtLst

# Limpia y tokeniza una línea de texto.
def process_line(line, remStopw=True, stemm=True, retSet=False):
    # Limpiando text.
    twTxtLst = cleanStr(line).split()
    if remStopw: twTxtLst = removeStopWords(twTxtLst)
    if stemm: twTxtLst = stemming(twTxtLst)
    if retSet:
        return set(twTxtLst)
    else:
        return twTxtLst

class InvertedIdx:
    tokenized_docs_filename='tokenized_docs.pickle'
    tokenized_docs=None
    idx_mtx=None
    postlists=None
    docs_norms=[]

    def __init__(self, filename):
        self.filename = filename
    
    def process(self, showProgressEach=None, remStopw=True, stemm=True, replace=False):
        """Procesa todos los tweets, los tokeniza y almacena en un archivo pickle."""
        if self.tokenized_docs:            
            print(f"** Processed. tokenized_docs is already loaded.")
            return
        
        # Si existe el archivo. 
        if os.path.isfile(self.tokenized_docs_filename) and not replace:
            with open(self.tokenized_docs_filename, 'rb') as f:
                self.tokenized_docs= pickle.load(f)   
            print(f"** Processed. {os.path.abspath(self.tokenized_docs_filename)} already exists. tokenized_docs loaded.") 
            return
        
        print(f"** Processing {self.filename} ...")
        self.tokenized_docs=[]
        tx = datetime.datetime.now()
        for idx, tw in enumerate(tweet_iterator(self.filename)):
            show_progress(showProgressEach, tx, idx)
            twTokens = process_line(tw['text'], remStopw, stemm)
            self.tokenized_docs.append(twTokens)
#             if not twTokens:
#                 print(idx, twTokens, tw['text'])

        self.n_tw = idx + 1
        self.docs_range = set(range(1, self.n_tw + 1))
        
        with open(self.tokenized_docs_filename, 'wb') as f:
            pickle.dump(self.tokenized_docs, f)            
        print(f"** Processed {self.n_tw} lines. Saved to {os.path.abspath(self.tokenized_docs_filename)}.")
        
    def compute_mtx(self, n_poda=0):
        """
        Genera la matriz self.idx_mtx que contiene los vectores de cada palabra.
        Almacena la matriz resultante en un archivo pickle.
        """
        if self.idx_mtx is None:
            print("** Computing TF-IDF...")
            tfidf_vectorizer=TfidfVectorizer(use_idf=True,
                                             norm='l2',
                                             analyzer='word',
                                             tokenizer=lambda x: x,
                                             preprocessor=lambda x: x,
                                             token_pattern=None) 
            self.idx_mtx=sp.lil_matrix(tfidf_vectorizer.fit_transform(self.tokenized_docs).T)
            self.corpus=tfidf_vectorizer.get_feature_names()
            print("** Computed TF-IDF Matrix !!!")
        else:            
            print("** TF-IDF Matrix was already computed !!!")
        
        if n_poda != 0:        
            print(f"** Starting poda{n_poda} TF-IDF Matrix...")
            tx = datetime.datetime.now()
            for idx_w, _ in enumerate(self.corpus):
                show_progress(1000, tx, idx_w)
                vd=np.asarray(self.idx_mtx.getrow(idx_w).todense()).flatten()
                cols=vd.argsort()[:-n_poda]
                vd[cols]=0
                self.idx_mtx[idx_w,:]=vd
        
        filename=f"poda{n_poda}_mtx.pickle"
        with open(filename, 'wb') as f:
            pickle.dump(self.idx_mtx, f)
        print(f"** Done poda{n_poda} TF-IDF Matrix !!!")
        print(f"{filename} written.")
            
        return self.idx_mtx

    def compute_plists(self,n_poda=0):
        """
        Computa las listas de posteo para el tamaño de poda n_poda.
        Guarda las listas de posteo en la variable self.postlists.
        Almacena las listas de posteo resultantes en un archivo pickle.
        """
        print(f"** Computing Posting lists with poda{n_poda} ...")
        self.postlists=defaultdict(list)
        tx = datetime.datetime.now()
        for idx_w, w in enumerate(self.corpus):
            show_progress(10000, tx, idx_w)
            vec=np.asarray(self.idx_mtx.getrow(idx_w).todense()).flatten()
            plen=0
            for idx_v in reversed(vec.argsort()):
                v = vec[idx_v]
                if v == 0 : break
                self.postlists[w].append((idx_v,v))
                plen += 1
                if n_poda != 0 and plen == n_poda: break
        
        filename=f"poda{n_poda}_plist.pickle"
        with open(filename, 'wb') as f:
            pickle.dump(self.postlists, f)
        print(f"{filename} written.")
        print(f"** Computed Posting lists with poda{n_poda} !!!")    
        return self.postlists
    
    def plist_index(self, token):
        """Obtiene solo los índices en los que ocurre el token dado."""
        return [idx for idx, _ in self.postlists[token]]
    
    def a(self, q):
        """Realiza la búsqueda AND de todos los tokens q."""
        d = self.docs_range
        for word in q:
            d = d.intersection(self.plist_index(word))
        return list(sorted(d))

    def o(self, q):
        """Realiza la búsqueda OR de todos los tokens q."""
        d = set()
        for word in q:
            d = d.union(self.plist_index(word))
        return list(sorted(d))

    def getTw(self, n):
        """Obtiene el tweet en la posición n."""
        for idx, tw in enumerate(tweet_iterator(self.filename)):
            if idx == n: return tw;
    
    def getDocs(self, nTweets):
        """Obtiene el(los) documento(s) tokenizado del tweet en la posición n."""
        return [self.tokenized_docs[i] for i in nTweets]

    # ONLY FOR DEBUGGING.
    def printPostlist(self, u, printText=False):
        A = self.sorted_inv_indx[u]
        print(u, len(A), A)
        if printText:
            for idx in A:
                print(idx, self.getTw(idx)['text'])

# os.system('rm -f *.pickle')
# os.system('ls -l *.pickle')
#
# inv_idx = InvertedIdx("data\haha_train_ft_pre.json")
# inv_idx.process(showProgressEach=10000, stemm=False)
#
# inv_idx.compute_mtx()
# pd.DataFrame(inv_idx.idx_mtx.todense(), index=inv_idx.corpus).T
#
# n_docs = inv_idx.idx_mtx.shape[1]
# n_pairs = n_docs ** 2 - n_docs
# docs_prods = sp.lil_matrix((n_docs, n_docs))
#
# docs_idx = []
# docs_val = []
# for j in n_docs:
#     docs_idx.append(inv_idx.idx_mtx[:, j].nonzero()[0])
#     docs_val.append(np.asarray(inv_idx.idx_mtx[docs_idx[j], j].todense()).reshape(-1))
#
#
# def plot_queries(qs):
#     """
#     Se hace una consulta OR con el query query.
#     Se obtiene la representación BOW de índices y valores del query.
#     Se obtiene la representación BOW de índices y valores de los documentos resultado de la consulta OR.
#     Se calcula la similitud coseno entre el query y los documentos.
#     Se acumulan el valor de la similitud coseno recién calculada.
#     Se grafican los valores acumulados en el histograma.
#     """
#     sims = []
#     tx = datetime.datetime.now()
#     print(f"Preprocessed {count_nonzero_triangular(docs_prods)} of {n_pairs} docs pairs.")
#     n = 0
#     for i in qs:
#         q = inv_idx.tokenized_docs[i]
#         ds = inv_idx.o(q)  # Consulta OR.
#         qi = docs_idx[i]
#         qv = docs_val[i]
#         qi_set = set(qi)
#         for j in ds:
#             if i == j: continue
#             dot_sum = 0
#             # Si existe el valor calculado se consulta y no se calcula de nuevo.
#             if docs_prods[i, j] != 0:
#                 dot_sum = docs_prods[i, j]
#             # Si no existe se calcula.
#             else:
#                 # Se recuperan los vectores preprocesados.
#                 di = docs_idx[j]
#                 dv = docs_val[j]
#                 # Multiplicación de cada elemento.
#                 a = [qv[np.where(qi == x)] * dv[np.where(di == x)] for x in qi_set.intersection(di)]
#                 # Suma de todos los elelemtos.
#                 dot_sum = np.sum(a)
#                 # Se almancena el valor en una matriz de manera simétrica.
#                 docs_prods[i, j] = docs_prods[j, i] = dot_sum
#             sims.append(dot_sum)  # Se almacena el producto punto para no calcularlo de nuevo.
#
#         tx, label = show_progress(100, tx, n, ret_string=True)
#         if label is not None:
#             print(label + f"\n\tQuery {n} of {len(qs)} with {len(ds)} results. " +
#                    f"Preprocessed {count_nonzero_triangular(docs_prods)} of {n_pairs} docs pairs.")
#         n += 1
#
#     _ = plt.hist(sims, bins=list(np.arange(0, 0.4, 0.01)))
#     plt.show()
#
#
# k = 1000  # Documentos a elegir.
# qs = random.sample(range(0, len(inv_idx.tokenized_docs)), k)
# queries = [inv_idx.tokenized_docs[i] for i in qs]
#
# _ = inv_idx.compute_plists()
# print(f"Longitud de posting list 'amlo'= {len(inv_idx.postlists[process_line('amlo')[0]])}")
#
# plot_queries(qs)
