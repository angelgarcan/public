import numpy as np
import toolbox as tb
import pandas as pd
from gensim.models import FastText
import os
from microtc.utils import tweet_iterator
import datetime
from collections import Counter
import gc
from collections import defaultdict

class Tokens():
    tokenized_pointer=None
    
    def tokenize(self,json_file,npy_file=None,replace=False,showProgressEach=1000):
        self.npy_file=str(os.path.splitext(json_file)[0]+".npy") if not npy_file else npy_file

        if not replace and os.path.isfile(self.npy_file):
            print(f"** Replace is off. {os.path.abspath(self.npy_file)} already exists, then load.")
        else:
            tokenized_docs=[]
            self.N=0
            print(f"** Processing {json_file} ...")
            tx = datetime.datetime.now()
            for idx, tw in enumerate(tweet_iterator(json_file)):
                tb.show_progress(showProgressEach, tx, idx)
                twTokens = tb.process_line(tw['text']) # Tokenizando tweet.
                tokenized_docs.append(twTokens)
            self.N=idx+1
        
            maxLen=len(max(tokenized_docs))
            for i, doc in enumerate(tokenized_docs):
                tokenized_docs[i]=[doc[x] if x<len(doc) else '' for x in range(maxLen)]
            
            np.save(self.npy_file,tokenized_docs)
            del(tokenized_docs)
            gc.collect()
            print(f"** Processed {self.N} lines. Saved to {os.path.abspath(self.npy_file)}.")
        
        self.pointer=np.load(self.npy_file, mmap_mode='r')
        return self
    
    def get(self,n):
        return [x for x in self.pointer[n,:] if x != '']
    
    def getDocs(self,docs):
#         if isinstance(docs,int): docs=[docs]
        res=[]
        for n in docs:
            res.append(self.get(n))
        return res
    
class Index():
    N = 0
    
    def computePostingLists(self, tokens, showProgressEach=1000):
        tx = datetime.datetime.now()
        DF = Counter({})
        TF = []
        
        print("** Counting TF & DF ...")
        for idx in range(tokens.N):
            tb.show_progress(showProgressEach,tx,idx)
            tok=tokens.get(idx)
            twCnt = Counter(tok) # Sumando frecuencias individiales.
            TF.append(twCnt)
            DF.update(list(twCnt.keys())) # Sumando frecuencias por documento (DF).
            
        self.N=idx+1 # Contabilizando documentos.
        self.postlists=defaultdict(list)
        print("** Calculating TF-IDF...")
        for i,c in enumerate(TF):
            tb.show_progress(showProgressEach,tx,i)
            for w,tf in c.items():
                self.postlists[w].append((tf*np.log2(self.N/(DF[w]+1)),i))
        del(TF)
        del(DF)
        gc.collect()
        
        print("** Sorting posting lists...")
        for idx,(w,p) in enumerate(self.postlists.items()):
            tb.show_progress(showProgressEach,tx,idx)
            p.sort()
            p=list(zip(*p))
            self.postlists[w]=[list(p[1]),list(p[0])]
        
        return self
    
    def getIdxs(self,word):
        return self.postlists[word][0]
    
    def getScores(self,word):
        return self.postlists[word][1]
    
    def a(self, q, isText=False):
        """Búsqueda AND."""
        if isText: q=tb.process_line(q)
        res = None
        for word in q:
            if not res:
                res = set(self.getIdxs(word))
            else:
                res = res.intersection(self.getIdxs(word))
        return list(sorted(res))

    def o(self, q, isText=False):
        """Búsqueda OR."""
        if isText: q=tb.process_line(q)
        res = set()
        for word in q:
            res = res.union(self.getIdxs(word))
        return list(sorted(res)) 
