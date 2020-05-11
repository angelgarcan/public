import numpy as np
#from scipy.spatial.distance import cosine
from sklearn.cluster import KMeans as skmeans
from matplotlib import transforms
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
import nltk, itertools, string, unicodedata, re
from nltk.tokenize import word_tokenize
from sklearn.decomposition import PCA
import pandas as pd
import faiss 
from mpl_toolkits.mplot3d import axes3d


#Tabla de elementos de puntuación
punctuation_table = str.maketrans({key: None for key in string.punctuation+"¿¡?!"})
#Stop words sin acentos
stop_words = [unicodedata.normalize('NFKD', stw).encode('ASCII', 'ignore').decode()
              for stw in nltk.corpus.stopwords.words('spanish')]

#distancia euclideana
def euclidiana(x,y):
    m=x-y
    return np.sqrt(np.sum(m*m))

#distancia coseno
def coseno(x,y):
    dist=1.0 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    # si los vectores no están normalizados se podría utilizar la siguente linea
    #dist=1.0 - np.dot(x, y)
    return dist

# función para graficar clusters en dos y tres dimensiones
def plotClusters(data,labels,centroids={},f="",centroids_txt_labels={}):
    fig=plt.figure(figsize=(5, 5))
    sbox = dict(boxstyle='round', facecolor='white', alpha=0.4)
    d=len(data[1])
    if d==3:
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.add_subplot(111)
    K=np.unique(labels)
    color_map=iter(cm.viridis(np.linspace(0,1,len(K))))
    for k in K:
        D=data[np.where(labels==k)]
        x,y=D[:,0],D[:,1]
        cl=next(color_map)
        if d==3:
            z=D[:,2]
            ax.scatter(x,y,z, color=cl,s=32)
        else:
            ax.scatter(x,y, color=cl,s=32)
        if len(centroids):
            txt_label=centroids_txt_labels and str(centroids_txt_labels[k]) or str(k)
            if len(centroids[k])==3:
                xc,yc,zc=centroids[k]
                ax.text(xc,yc,zc,txt_label,bbox=sbox,fontsize=14)
            else:
                xc,yc=centroids[k]
                ax.text(xc,yc,txt_label,bbox=sbox,fontsize=14)
    if len(data[0])==3:
        ax.set_zticks([])
    ax.set_xticks([])
    ax.set_yticks([])
    if f:
        fig.savefig(f)

# Transformar datos N>3 dimensionales a 2 o tres dimensiones usando PCA                  
def plotPCA(data, labels, d=2,f="",centroids={},vectors=False):
    pca = PCA(n_components=d)
    pca.fit(data)
    X=pca.transform(data)
    origin2d=[0],[0]
    origin3d=[0],[0],[0]
    pca_centroids={}
    for k,c in centroids.items():
        pca_centroids[k]=pca.transform([centroids[k]])[0,:]
    plotClusters(X,labels,f=f,centroids=pca_centroids)
    if len(centroids)>0 and vectors:
        for k,c in pca_centroids.items():
            if d==2:
                plt.quiver(*origin2d, pca_centroids[k][0],pca_centroids[k][1],angles='xy',
                        scale_units='xy', scale=1, color='skyblue')
            else: 
                plt.quiver(*origin3d, pca_centroids[k][0],pca_centroids[k][1],pca_centroids[k][2],color='skyblue')

# Preprocesamiento simple
def preprocess(sentence):
     st=sentence.lower()
     st=re.sub(r"http\S+", "", st)
     st=st.translate(punctuation_table)   
     st=unicodedata.normalize('NFKD', st).encode('ASCII', 'ignore').decode()
     tokens=[word for word in word_tokenize(st) if word not in stop_words]
     return tokens

#Para tokenizar una lista de textos                            
def tokenize_sentences(texts):
    tokenized_texts=[preprocess(txt) for txt in texts]
    return np.array(tokenized_texts)

#Método bruto para calcular  atriz de cocurrencia de palabrar 
def cocurrency_matrix(sentences):
    voc=list(set(list(itertools.chain.from_iterable(sentences))))
    voc.sort()
    V=len(voc)
    matrix=pd.DataFrame(index=voc,columns=voc,data=np.zeros((V,V)))
    for word in voc:
        for sentence in sentences:
            if word in sentence:
                for col in set(sentence):
                    matrix[word][col]+=1
                    #if col!=word:
                    #    matrix[col][word]=matrix[col][word]+1
    return np.array(voc),matrix

def plotDecisionBoundary(clf,data,labels,points=True):
    n,m=data.shape
    fig=plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    K=len(set(labels))
    color_map=list(iter(cm.viridis(np.linspace(0,1,K))))
    if m>2:
        pca = PCA(n_components=2)
        pca.fit(data)
        X=pca.transform(data)
    else:
        X=data
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
    if m>2:
        Z = clf.predict(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()]))
    else:
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.4)
    for j,c in zip(np.unique(labels),color_map):
        idx = np.where(labels == j)
        #print(c)
        if points:
            ax.scatter(X[idx][:, 0], X[:, 1][idx], c = [c], label = j)
    #ax.scatter(X[:, 0], X[:, 1], c=labels, s=20, edgecolor='k',label=list(set(labels)))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc='lower right');
    plt.show()
                
class NearestCentroid:
        
    # Encontar el centroide más cercano
    def _find_nearest_centroid(self,sample):
        #Calcular las distancias con respecto a cada centroide
        dists=[(self.distance(c,sample),i) for i,c in self.centroids_.items()]
        #ordenar
        dists.sort()
        #regresar el indice del elemento con la menor distancia
        return dists[0][1] 
        
    # Ejemplo del la formula del promedio
    def Average(self, **kwargs):
        lb=list(set(self.labels))
        self.centroids_={}
        for j in lb:
            Gj=self.data[np.where(self.labels==j)]
            #print(lb,j,Gj.shape)
            self.centroids_[j]=np.sum(Gj,axis=0)/len(Gj)
        return self
        
    def Sum(self,**kwarg):
        lb=list(set(self.labels))
        self.centroids_={}
        for j in lb:
            Gj=self.data[np.where(self.labels==j)]
            #print(lb,j,Gj.shape)
            self.centroids_[j]=np.sum(Gj,axis=0)
        return self

    #Para Rocchio debería poder pasar los parametros beta y gamma (para eso lo **kwargs)
    def Rocchio(self,beta=16,gamma=4,**kwargs):
#         print("kwargs:",kwargs)
#         print("beta gamma ->",beta,gamma)
        lb=list(set(self.labels))
        self.centroids_={}
        for j in lb:
            Gj=self.data[np.where(self.labels==j)]
            Gnj=self.data[np.where(self.labels!=j)]
            #print(lb,j,Gj.shape)
            self.centroids_[j]=beta*(np.sum(Gj,axis=0)/len(Gj))+gamma*(np.sum(Gnj,axis=0)/len(Gnj))
        return self
        
    def NormSum(self,**kwarg):
        lb=list(set(self.labels))
        self.centroids_={}
        for j in lb:
            Gj=self.data[np.where(self.labels==j)]
            #print(lb,j,Gj.shape)
            s=np.sum(Gj,axis=0)
            self.centroids_[j]=s/np.linalg.norm(s)
        return self
    
    # Predecir las etiquetas para un conjunto de datos
    def predict(self,unlabeled_samples):
        y=[self._find_nearest_centroid(sample) for sample in unlabeled_samples]
        return np.array(y)

    # Metodo para entrenar el modelo, solo recibe un numpy.array con los dato de n x N.
    # Donde n es el número de elmentos y N la dimensión            
    def fit(self,data,labels):
        self.data=data
        self.labels=labels    
        self.algorithm(**self.kwargs_)
        return self

    #estructura propuesta para los algoritmos
    # La variable centroid_type es un string con el nombre de su función de calculo de centroides
    def __init__(self,distance='euclidiana',centroid_type='Average', **kwargs):
        #Funcion de similitud/distancia, por defecto similitud coseno
        self.distance=eval(distance) 
        self.algorithm=getattr(self, centroid_type)
        self.kwargs_=kwargs
    
    
class kNN:
    def _uniform(self,unlabeled_samples):
        samples=unlabeled_samples
        if self.distance=='coseno':
            vnorm=np.linalg.norm(samples,axis=1)
            samples=samples/vnorm.reshape(len(vnorm),1)
        neighbors,n_ids=self.index.search(samples,self.k)
#         print(neighbors.shape)
#         print(neighbors[:3])
#         print(n_ids.shape)
#         print(n_ids[:3])
#         print(np.max(n_ids))
#         print(np.min(n_ids))
        labels=[np.argmax(np.bincount(self.labels[n_id])) for n_id in n_ids]
        return np.array(labels)
    
    def _mean_dist(self,unlabeled_samples):
        print("Implementación mean_dist")
        samples=unlabeled_samples
        if self.distance=='coseno':
            vnorm=np.linalg.norm(samples,axis=1)
            samples=samples/vnorm.reshape(len(vnorm),1)
        self.n_dists,self.n_ids=self.index.search(samples,self.k)
        
        labels=[]
        for i,n_id in enumerate(self.n_ids):
            n_class=self.labels[n_id]

            cs=list(set(n_class))
            if len(cs)==1:
                labels.append(cs[0])
                continue

            mean_dists=[]
            for j in cs:
                mean_dists.append(
                    np.mean(self.n_dists[i][np.where(n_class==j)])
                )
            labels.append(cs[np.argmin(mean_dists)])
        return np.array(labels)
    
    def _weighed_dist(self,unlabeled_samples):
        print("Implementación weighed_dist")
        samples=unlabeled_samples
        if self.distance=='coseno':
            vnorm=np.linalg.norm(samples,axis=1)
            samples=samples/vnorm.reshape(len(vnorm),1)
        self.n_dists,self.n_ids=self.index.search(samples,self.k)
        
        labels=[]
        for i,n_id in enumerate(self.n_ids):
            n_class=self.labels[n_id]

            cs=list(set(n_class))
            if len(cs)==1:
                labels.append(cs[0])
                continue

            ws=[]
            for j in cs:
                ws.append(
                    np.sum(
                        np.reciprocal(self.n_dists[i][np.where(n_class==j)],
                    dtype='float32')
                    )
                )
            labels.append(cs[np.argmax(ws)])
        return np.array(labels)
        
    def predict(self,unlabeled_samples):
        return self.weight(unlabeled_samples.astype('float32'))
        
    def fit(self,data,labels):
        # faiss solo acepta float32
        self.data=data.astype('float32')
        self.labels=labels
        n,d=self.data.shape
        # si se utiliza distancia coseno deben normalizarse los vectores
        if self.distance=="coseno":
            vnorm=np.linalg.norm(self.data,axis=1)
            self.index=faiss.IndexFlatIP(d) # indice que utiliza el producto punto
            self.data=self.data/vnorm.reshape(len(vnorm),1)
        else:
            self.index= faiss.IndexFlatL2(d) # indice que utiliza L2
        self.index.add(self.data) 
        return self
    
    def __init__(self,k=1,distance='coseno',weight_type='uniform'):
        #self.function=function #Funcion de similitud/distancia, por defecto similitud coseno
        self.weight=getattr(self, '_{}'.format(weight_type))
        self.distance=distance
        self.k=k
    
    
# Plantilla simple para implementar métodos de clustering
class Clustering:
    
    ## Calcular SSE se usa inertia igual que en la implementacion de sckit-learn 
    def _inertia(self):
        self.inertia_=0
        for j in range(len(self.data)):
            dists=[(self.distance_function(c,self.data[j]),i) for i,c in self.centroids_.items()]
            self.inertia_+=dists[0][0]
            
    # asigna los elementos en la colección a su centroide más cercano
    # genera las etiquetas de los clusters 
    def _assign_nearest_centroids(self):
         self.labels_=[-1 for x in self.data]
         for j in range(len(self.data)):
             dists=[(self.distance_function(c,self.data[j]),i,self.data[j])
                    for i,c in self.centroids_.items()]
             dists.sort()
             self.labels_[j]=dists[0][1]
       
    # Ejemplo de random Clustering, es equivalente a la primera iteración de KMeans
    def randomClustering(self):
         # seleccionamos K elmentos de forma aleatoria
         idx=np.random.randint(self.data.shape[0], size=self.n_clusters)
         self.centroids_=dict(zip(idx,self.data[idx,:])) # creamos un diccionario {id_cluster: vector} 
         self._assign_nearest_centroids() #asignamos las etiquetas
         self._inertia() # calculamos el SSE
         return self

    def KMeans(self):
         print("Su implentación de KMeans")

    def FFTraversal(self):
        print("su implementación de Farthest First Traversal") 

    # Metodo para entrenar el modelo, solo recibe un numpy.array con los dato de n x N.
    # Donde n es el número de elmentos y N la dimensión            
    def fit(self,data):
        self.data=data
        self.algorithm()
        return self

    #estructura propuesta para los algoritmos
    # La variable algorithm es un string con el nombre de su función de clustering
    def __init__(self,n_clusters=3,distance_function=euclidiana,algorithm='randomClustering'):
        self.n_clusters=n_clusters  # número de clusters K
        self.inertia_=0 # SSE
        self.distance_function=distance_function #Funcion de distancia, por defecto euclidiana
        self.algorithm=getattr(self, algorithm) 
