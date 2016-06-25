# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 17:42:51 2016

@author: Hugo
"""
import numpy as np
import scipy.sparse
# fonction pour créer les n_grams

# cette fonction permet de calculer la taille de la matrice en sortie
def size_n_gram_matrix(X,n):
    max_len = 0
    for i in range(0,X.shape[0]):
        words = X[i].split()
        if len(words)>max_len:
            max_len = len(words)
    return max_len-(n-1)

# cette fontion retourne une matrice de n_gram ou on assigne la valeur nan
# pour les n_grams n'existant pas
def compute_ngrams(X, n):
    Result_mat = np.empty((X.shape[0],size_n_gram_matrix(X,n),n),dtype='a100')
    for i in range(0,X.shape[0]):
        words = X[i].split()
        print(np.array(create_n_grams(words,n)))
        Result_mat[i,0:len(create_n_grams(words,n))] = np.array(create_n_grams(words,n))
        Result_mat[i,len(create_n_grams(words,n))::] = np.nan
    return Result_mat
    
def vectorize_n_grams(X, n):
    # Creation de l'ensemble des n_grams dans tous les exemples
    global_set = set()
    for sentence in X:
        global_set =  global_set | create_n_grams(sentence, n)
    
    # On compte l'occurence de chaque n_gram
    global_array = np.array(list(global_set))
    n_samples = X.shape[0]
    n_ngrams = global_array.shape[0]
    
    i_array = []
    j_array = []
    data = []
    for i, sentence in enumerate(X):
        for j, n_gram in enumerate(global_array):
            counter = sentence.count(n_gram)
            if counter != 0:
                i_array.append(i)
                j_array.append(j)
                data.append(counter)
    
    count_vectorize = scipy.sparse.coo_matrix((data, (i_array, j_array)), shape=(n_samples, n_ngrams))
    return count_vectorize, global_array
    
# cette fonction calcule les n_grams pour une liste de mots donnés
def create_n_grams(input_list, n):
    n_gram_set = set()
    for i in range(len(input_list) - n + 1):
        n_gram_set.add("".join([input_list[i+j] for j in range(n)]))
    return n_gram_set
    
if __name__ == '__main__':
    s1 = "Je vais enculer"
    s2 = "Est ce que tu suces Porta ?"
    s3 = "Je vais voir ce que je peux faire"
    X = np.array([s1, s2, s3])
    
    mat = vectorize_n_grams(X, 4)[0]
    
    from tfidfvectorizer import TfidfVectorizer
    tf = TfidfVectorizer()
    
    print(tf.transform(mat))