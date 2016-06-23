# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 17:42:51 2016

@author: Hugo
"""
import numpy as np
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
def compute_ngrams(X,n):
    Result_mat = np.empty((X.shape[0],size_n_gram_matrix(X,n),n),dtype='a100')
    for i in range(0,X.shape[0]):
        words = X[i].split()
        print(np.array(create_n_grams(words,n)))
        Result_mat[i,0:len(create_n_grams(words,n))] = np.array(create_n_grams(words,n))
        Result_mat[i,len(create_n_grams(words,n))::] = np.nan
    return Result_mat
# cette fonction calcule les n_grams pour une liste de mots donnés
def create_n_grams(input_list,n):
    n_gram_list = []
    for i in range(len(input_list)-n+1):
        n_gram_list.append([input_list[i+j] for j in range(n)])
    return n_gram_list