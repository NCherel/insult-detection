# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 17:42:51 2016

@author: Hugo
"""

# fonction pour créer les n_grams
def create_n_grams(input_list_words,n):
    n_gram_list = []
    for i in range(len(input_list_words)-n):
        n_gram_list.append([input_list[i+j] for j in range(n)])
    return n_gram_list