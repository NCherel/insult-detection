# coding: utf-8

"""
Main script used for the whole process
Authors : Mohamed, Hugo, Nicolas

"""
import numpy as np
import pandas as pd
from text_mining.tokenizer import Tokenizer
from text_mining.n_gram_creation import vectorize_n_grams_no_dict
from text_mining.n_gram_creation import vectorize_n_grams_with_dict
from text_mining.tfidfvectorizer import TfidfVectorizer

from classifier.randomforest import RandomForestClassifier
from classifier.LogisticRegression import LogisticRegression

data = pd.read_csv('data/train.csv', header=None)
X = data.iloc[:,1]
y = data.iloc[:,0]

tokenizer = Tokenizer()
X_tokens = tokenizer.tokenize_array(X)

X_reconstruct = []
for token in X_tokens:
    X_reconstruct.append(tokenizer.reconstruct(token))
    
X_reconstruct = np.array(X_reconstruct)

n_grams, dictionary = vectorize_n_grams_no_dict(X_reconstruct, 3)


tf = TfidfVectorizer()
X_tf = tf.transform(n_grams)

X_tf = X_tf.todense()
X_tf = np.array(X_tf)

X_train = X_tf[:2000]
y_train = y[:2000]

X_test = X_tf[2000:]
y_test = y[2000:]

print X_train.shape

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
print logreg.score(X_test, y_test)
