# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 20:10:34 2016

@author: Hugo
"""

import numpy as np
import pandas as pd
#from sklearn.linear_model import LogisticRegression
from text_mining.tokenizer import Tokenizer
from text_mining.n_gram_creation import vectorize_n_grams_no_dict
from text_mining.n_gram_creation import vectorize_n_grams_with_dict
from text_mining.tfidfvectorizer import TfidfVectorizer

from classifier.randomforest import RandomForestClassifier
from classifier.LogisticRegression2 import LogisticRegression2
from classifier.LogisticRegression import LogisticRegression

data = pd.read_csv('data/train.csv', header=None)
X = data.iloc[:,1]
y = data.iloc[:,0].values

# Uncomment for test
data_test = pd.read_csv('data/test.csv', header=None)
X_test = data_test.iloc[:,0]

tokenizer = Tokenizer()
X_tokens = tokenizer.tokenize_array(X)
X_test_tokens = tokenizer.tokenize_array(X_test)

X_reconstruct = tokenizer.reconstruct_array(X_tokens)
X_reconstruct_test = tokenizer.reconstruct_array(X_test_tokens)


n_grams, dictionary = vectorize_n_grams_no_dict(X_reconstruct, 3, method='char')
n_grams_test = vectorize_n_grams_with_dict(X_reconstruct_test, dictionary)


tf = TfidfVectorizer()
X_tf = tf.transform(n_grams)
X_tf_test = tf.transform(n_grams_test)

X_tf = X_tf.todense()
X_tf = np.array(X_tf)

X_tf_test = X_tf_test.todense()
X_tf_test = np.array(X_tf_test)

X_train = X_tf
y_train = y



print X_train.shape



logreg = LogisticRegression2()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
np.savetxt('y_pred.txt', y_pred, fmt='%s')

