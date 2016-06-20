# -*- coding: utf-8 -*-
"""
Hugo, Nicolas, Mohammed

Utils
"""
import numpy as np
def cross_val_score(estimator,X,y,cv=5):
    size_fold = X.shape[0]//cv
    data = np.c_[X,y]
    samples_X = []
    samples_Y = []
    a = 0
    b = size_fold
    np.random.shuffle(data)
    X_shuffle = data[:,0:X.shape[1]]
    y_shuffle = data[:,X.shape[1]]
    scores = []
    for i in range(cv):
        X_test =  X_shuffle[a:b,:]
        y_test = y_shuffle[a:b]
        X_train = np.ma.array(X_shuffle,mask=False)
        y_train = np.ma.array(y_shuffle,mask=False)
        X_train.mask[a:b,:] = True
        y_train.mask[a:b] = True
        X_train = X_train.compressed().reshape((X.shape[0]-size_fold,X.shape[1]))
        y_train = y_train.compressed()
        estimator.fit(X_train,y_train)
        scores.append(estimator.score(X_test,y_test))
        a = b
        b = b + size_fold
    return scores
