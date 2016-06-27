# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 19:54:50 2016

@author: Hugo, Nicolas, Mohammed
"""

import numpy as np
from __future__ import division
# définition du gradient, de la hessienne et de la fonction pour la pénalisation 2
def line_search(w,X,y, a, b, beta):
    l = 1
    wplus = w - b*pow(a,l)*grad_f(w,X,y)
    while f(wplus,X,y) > f(w,X,y) + beta*np.inner(grad_f(w,X,y), wplus - w):
        l = l + 1
        wplus = w - b*pow(a,l)*grad_f(w,X,y)
    return b*pow(a,l)

def f(w2,X,y):
    n,p = X.shape
    rho = 1.0/n

    # X1 est la matrice X à laquelle on a rajouté une colonne de 1
    ones = np.ones((n,1))
    X1 = np.hstack([X,ones])
    w = w2[1:]
    
    return 1.0/n * np.sum(np.log(1 + np.exp(-y*(X1.dot(w2))))) + (rho/2.0)*(np.sum(w**2))

def grad_f(w2,X,y):
    n,p = X.shape
    rho = 1.0/n

    # X1 est la matrice X à laquelle on a rajouté une colonne de 1
    ones = np.ones((n,1))
    X1 = np.hstack([X,ones])
    w = w2[1:]
    
    # On crée le vecteur [0 w]^T
    vect = np.concatenate([[0], w], axis=0)
    
    temp = np.exp(-y*(X1.dot(w2)))
    U = -y*temp/(1.0 + temp)
    
    return 1.0/n*(X1.T.dot(U)) + rho*vect

def hessienne_f(w,X,y):
    n,p = X.shape
    rho = 1.0/n
    ones = np.ones((n,1))
    X1 = np.hstack([X,ones])
    eye = rho*np.eye(p + 1)
    eye[0,0] = 0
    
    temp = np.exp(-y*(X1.dot(w)))
    U = np.eye(n) * (y**2 * temp)/((1+temp)**2)
    
    return 1.0/n*((X1.T.dot(U).dot(X1))) + eye

def newton(x0,X,y, epsilon=10e-10):
    xk = x0
    grad = grad_f(xk,X,y)
    norm = np.sum(grad**2)
    hessienne = hessienne_f(xk,X,y)
    
    grads = []
    grads.append(norm)
    
    while norm > epsilon:
        xk = xk - np.linalg.inv(hessienne).dot(grad)
        grad = grad_f(xk,X,y)
        norm = np.sum(grad**2)
        hessienne = hessienne_f(xk,X,y)
        
        grads.append(norm)
        print(norm)
    return xk

def newton_line_search(x0,X,y, epsilon=10e-10):
    xk = x0
    grad = grad_f(xk,X,y)
    norm = np.sum(grad**2)
    hessienne = hessienne_f(xk,X,y)
    gamma = line_search(xk,X,y,0.5,2,0.5)
    grads = []
    grads.append(norm)
    
    while norm > epsilon:
        xk = xk - gamma*np.linalg.inv(hessienne).dot(grad)
        gamma = line_search(xk,X,y,0.5,2,0.5)
        grad = grad_f(xk,X,y)
        norm = np.sum(grad**2)
        hessienne = hessienne_f(xk,X,y)
        
        grads.append(norm)
        print(norm)
    return xk



class LogisticRegression():
    def __init__(self, penalty='i2',max_iter=100,solver='Newton',tol=0.0001,C=1.0):
        self.penalty = penalty
        self.max_iter = max_iter
        self.solver = solver
        self.tol = tol
        self.C = C
    
    def fit(self,X,y):
        self.label = np.unique(y)
        w_init = np.zeros(X.shape[1])
        w_init0 = 0
        w = np.append(w_init,w_init0)
        if self.solver == 'Newton':
            if self.penalty == 'i2':
                self.coef = newton(w,X,y,self.tol)
            else:
                print("Error of penalisation for the solver")
                
        if self.solver == 'Newton_line':
            if self.penalty == 'i2':
                self.coef = newton_line_search(w,X,y,self.tol)
            else:
                print("Error of penalisation for the solver")

    def predict(self,X):
        y_pred = np.empty((X.shape[0]))
        for i in range(0,X.shape[0]):
            h_X = np.dot(X[i],np.transpose(self.coef[0:X.shape[1]])) + self.coef[X.shape[1]]
            if h_X > 0:
                y_pred[i] = self.label[1]
            else:
                y_pred[i] = self.label[0]
        return y_pred
    
    def score(self,X,y):
        y_pred = self.predict(X)
        score = 0
        for i in range(0,X.shape[0]):
            if y_pred[i] == y[i]:
                score = score + 1
        score = score / X.shape[0]
        return score