# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 16:56:42 2016

@author: Hugo
"""

import numpy as np
class LogisticRegression2:
 
    def __init__(self,C=0.001):
        """Initializes Class for Logistic Regression
 
        Parameters
        ----------
        X : ndarray(n-rows,m-features)
            Numerical training data.
 
        y: ndarray(n-rows,)
            Interger training labels.
 
        tolerance : float (default 1e-5)
            Stopping threshold difference in the loglikelihood between iterations.
 
        """
        #create weights equal to zero with an intercept coefficent at index 0
        self.C = C
        self.tableau_vraissemblance = []
 
    def prob(self):
        """On calcule la probabilité de P(y=1|x)
        """
        P = 1/(1+np.exp(-np.dot(self.features,self.w)))
        return P
    def probX(self,X):
        
        features = np.ones((X.shape[0],X.shape[1]+1))
        features[:,1:] = X
        P = 1/(1+np.exp(-np.dot(features,self.w)))
        return P
 
    def log_vraissemblance(self):
        """On calcule la log vraissemblace à partir de la proba déjà calculé
        et des labels, on retourne l'inverse
        """
        prob =self.prob()
        Lv = self.labels * np.log(prob) + (1-self.labels) * np.log(1-prob)
        return -1*Lv.sum()
 
    def log_vraissemblance_gradient(self):
        """On calcule la formule du gradient de 
        """
        erreur = self.labels-self.prob()
        grad = erreur*self.features
        gradient = grad.sum(axis=0)
        result = gradient.reshape(self.w.shape) + self.C*self.w
        return result
 
    def log_vraissemblance_gradient_descente(self,alpha=1e-6,max_iter=1e3):
        """Runs the gradient decent algorithm
 
        Parameters
        ----------
        alpha : float
            The learning rate for the algorithm
 
        max_iterations : int
            The maximum number of iterations allowed to run before the algorithm terminates
 
        """
        nbr_iter = 0
        vraissemblance_init = self.log_vraissemblance()
        epsilon = self.tolerance + 1
        self.tableau_vraissemblance.append(vraissemblance_init)
        while (epsilon > self.tolerance) and (nbr_iter < max_iter):
            self.w = self.w + alpha*self.log_vraissemblance_gradient()
        
            vraissemblance = self.log_vraissemblance()
            epsilon = np.abs(vraissemblance_init - vraissemblance)
            vraissemblance_init = vraissemblance
            self.tableau_vraissemblance.append(vraissemblance_init)
            nbr_iter += 1
            print(epsilon)
            print(nbr_iter)
    
    def fit(self,X,y,tolerance=1e-5):
        self.tolerance = tolerance
        self.labels = y.reshape(y.size,1)
        self.w = np.zeros((X.shape[1]+1,1))
        self.features = np.ones((X.shape[0],X.shape[1]+1))
        self.features[:,1:] = X
        self.log_vraissemblance_gradient_descente()
 
    def predict(self,X):
        """Computes the logistic probability of being a positive example
 
        Parameters
        ----------
        X : ndarray (n-rows,n-features)
            Test data to score using the current weights
 
        Returns
        -------
        out : ndarray (1,)
            Probablity of being a positive example
        """
        y = np.zeros(X.shape[0])
        proba_pred = self.probX(X)
        print(proba_pred)
        for i in range(0,X.shape[0]):
            if proba_pred[i] > 0.5:
                y[i]=1.0
        return y
        
    def score(self,X,y):
        n_samples, _ = X.shape
        return np.sum(self.predict(X) == y)/float(n_samples)
                