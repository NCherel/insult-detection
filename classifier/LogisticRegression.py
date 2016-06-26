# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 19:54:50 2016

@author: Hugo, Nicolas, Mohammed
"""

import numpy as np
from __future__ import division
# définition du gradient, de la hessienne et de la fonction pour la pénalisation 2
def func1(X,Y,w,w0,rho):
    N = X.shape[0]
    l = 1/N
    One  = np.ones(X.shape[0])
    X2 = np.c_[X,One]
    w2 = np.append(w,w0)
    res = 0
    
    for i in range(0,X.shape[0]-1):
        prd = np.dot(X2[i],np.transpose(w2))
        res = res + np.log(1+np.exp(-Y[i]*(prd)))
    f1 = l*(res) + (rho/2)*(np.linalg.norm(w)**2)
    return f1

def gradf1(X,Y,w,w0,rho):
    
    N = X.shape[0]
    l = 1/N
    One  = np.ones(X.shape[0])
    X2 = np.c_[X,One]
    w2 = np.append(w,w0)
    w3 = np.append(w,1)
    res = 0
    for i in range(0,X.shape[0]-1):
        prd = np.dot(X2[i],np.transpose(w2))
        if Y[i]*prd<0:
            res = res - Y[i]*(1)/(1+np.exp(Y[i]*(prd)))*X2[i]
        else:
            res = res - Y[i]*(np.exp(-Y[i]*(prd))/(1+np.exp(-Y[i]*(prd))))*X2[i]
    f1 = l*(res) + (rho)*w3
    return f1

def hessf1(X,Y,w,w0,rho):
    
    N = X.shape[0]
    l = 1/N
    One  = np.ones(X.shape[0])
    X2 = np.c_[X,One]
    w2 = np.append(w,w0)
    Ip = np.identity(X.shape[1]+1)
    Ip[X.shape[1]][X.shape[1]] = 0
    res = 0
    for i in range(0,X.shape[0]-1):
        prd = np.dot(X2[i],np.transpose(w2))
        if Y[i]*prd<0:
            res = res + (1/(1 + np.exp(Y[i]*(prd))))*(1-1/(1 + np.exp(Y[i]*(prd))))*np.outer(X2[i],X2[i])
        else:
            res = res + (Y[i]**2)*(np.exp(-Y[i]*(prd)))/((1+np.exp(-Y[i]*(prd)))**2)*np.outer(X2[i],X2[i])
    f1 = l*(res) + (rho)*Ip
    return f1

# définition du gradient et de la fonction pour la pénalisation 2
def func2(X,Y,w,w0,rho):
    N = X.shape[0]
    l = 1/N
    One  = np.ones(X.shape[0])
    X2 = np.c_[X,One]
    w2 = np.append(w,w0)
    res = 0
    
    for i in range(0,X.shape[0]-1):
        prd = np.dot(X2[i],np.transpose(w2))
        res = res + np.log(1+np.exp(-Y[i]*(prd)))
    f1 = l*(res)
    return f1


def gradf2(X,Y,w,w0,rho):
    N = X.shape[0]
    l = 1/N
    One  = np.ones(X.shape[0])
    X2 = np.c_[X,One]
    w2 = np.append(w,w0)
    res = 0
    for i in range(0,X.shape[0]-1):
        prd = np.dot(X2[i],np.transpose(w2))
        if Y[i]*prd<0:
            res = res - Y[i]*(1)/(1+np.exp(Y[i]*(prd)))*X2[i]
        else:
            res = res - Y[i]*(np.exp(-Y[i]*(prd))/(1+np.exp(-Y[i]*(prd))))*X2[i]
    f1 = l*(res)
    return f1

# méthode du gradient proximal
def prox(w,gamma,rho):
    prox = np.empty(w.shape[0])
    prox[w.shape[0]-1] = 0
    for i in range(0,w.shape[0]-2):
        if (abs(w[i]) - gamma*rho) > 0:
            prox[i] = np.sign(w[i])*(abs(w[i]) - gamma*rho)
        else: 
            prox[i] = 0
    return prox

def Xplus2(X,Y,w,rho,gamma):
    w = w-gamma*gradf2(X,Y,w[0:X.shape[1]],w[X.shape[1]],rho)
    return w

def calcL2(X,Y,w,w0,rho,gamma_1):
    w2 = np.append(w,w0)
    a = 0.9
    b = 2*gamma_1
    l=1
    C = b*(a**l)
    Lim = func2(X,Y,w,w0,rho) + np.transpose(gradf2(X,Y,w,w0,rho)).dot(Xplus2(X,Y,w2,rho,C)-w2)+(1/(2*C))*np.sum((Xplus2(X,Y,w2,rho,C)-w2)**2)
    mina = func2(X,Y,Xplus2(X,Y,w2,rho,b*(a**l))[0:X.shape[1]],Xplus2(X,Y,w2,rho,b*(a**l))[X.shape[1]],rho)
    while(mina>Lim):
        l=l+1
        C = b*(a**l)
        Lim = func2(X,Y,w,w0,rho) + np.transpose(gradf2(X,Y,w,w0,rho)).dot(Xplus2(X,Y,w2,rho,C)-w2)+(1/(2*C))*np.sum((Xplus2(X,Y,w2,rho,C)-w2)**2)
        mina = func2(X,Y,Xplus2(X,Y,w2,rho,b*(a**l))[0:X.shape[1]],Xplus2(X,Y,w2,rho,b*(a**l))[X.shape[1]],rho)
    return b*(a**(l-1))

def proxMethod(X,Y,w,w0,rho,epsilon):
    norm = []
    winit = np.append(w,w0)
    gamma = calcL2(X,Y,w,w0,rho,10)
    count = 0
    dist = 100
    while(dist>epsilon):
        A = winit - gamma*gradf2(X,Y,winit[0:X.shape[1]],winit[X.shape[1]],rho)
        temp = winit
        winit = prox(A,gamma,rho)
        gamma = calcL2(X,Y,winit[0:X.shape[1]],winit[X.shape[1]],rho,gamma)
        dist = np.linalg.norm(winit-temp)
        norm.append(dist)
        print(dist)
    return winit, norm

# méthode de la descente de gradient à pas constant
# à implémenter
def calcConstant_step():
    return 0.5

def gradMethod(X,Y,w,w0,rho,epsilon):
    gamma = calcConstant_step()
    winit = np.append(w,w0)
    lim = np.linalg.norm(gradf1(X,Y,w,w0,rho))
    while(lim>epsilon):
        winit = winit - gamma*(gradf1(X,Y,winit[0:X.shape[1]],winit[X.shape[1]],rho))
        lim = np.linalg.norm(gradf1(X,Y,winit[0:X.shape[1]],winit[X.shape[1]],rho))
    return winit

# méthode de la descente de gradient avec recherche linéaire
def gradMethod_line_search(X,Y,w,w0,rho,epsilon):
    gamma = calcL(X,Y,w,w0,rho,0.5)
    winit = np.append(w,w0)
    lim = np.linalg.norm(gradf1(X,Y,w,w0,rho))
    while(lim>epsilon):
        winit = winit - gamma*(gradf1(X,Y,winit[0:X.shape[1]],winit[X.shape[1]],rho))
        lim = np.linalg.norm(gradf1(X,Y,winit[0:X.shape[1]],winit[X.shape[1]],rho))
        print(lim)
        gamma = calcL(X,Y,winit[0:X.shape[1]],winit[X.shape[1]],rho,gamma)
    return winit

def Xplus(X,Y,w,rho,gamma):
    w = w-gamma*gradf1(X,Y,w[0:X.shape[1]],w[X.shape[1]],rho)
    return w

def calcL(X,Y,w,w0,rho,gamma_1):
    w2 = np.append(w,w0)
    a = 0.5
    b = 2*gamma_1
    l=1
    C = b*(a**l)
    Lim = func1(X,Y,w,w0,rho) + np.transpose(gradf1(X,Y,w,w0,rho)).dot(Xplus(X,Y,w2,rho,C)-w2)+(1/(2*C))*np.sum((Xplus(X,Y,w2,rho,C)-w2)**2)
    mina = func1(X,Y,Xplus(X,Y,w2,rho,b*(a**l))[0:X.shape[1]],Xplus(X,Y,w2,rho,b*(a**l))[X.shape[1]],rho)
    while(mina>Lim):
        l=l+1
        C = b*(a**l)
        Lim = func1(X,Y,w,w0,rho) + np.transpose(gradf1(X,Y,w,w0,rho)).dot(Xplus(X,Y,w2,rho,C)-w2)+(1/(2*C))*np.sum((Xplus(X,Y,w2,rho,C)-w2)**2)
        mina = func1(X,Y,Xplus(X,Y,w2,rho,b*(a**l))[0:X.shape[1]],Xplus(X,Y,w2,rho,b*(a**l))[X.shape[1]],rho)
    return b*(a**(l-1))

# méthode de newton classique
def Newton(X,Y,w,w0,rho,epsilon):
    norm = []
    winit = np.append(w,w0)
    lim = np.linalg.norm(gradf1(X,Y,w,w0,rho))
    norm.append(lim)
    while(lim > epsilon):
        #A = np.linalg.inv(hessf1(X,Y,winit[0:X.shape[1]],winit[X.shape[1]],rho))
        A = np.linalg.solve(hessf1(X,Y,winit[0:X.shape[1]],winit[X.shape[1]],rho), np.identity(X.shape[1]+1))
        B = np.transpose(gradf1(X,Y,winit[0:X.shape[1]],winit[X.shape[1]],rho))
        winit = winit - np.dot(A,B)
        lim = np.linalg.norm(gradf1(X,Y,winit[0:X.shape[1]],winit[X.shape[1]],rho))
        norm.append(lim)
    return winit

# méthode de Newton avec recherche linéaire utilisant les fonctions de la descente de gradient
def Newton2(X,Y,w,w0,rho,epsilon):
    norm = []
    winit = np.append(w,w0)
    lim = np.linalg.norm(gradf1(X,Y,w,w0,rho))
    norm.append(lim)
    gamma = calcL(X,Y,w,w0,rho,0.5)
    while(lim > epsilon):
        #A = np.linalg.inv(hessf1(X,Y,winit[0:X.shape[1]],winit[X.shape[1]],rho))
        A = np.linalg.solve(hessf1(X,Y,winit[0:X.shape[1]],winit[X.shape[1]],rho), np.identity(X.shape[1]+1))
        B = np.transpose(gradf1(X,Y,winit[0:X.shape[1]],winit[X.shape[1]],rho))
        winit = winit - gamma*np.dot(A,B)
        lim = np.linalg.norm(gradf1(X,Y,winit[0:X.shape[1]],winit[X.shape[1]],rho))
        gamma = calcL(X,Y,winit[0:X.shape[1]],winit[X.shape[1]],rho,gamma)
        norm.append(lim)
    return winit

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
        if self.solver == 'Newton':
            if self.penalty == 'i2':
                self.coef = Newton(X,y,w_init,w_init0,self.C,self.tol)
            else:
                print("Error of penalisation for the solver")
        if self.solver == 'Newton_line':
            if self.penalty == 'i2':
                self.coef = Newton2(X,y,w_init,w_init0,self.C,self.tol)
            else:
                print("Error of penalisation for the solver")
        if self.solver == 'constant_step':
            if self.penalty == 'i2':
                self.coef = gradMethod(X,y,w_init,w_init0,self.C,self.tol)
            else:
                print("Error of penalisation for the solver")
                
        if self.solver == 'line_search':
            if self.penalty == 'i2':
                self.coef = gradMethod_line_search(X,y,w_init,w_init0,self.C,self.tol)
            else:
                print("Error of penalisation for the solver")
        if self.solver == 'prox':
            if self.penalty == 'i1':
                self.coef = proxMethod(X,y,w_init,w_init0,self.C,self.tol)
            else:
                print("Error of penalisation for the solver")
    
    def predict(X):
        y_pred = np.empty((X.shape[0]))
        for i in range(0,X.shape[0]):
            h_X = np.dot(X[i],np.transpose(self.coef[0:X.shape[1]])) + self.coef[X.shape[1]]
            if h_X > 0:
                y_pred[i] = self.label[1]
            else:
                y_pred[i] = self.label[0]
        return y_pred
    
    def score(X,y):
        y_pred = self.predict(X)
        score = 0
        for i in range(0,X.shape[0]):
            if y_pred[i] == y[i]:
                score = score + 1
        score = score / X.shape[0]
        return score