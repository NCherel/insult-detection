# coding: utf-8

"""
Decision Tree Classifier
Authors : Mohamed, Hugo, Nicolas

"""
from __future__ import division
import numpy as np
from sklearn.datasets import load_digits

def impurity(y, criterion="gini"):
    """ Compute the impurity of a given set

    Parameters
    ----------

    y : np.array(n_samples, _)

    function : "gini" | "entropy"

    Returns
    -------

    entropy : float
    
    """
    n_samples = y.shape[0]
    probability_array = []

    classes = np.unique(y)

    # For each class, count the number of elements
    for classe in classes:
        total = np.sum(y == classe)
        probability_array.append(total)

    probability_array = np.array(probability_array, dtype=float)
    probability_array /= n_samples

    # Compute the impurity given the formula
    # impurity = 1 - sum pj^2
    if criterion == "gini":
        D = 1 - np.sum(probability_array**2)

    # Compute the impurity given the formula :
    # entropy = - sum (pj log_2(pj))
    elif criterion == "entropy":
        D = - np.sum(probability_array * np.log2(probability_array))

    return D

def loss_function(y, idxg, idxd, criterion="gini"):
    """ Loss function for the tree and a given split"""
    yg = y[idxg]
    yd = y[idxd]
    n = y.shape[0]
    nd = yd.shape[0]
    ng = yg.shape[0]
    loss = (1.0*nd)/n * impurity(yd, criterion) + (1.0*ng)/n * impurity(yg, criterion)
    return loss

def split_loss(X, y, feature_index, threshold):
    lidx, ridx = split(X, feature_index, threshold)
    return loss_function(y, lidx, ridx)

def split(X, feature_index, threshold):
    """ Split the data in 2 parts.

    Parameters
    ----------
    X : np.array(n_samples, n_features)
    
    feature : integer (should be between 0 and n_features - 1)
    
    threshold : float

    Returns
    -------
    left_array : np.array(_, n_features)
    
    right_array : np.array(_, n_features)
    """

    return X[:,feature_index] <= threshold, X[:,feature_index] > threshold
    
def vote(y):
    """ Returns the majoritarian class of a leaf"""
    classes = np.unique(y)
    counts = []
    
    for classe in classes:
        count = np.sum(y == classe)
        counts.append(count)
    
    counts = np.array(counts)
    return classes[np.argmax(counts)]
    
def dichotomie(id_features,X,y,epsilon):
    b = np.max(X[::,id_features])
    a = np.min(X[::,id_features])
    tho = (b+a)/2

    l_a = split_loss(X,y,id_features,a)
    l_b = split_loss(X,y,id_features,b)
    
    while b-a>epsilon :
        if l_b>l_a:
            b = tho
            tho = (b+a)/2
            l_b = split_loss(X,y,id_features,b)
        else:
            a = tho
            tho = (b+a)/2
            l_a = split_loss(X,y,id_features,a)

    lidx, ridx = split(X, id_features, tho)
    
    if np.sum(lidx)  == 0 or np.sum(ridx) == 0:
        return 0, np.inf
        
    return tho, split_loss(X, y, id_features, tho)
    
def best_features(X, y, epsilon):
    A = np.empty((X.shape[1],2))
    for i in range(0,X.shape[1]):
        A[i] = dichotomie(i,X,y,epsilon)
    
    #print A
    idx = np.argmin(A[:,1])
    lidx, ridx = split(X, idx, A[idx][0])
    
    while np.sum(lidx) == 0 or np.sum(ridx) == 0:
        A = np.delete(A, idx, 0)
        
        if len(A) == 0:
            return -1, np.inf
        
        idx = np.argmin(A[:,1])
        lidx, ridx = split(X, idx, A[idx][0])
        
    return idx, A[idx][0]

class Node():

    def __init__(self, depth, X, y):
        self.feature = None
        self.threshold = 0.
        self.leaf = False
        self.leftNode = None
        self.rightNode = None

        n_samples, n_features = X.shape
    
        # Check if the node must be a leaf
        if depth >= TreeClassifier.max_depth or n_samples <= TreeClassifier.min_samples_leaf:
            self.leaf = True
            self.value = vote(y)
            return

        # If the node is not a leaf
        # Compute the best feature and the best threshold
        self.feature, self.threshold = best_features(X, y, 0.1)
        
        # Can't cut right, this must be a leaf
        if self.feature == -1:
            self.leaf = True
            self.value = vote(y)
            return
        
        lidx, ridx = split(X, self.feature, self.threshold)
        
        # print "Feature number : ", self.feature
        # print "Feature threshold : ", self.threshold
        
    
        self.leftNode = Node(depth + 1, X[lidx], y[lidx])
        self.rightNode = Node(depth + 1, X[ridx], y[ridx])

    def predict(self, X):
        # If the node is a leaf, it has a value
        if self.leaf:
            return self.value

        # Else we must propagate
        if X[self.feature] <= self.threshold:
            return self.leftNode.predict(X)

        else:
            return self.rightNode.predict(X)
            
            
class TreeClassifier():
    
    max_depth = 10
    min_samples_leaf = 3

    
    def __init__(self, criterion='gini', max_depth=None, min_samples_split=2,
                     min_samples_leaf=1, random_state=None):
        self.criterion = criterion 
        self.depth = max_depth
        self.min_split = min_samples_split
        self.min_leaf = min_samples_leaf
        self.state = random_state
        self.root = None

    def fit(self, X, y):
        self.root = Node(0, X, y)

    def predict(self, X):
        n_samples, n_features = X.shape
        y = np.zeros(n_samples)
            
        for i in range(n_samples):
            y[i] = self.root.predict(X[i])

        return y

            
if __name__ == "__main__":
    from numpy.random import shuffle
    y = np.array([1,1,1,1,2,2,2,3,3,3])
    # entropy should be equal to 1.5709
#    print impurity(y, criterion="entropy")

    a = np.array([1,1,1,3,3,3,3])
 #   print vote(a)

    data = load_digits()
    
    y = data.target
    X = data.data
    
    n = range(len(X))
    shuffle(n)
    y = y[n]
    X = X[n]
    
  #  print X.shape
    
    X_train = X[:500]
    y_train = y[:500]
    t = TreeClassifier()
    t.fit(X_train,y_train)
    
    X_test = X[500:]
    y_test = y[500:]
    y_pred = t.predict(X_test)
   # print 1.0*np.sum(y_test.reshape(-1,1) == y_pred.reshape(-1,1))/len(X_test)
