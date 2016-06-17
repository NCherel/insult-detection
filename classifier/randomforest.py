# coding: utf-8

"""
Random Tree Classifier
Authors : Mohamed, Hugo, Nicolas

"""
import numpy as np
from TreeClassifier import TreeClassifier
from TreeClassifier import vote
from sklearn.datasets import load_digits

def average(y):
    """ Returns the majoritarian class of a leaf"""
    n_samples, _ = y.shape
    result = np.zeros(n_samples)

    for i in range(n_samples):
        result[i] = vote(y[i])
    
    return result
    

class RandomForestClassifier():
    """ Random forest implementation using TreeClassifier
    
    The forest has n_estimators TreeClassifier predicting using different
    features.
    For each estimator, we need to have the features_index.
    """
    
    def __init__(self, n_estimators=10, max_features="auto", max_depth=3):
        self.n_estimators = n_estimators
        self.max_features = max_features
        
        self.estimators = []
        
        for i in range(self.n_estimators):
            self.estimators.append(TreeClassifier(max_depth=max_depth))
    
    def fit(self, X, y):
        
        n_samples, n_features = X.shape
        
        # Fit each estimator
        for estimator in self.estimators:
            
            # Select a subset of features
            features = self.select_features(n_features, self.max_features)
            X_estimator = X[:, features]
            
            # Save the features used and fit
            estimator.features = features
            estimator.fit(X_estimator, y)

    
    def predict(self, X):
        """ Predict using binary classification"""
        n_samples, n_features = X.shape
        y_pred = np.zeros((n_samples, self.n_estimators))
        
        for i in range(self.n_estimators):
            estimator = self.estimators[i]
            X_estimator = X[:, estimator.features]
            y_pred[:,i] = estimator.predict(X_estimator)

        # In the end we will average each line to get the result
        print y_pred
        y_pred = average(y_pred)

        return y_pred
    
    
    def select_features(self, n_features, max_features="auto"):
        
        # For max_features auto or sqrt
        if max_features in ["auto", "sqrt"]:
            sqrt_features = int(np.sqrt(n_features))
            features = np.random.choice(n_features, sqrt_features,
                                        replace=False)
        
        # TODO : add different modes to select the features
        return features
        
        
if __name__ == "__main__":
    from numpy.random import shuffle

    data = load_digits()
    
    y = data.target
    X = data.data
    
    n = range(len(X))
    shuffle(n)
    y = y[n]
    X = X[n]
    
    print X.shape
    
    X_train = X[:500]
    y_train = y[:500]
    rf = RandomForestClassifier(n_estimators=20, max_depth=10)
    rf.fit(X_train,y_train)
    
    X_test = X[500:]
    y_test = y[500:]
    y_pred = rf.predict(X_test)
    print 1.0*np.sum(y_test.reshape(-1,1) == y_pred.reshape(-1,1))/len(X_test)