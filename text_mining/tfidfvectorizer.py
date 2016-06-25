# coding: utf-8

"""
The tokenizer is responsible for transforming the text into an array of string

Authors : Mohamed, Hugo, Nicolas

"""
import numpy as np
import scipy.sparse

class TfidfVectorizer():
    """ Transform a count matrice to a tf-idf matrice

    TODO : add different weighting scheme (binary, log)
    """
    
    def __init__(self):
        """ Constructor, sets the parameters """
        pass

    def transform(self, X):
        """ Return the tf-idf matrice taking count sparse matrice as X
        """
        result = np.ndarray(X.shape)
        X = X.todense()
        n_samples = X.shape[0]
        idf = self.idf(X)
        result = np.multiply(X, idf)
     
        return result
    
    def idf(self, X):
        n_samples, n_terms = X.shape
        n_t = np.array(n_terms)
        
        # No to sparse matrix        
        n_t = np.sum(X > 0, axis=0)
        return np.log(n_samples/n_t)
