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
        idf = np.squeeze(self.idf(X))
        print(idf.shape)
        idf = np.squeeze(idf, axis=0)
        print(idf.shape)
        for j in range(idf.shape[1]):
            result[:,j] = X[:,j] * idf[j]
     
        return result
    
    def idf(self, X):
        n_samples, n_terms = X.shape
        n_t = np.array(n_terms)
        
        # No to sparse matrix        
        n_t = np.sum(X > 0, axis=0)
        
        return np.log(n_samples/n_t)
