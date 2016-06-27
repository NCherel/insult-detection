# coding: utf-8

"""
The tokenizer is responsible for transforming the text into an array of string

Authors : Mohamed, Hugo, Nicolas

"""
import numpy as np
from scipy import sparse


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
        idf = sparse.csr_matrix(self.idf(X))
        result = X.tocsr().multiply(idf)
        return result

    def idf(self, X):
        n_samples, n_terms = X.shape
        # WARNING : non sparse operation
        n_t = np.diff(X.tocsc().indptr).astype(float)
        return np.log(n_samples/n_t)
