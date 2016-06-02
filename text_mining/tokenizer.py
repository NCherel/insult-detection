# coding: utf-8

"""
The tokenizer is responsible for transforming the text into an array of string

Authors : Mohamed, Hugo, Nicolas

"""


def tokenize(document, removePunctuation=False):
    """ Tokenize an array of text using space delimitors

    The function takes an numpy array of text and returns
    a numpy array of numpy arrays
    
    Note : maybe it is easier to return a sparse matrice with the number of
    occurences of every word for every text
    Numpy doesn't really support multi-dimensional arrays of arbitrary length
    See : http://stackoverflow.com/questions/3386259/how-to-make-a-multidimension-numpy-array-with-a-varying-row-size
    
    Different parameters can be set :

    removePunctuation=False
    """

    # Just take the last example
    # http://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html

    pass
