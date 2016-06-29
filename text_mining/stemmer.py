# coding: utf-8

"""
The stemmer is used to reduce the number of words

Authors : Mohamed, Hugo, Nicolas

"""
from tokenizer import Tokenizer
import re
import numpy as np

def stem(X, dictionary):
    """ Stem X according to the dictionary provided
    
    Parameters
    ----------
    
    X: array, shape(n_samples, n_words)
    
    dictionary : array, shape(n_words,)
    
    Returns
    -------

    X_reduced : array, shape(n_samples n_reduced_words)
    
    """
    
    
# test program
if __name__ == "__main__":

# A mettre à jour en fonction des token que l'on risque de trouver
  TOKENS = [
    ('PUNCT_!'      , r"(!+|\?+|\.+)", r" \1 "),
    ('NIL'          , r"\'|,", r" "),
#    ('TRUE'       , r'true|#t'),
#    ('FALSE'      , r'false|#f'),
#    ('NUMBER'     , r'\d+'),
#    ('STRING'     , r'"(\\.|[^"])*"'),
#    ('SYMBOL'     , r'[\x21-\x26\x2a-\x7e]+'),
#    ('QUOTE'      , r"'"),
#    ('LPAREN'     , r'\('),
#    ('RPAREN'     , r'\)'),
#    ('DOT'        , r'\.'),
    ('BEGIN_WHITESPACE' , r"^\s+", r""),
    ('END_WHITESPACE' , r"\s+$", r""),
    ('WHITESPACE' , r"\s+", r" "),
  ]

  myTokenizer = Tokenizer(TOKENS)
  print(myTokenizer.tokenize('Bonjour!!??, je m\'appelle Mohamed !'))
  #print(myTokenizer.tokenize('Ceci est un test du Tokeni zer :) : )'))
  
  s1 = 'Bonjour!!!!??, je m\'appelle Mohamed Mohamed ... !!'
  s2 = 'Ceci est un test du Tokeni zer :) : )'
  t = [s1, s2]
  #print(myTokenizer.tokenize_array(t))

  print(re.sub(r"(!+)" , r" \1 ", s1))
  
  tokens = myTokenizer.tokenize_array(t)
  dictionary = myTokenizer.create_dict(tokens)
  dictionary = list(dictionary)
  print(dictionary)
  
  X = myTokenizer.count_occurences(tokens, dictionary)
  
  