# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 16:06:54 2016

@author: mohamedmf
"""

import re
from collections import namedtuple
import numpy as np

class Tokenizer:

  Token = namedtuple('Token', 'name text span')

  def __init__(self, tokens):
    self.tokens = tokens
    ## regular expression from patterns given by tokens : '|' = OU logique

  def iter_tokens(self, input, ignore_ws=True):
    for match in self.regex.finditer(input):
      if ignore_ws and match.lastgroup == 'WHITESPACE':
        continue
      yield Tokenizer.Token(match.lastgroup, match.group(0), match.span(0))
      
  def tokenize_momo(self, input, ignore_ws=True):
    tokenized_text = []
    for token in self.iter_tokens(input, ignore_ws):
        tokenized_text.append(token.text)
    return tokenized_text

  def tokenize(self, input, ignore_ws=True):
    tokenized_text = input
    
    for token in self.tokens:
        tokenized_text = re.sub(token[1], token[2], tokenized_text)
    
    return tokenized_text.split(" ")

  def tokenize_array(self, input, ignore_ws=True):
     tokenized_text = []
     for text in input:
         tokenized_text.append(self.tokenize(text))

     return tokenized_text

  def create_dict(self, tokens):
      dictionary = set()

      for words in tokens:
          for word in words:
              dictionary.add(word)

      return dictionary

  def count_occurences(self, array, dictionary):
      n_samples = len(array)
      n_words = len(dictionary)
      result = np.zeros((n_samples, n_words))
      
      for i, words in enumerate(array):
          for word in words:
              idx = dictionary.index(word)
              result[i][idx] += 1
              
      return result
      
      
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
  print(myTokenizer.count_occurences(tokens, dictionary))
  #print(dictionary)