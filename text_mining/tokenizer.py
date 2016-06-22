# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 16:06:54 2016

@author: mohamedmf
"""

import re
from collections import namedtuple
import numpy as np
### Stemming with NLTK ###
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import EnglishStemmer
from nltk.corpus import stopwords

# Tokens Globaux
TOKENS = [
    ('PUNCT'      , r"(!+|\?+|\.+)", r" \1 "),
    ('NIL'          , r"\'|,", r" "),
    ('SMILEY'       , r"(\:\w+\:|\<[\/\\]?3|[\(\)\\\D|\*\$][\-\^]?[\:\;\=]|[\:\;\=B8][\-\^]?[3DOPp\@\$\*\\\)\(\/\|])(?=\s|[\!\.\?]|$)|xD|XD", r" \1 "),
    ('PUNCT'      , r"(;+|\,+|\)+|\(+)", r" \1 "),
    ('BREAK'        , r"\\n", r" \1 "),
#    ('NOISE'        , r"("\\\\xc2|\\\\xa0|\\xa0|\\xa1|\\xa3|\\xa9"), r" \1 "),
    ('QUOTE'      , r"(\"|\')+(\w+)(\"|\')+", "QUOTE"), # A améliorer pour mettre la citation entre deux <QUOTE>
    ('BEGIN_WHITESPACE' , r"^\s+", r""),
    ('END_WHITESPACE' , r"\s+$", r""),
    ('WHITESPACES' , r"\s+", r" "),
  ]

class Tokenizer:

  Token = namedtuple('Token', 'name text span')

  def __init__(self, tokens):
    self.tokens = tokens
    self.stemmer = EnglishStemmer(ignore_stopwords=True)
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
     
##### STEMMING WITH NLTK #####
  def stem(self, tokens):
      stemed = []
      for token in tokens:
          stemed.append(self.stemmer.stem(token))
             
      return stemed
      
  def stem_array(self, array):
      stemed = []
      for tokens in array:
          stemed.append(self.stem(tokens))
    
      return stemed
      
  def reconstruct(self, tokens):
      return " ".join(self.stem(tokens))
          
#### CREATE DICTIONARY #####
      
# Use when all the strings are concatenated
  def create_dict(self, tokens):
      #dictionary = set()
      dictionary = []
      for array_token in tokens:
          for word in array_token:
              if word not in dictionary:
                  dictionary.append(word)

      return dictionary

  def count_occurences(self, array, dictionary):
      n_samples = len(array)
      n_words = len(dictionary)
      ### TODO : Use saprse matrix ###
      result = np.zeros((n_samples, n_words))
      
      for i, words in enumerate(array):
          for word in words:
              idx = dictionary.index(word)
              result[i][idx] += 1
              
      return result
      
      
# test program
if __name__ == "__main__":

# A mettre à jour en fonction des token que l'on risque de trouver
  myTokenizer = Tokenizer(TOKENS)
  #print(myTokenizer.tokenize('Bonjour!!??, je m\'appelle Mohamed !'))
  #print(myTokenizer.tokenize('Ceci est un test du Tokeni zer :) : )'))
  
  s1 = 'cook cooker cooked cooking well-cooked cock'
  s2 = "I played a really good game but i lost all my matches"
  s3 = " you are a shitty piece of shit !!!!! BITCH BITCHES CUNT CUNTS"
  #s2 = 'Hello World!!'
  t = [s1, s2, s3]
  #print("### Tokenize Array ###")
  #print(myTokenizer.tokenize_array(t))

  #print(myTokenizer.create_dict(myTokenizer.tokenize_array(t)))
  #print(re.sub(r"(!+)" , r" \1 ", s1))
  
  # First compute tokens of all strings we have
  tokens = myTokenizer.tokenize_array(t) # tokenize array problem
  print(tokens)
  # Then stem all the tokens and creates a dictionary
  print(myTokenizer.stem_array(tokens))
  print("\n ### Dictionnaire ###")
  print(" ### DICO 2 ###")
  dict2 = myTokenizer.create_dict(myTokenizer.stem_array(tokens))
  print(len(dict2))
  print(dict2)
  
  # Computes the count matrix
  print("\n \n")
  print(t)
  print(myTokenizer.tokenize_array(t))
  print(myTokenizer.stem_array(myTokenizer.tokenize_array(t)))
  #myTokenizer.count_occurences(myTokenizer.stem(t), dict2)
  
  # STOP WORDS
  #print(stopwords.words("english"))
  
  print('\n ##### Print occurences #####')
  print(myTokenizer.count_occurences(myTokenizer.stem_array(myTokenizer.tokenize_array(t)), dict2))
  #print(dictionary)