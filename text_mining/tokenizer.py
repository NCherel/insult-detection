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

# To Keep updated
TOKENS = [
    ### Replacements first ###
    ('BREAK'        , r"\\n", r" "),
    ('SLASH'           , r"\/+", r" "),
    ('WHITE_SPACE'     , r"(\\\\xc2|\\\\xa0|\\xa0|\\xa1|\\xa3|\\xa9|\\r|\\ufeff|\\u2013|\\u2016|\\u200f|\\u2665|\\U0001f308|\\U0001f3e9|\\U0001f48b|\\\\|\\|&nbsp|&amp|=|\"\"|\"|\\u2018|\\u2019)", r" "),
    ('LETTER_A'        , r"(\\xe1|\\xe3|\\xe0|\\xc2|\\u1ef1|\\u0105|\\u1eb7|\\xe5|\\xe4)", r"a"),
    ('LETTER_E'        , r"(\\xe9|\\xe8|\\u1ec3|\\u1ec5|\\u1ebf|\\u0119)", r"e"),
    ('LETTER_I'        , r"(\\xed)", r"i"),
    ('LETTER_O'        , r"(\\x00|\\xf8|\\xf6|\\xf3)", r"o"),
    ('LETTER_U'        , r"(\\xfa|\\xdc|\\xfc|\\u1ee9|\\u0169)", r"u"),
    ('LETTER_Y'        , r"(\\xfd)", r"u"),
    ('LETTER_C'        , r"(\\xe7|\\u0107)", r"c"),
    ('LETTER_Z'        , r"(\\u017a|\\u017e|\\u017c)", r"z"),
    ('PONCT_!'        , r"(\\u203d)", r"!"),
    ### Tokens ###
    ('NOT'             , r"n\'t", r" not "),
    ('PUNCT'      , r"(!+|\?+|\.+)", r" \1 "),
    ('NIL'          , r"\'|,", r" "),
    ('SMILEY'       , r"(\:\w+\:|\<[\/\\]?3|[\(\)\\\D|\*\$][\-\^]?[\:\;\=]|[\:\;\=B8][\-\^]?[3DOPp\@\$\*\\\)\(\/\|])(?=\s|[\!\.\?]|$)|xD|XD", r" smiley "),
    ('PUNCT'      , r"(;+|\,+|\)+|\(+)|\-|\_", r" "),
    ('ALONE'        , r"\s[a-hj-zA-HJ-Z]\s", r" "), # delete all lone letters except I&i
    ('DOTS'         , r"\.+", r" "),
#    ('QUOTE'      , r"(\"|\')+(\w+)(\"|\')+", "QUOTE"), # A améliorer pour mettre la citation entre deux <QUOTE>
    ('BEGIN_WHITESPACE' , r"^\s+", r""),
    ('END_WHITESPACE' , r"\s+$", r""),
    ('WHITESPACES' , r"\s+", r" ")
]

class Tokenizer:

  Token = namedtuple('Token', 'name text span')

  def __init__(self, tokens=TOKENS):
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
  print("### Tokenize String ###")
  #print(myTokenizer.tokenize_array(t))
  tokens1 = myTokenizer.tokenize('Bonjour!!??, je m\'appelle Mohamed !')
  print(tokens1)
  print(myTokenizer.reconstruct(tokens1))

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
