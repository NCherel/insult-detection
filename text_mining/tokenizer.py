# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 16:06:54 2016

@author: mohamedmf
"""

import re
from collections import namedtuple

class Tokenizer:

  Token = namedtuple('Token', 'name text span')

  def __init__(self, tokens):
    self.tokens = tokens
    pat_list = []
    for tok, pat in self.tokens:
      pat_list.append('(?P<%s>%s)' % (tok, pat))
    self.regex = re.compile('|'.join(pat_list)) 
    ## regular expression from patterns given by tokens : '|' = OU logique

  def iter_tokens(self, input, ignore_ws=True):
    for match in self.regex.finditer(input): # Problem here
      if ignore_ws and match.lastgroup == 'WHITESPACE':
        continue
      yield Tokenizer.Token(match.lastgroup, match.group(0), match.span(0))

  def tokenize(self, input, ignore_ws=True):
    tokenized_text = []
    for token in self.iter_tokens(input, ignore_ws):
        tokenized_text.append(token.text)
    return tokenized_text

# test program
if __name__ == "__main__":

# A mettre à jour en fonction des token que l'on risque de trouver
  TOKENS = [
    ('NIL'        , r"nil|\'()"),
    ('TRUE'       , r'true|#t'),
    ('FALSE'      , r'false|#f'),
    ('NUMBER'     , r'\d+'),
    ('STRING'     , r'"(\\.|[^"])*"'),
    ('SYMBOL'     , r'[\x21-\x26\x2a-\x7e]+'),
    ('QUOTE'      , r"'"),
    ('LPAREN'     , r'\('),
    ('RPAREN'     , r'\)'),
    ('DOT'        , r'\.'),
    ('WHITESPACE' , r'\w+'),
  ]

  myTokenizer = Tokenizer(TOKENS)
  print(myTokenizer.tokenize('Bonjour!!, je m\'appelle Mohamed !'))
  print(myTokenizer.tokenize('Ceci est un test du Tokeni zer :) : )'))
  