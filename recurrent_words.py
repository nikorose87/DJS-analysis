#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 20:31:25 2018
Natural Lenguage process analysis, using 
@author: eprietop
"""

from nltk import sent_tokenize, word_tokenize, regexp_tokenize, FreqDist
from nltk.corpus import stopwords
from sklearn.datasets import fetch_20newsgroups
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS


def tokenize(text, pat='(?u)\\b\\w\\w+\\b', stop_words='english', min_len=2):
    if stop_words:
        stop = set(stopwords.words(stop_words))
    return [w
            for w in regexp_tokenize(text.casefold(), pat)
            if w not in stop and len(w) >= min_len]

def get_data():
    categories = ['alt.atheism', 'soc.religion.christian',
                  'comp.graphics', 'sci.med']
    twenty_train = \
        fetch_20newsgroups(subset='train',
                           categories=categories, shuffle=True)
    twenty_test = \
        fetch_20newsgroups(subset='test',
                           categories=categories, shuffle=True)
    X_train = pd.DataFrame(twenty_train.data, columns=['text'])
    X_test = pd.DataFrame(twenty_test.data, columns=['text'])
    return X_train, X_test, twenty_train.target, twenty_test.target

X_train, X_test, y_train, y_test = get_data()

words = tokenize(X_train.text.str.cat(sep=' '), min_len=4)

fdist = FreqDist(words)

wc = WordCloud(width=800, height=400, max_words=100).generate_from_frequencies(fdist)

plt.figure(figsize=(12,10))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.savefig('result.png')