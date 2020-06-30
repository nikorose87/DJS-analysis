#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 20:12:45 2018
Trying to find the most common words in the papers found with appendices.
first, let's try with this code. tken from 
Code taken from 
@author: eprietop
"""

import collections
import pandas as pd
import matplotlib.pyplot as plt
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
#%matplotlib inline

# Read input file, note the encoding is specified here 
# It may be different in your text file
file = open(dir_path+'/Papers_txt/Schwartz2008.txt', encoding="utf8")
a= file.read()

# Stopwords
stopwords = set(line.strip() for line in open(dir_path+'/Papers_txt/stopwords.txt'))
stopwords = stopwords.union(set(['mr','mrs','one','two','said']))

# Instantiate a dictionary, and for every word in the file, 
# Add to the dictionary if it doesn't exist. If it does, increase the count.
wordcount = {}

# To eliminate duplicates, remember to split by punctuation, and use case demiliters.
for word in a.lower().split():
    word = word.replace(".","")
    word = word.replace(",","")
    word = word.replace(":","")
    word = word.replace("\"","")
    word = word.replace("!","")
    word = word.replace("â€œ","")
    word = word.replace("â€˜","")
    word = word.replace("*","")
    if word not in stopwords:
        if word not in wordcount:
            wordcount[word] = 1
        else:
            wordcount[word] += 1

# Print most common word
n_print = int(input("How many most common words to print: "))
print("\nOK. The {} most common words are as follows\n".format(n_print))
word_counter = collections.Counter(wordcount)
for word, count in word_counter.most_common(n_print):
    print(word, ": ", count)

# Close the file
file.close()

# Create a data frame of the most common words 
# Draw a bar chart
lst = word_counter.most_common(n_print)
df = pd.DataFrame(lst, columns = ['Word', 'Count'])
df.plot.bar(x='Word',y='Count')