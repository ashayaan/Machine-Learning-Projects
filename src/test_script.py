#!/usr/bin/python

import cgi   
import pickle
import pandas as pd
import string
import sys
import numpy as np

from optparse import OptionParser
from time import time
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize 
from nltk.stem import PorterStemmer

#############################################################################
#Stemming

ps = PorterStemmer()

def stemming(sentence):
    s = sentence.translate(None, string.punctuation)
    str1 = ''.join([i if ord(i) < 128 else ' ' for i in s]) #selecting only ascii characters
    words = word_tokenize(str1)
    words_stemmed = map(lambda x : ps.stem(x), words)
    return ' '.join([s for s in words_stemmed])

def Predict(keyword):
	d = pd.read_csv('/home/sharvani/ML1/Machine-Learning-Projects/data/datagov.csv')
	data_raw = d['Name']+d['Description']
	data = data_raw.apply(stemming)
	loaded_vectorizer = pickle.load(open('/home/sharvani/ML1/Machine-Learning-Projects/src/vectorizer.pkl', 'rb'))
	X_test = loaded_vectorizer.transform(data)
	le = preprocessing.LabelEncoder()
	le_classes = np.load('/home/sharvani/ML1/Machine-Learning-Projects/src/classes.npy')
	le.fit(le_classes)
	filename = '/home/sharvani/ML1/Machine-Learning-Projects/src/finalized_model_SGD.sav'
	loaded_model = pickle.load(open(filename, 'rb'))
	pred = loaded_model.predict(X_test)
	se = pd.Series(pred)
	d['Predicted'] = se.values
	
	print d[d['Predicted'] == le.transform([keyword])[0]]
	


form = cgi.FieldStorage()   
input_text = form.getfirst("textinput", "0")
Predict('crime')
print "Content-type:text/html\r\n\r\n"
print "<html>"
print "<body>"
print "<p>%s</p>" % input_text
print "</body>"
print "</html>"
