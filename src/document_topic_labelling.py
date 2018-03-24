import logging
import math
import string
import pandas as pd
import numpy as np
from optparse import OptionParser
import sys
from time import time
import matplotlib.pyplot as plt
import pickle

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.extmath import density
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

ps = PorterStemmer()


#'# From Caravan Institute #'.translate(None, string.punctuation)

def stemming(sentence):
    if (type(sentence) is float):
	    sentence = '-'
    print (sentence)
    s = sentence.translate(None, string.punctuation)
    str1 = ''.join([i if ord(i) < 128 else ' ' for i in s]) #selecting only ascii characters
    words = word_tokenize(str1)
    words_stemmed = map(lambda x : ps.stem(x), words)
    return ' '.join([s for s in words_stemmed])

#d = pd.read_csv('/home/shayaan/sem_7/ML/Machine-Learning-Projects/data/datagov.csv')
#d_1 = d[['Name','Description','Publisher','URL','Views']]
d = pd.read_csv('../data/new_data/all_data.csv')
#data_raw1 = d['Name']+d['Description']
data_raw = d['Description']
#data_raw = pd.concat([data_raw1,data_raw2],axis=0)
#data_raw = d['Name']+d['Big Description']+d['Summary']
data = data_raw.apply(stemming)
loaded_vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))
X_test = loaded_vectorizer.transform(data)
le = preprocessing.LabelEncoder()
le_classes = np.load('models/classes.npy')
le.fit(le_classes)
#print le_classes
filename = 'models/finalized_model_LRl2.sav'
loaded_model = pickle.load(open(filename, 'rb'))
pred = loaded_model.predict(X_test)
print (pred) 



