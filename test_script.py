#!/usr/bin/python

import cgi   
import pickle
import heapq
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

def super_classes(keyword,targets,le,prob):
	if (keyword == 'Law and Police'):
		crimes = le.transform(['crime'])[0]
		accidents = le.transform(['accidents'])[0]
		traffic_violations = le.transform(['traffic violations'])[0]
		cases = le.transform(['court cases'])[0]
		prob_law = prob[:,accidents] + prob[:,crimes]+prob[:,traffic_violations]+prob[:,cases]
		return prob_law
	elif (keyword == 'Governance'):
		politics = le.transform(['politics'])[0]
		elections = le.transform(['elections'])[0]
		prob_gov = prob[:,politics]+prob[:,elections]
		return prob_gov
	elif (keyword == 'Finance'):
		stocks = le.transform(['stock data'])[0]
		banking = le.transform(['banking'])[0]
		insurance = le.transform(['Insurance'])[0]
		prob_finance = prob[:,stocks]+prob[:,banking]+prob[:,insurance]
		return prob_finance
	elif (keyword == 'Education'):
		students = le.transform(['student performances'])[0]
		univs = le.transform(['universities and colleges'])[0]
		prob_edu = prob[:,students]+prob[:,univs]
		return prob_edu
	elif (keyword == 'Transport'):
		airways = le.transform(['airways'])[0]
		rail = le.transform(['rail and metro'])[0]
		road = le.transform(['roadways'])[0]
		prob_transport = prob[:,airways]+prob[:,rail]+prob[:,road]
		return prob_transport
	elif (keyword == 'Medicine'):
		diagnostics = le.transform(['diagnostics'])[0]
		epidemics = le.transform(['diseases and epidemics'])[0]
		genomics = le.transform(['genomics'])[0]
		hospitals = le.transform(['health infrastructure'])[0]
		fitness = le.transform(['fitness and personal well being'])[0]
		prob_med = prob[:,diagnostics]+prob[:,epidemics]+prob[:,genomics]+prob[:,hospitals]+prob[:,fitness]
		return prob_med
	elif (keyword == 'Environment'):	
		pollution = le.transform(['air pollution'])[0]
		disasters = le.transform(['disasters'])[0]
		weather = le.transform(['weather and climate'])[0]
		prob_env = prob[:,pollution]+prob[:,disasters]+prob[:,weather]
		return prob_env
	elif (keyword == 'Languages'):
		grammar = le.transform(['grammar'])[0]
		text = le.transform(['text analysis'])[0]
		prob_lang = prob[:,grammar] + prob[:,text]
		return prob_lang
	elif (keyword == 'Multimedia'):
		video_games = le.transform(['video games'])[0]
		news = le.transform(['news'])[0]
		music = le.transform(['music'])[0]
		movies = le.transform(['movies'])[0]
		books = le.transform(['books and comics'])[0]
		prob_media = prob[:,video_games] + prob[:,news]+prob[:,music]+prob[:,movies]+prob[:,books]
		return prob_media
	elif (keyword == 'Sports'):
		players = le.transform(['sports teams and players'])[0]
		events = le.transform(['sports events'])[0]
		football = le.transform(['football'])[0]
		prob_sports = prob[:,players]+prob[:,events]+prob[:,football]
		return prob_sports
	elif (keyword == 'Internet'):
		online_info = le.transform(['online information'])[0]
		social_media = le.transform(['social media'])[0]
		#online_forums = le.transform(['Online forums'])[0]
		online_shopping = le.transform(['online shopping'])[0]
		prob_internet = prob[:,online_info]+prob[:,social_media]+prob[:,online_shopping]
		return prob_internet
	else:
		return None


def super_super_classes(keyword,targets,le,prob):
	if (keyword == 'Social Sciences'):
		prob_gov = super_classes('Governance',targets,le,prob)
		prob_law = super_classes('Law and Police', targets, le,prob)
		history = le.transform(['history'])[0]
		demography = le.transform(['demography'])[0]
		geo_info = le.transform(['geographical information'])[0]
		prob_social = prob_gov + prob_law + prob[:,history]+prob[:,demography]+prob[:,geo_info]
		return prob_social
	elif (keyword == 'Economics'):
		prob_finance = super_classes('Finance',targets,le,prob)
		trade = le.transform(['trade and business'])[0]
		prices = le.transform(['commodity prices'])[0]
		real_estate = le.transform(['real estate'])[0]
		prob_eco = prob_finance + prob[:,trade] + prob[:,prices]+prob[:,real_estate]
		return prob_eco
	elif (keyword == 'Technology'):
		prob_multimedia = super_classes('Multimedia',targets,le,prob)
		prob_internet = super_classes('Internet',targets,le,prob)
		gadgets = le.transform(['gadgets'])[0]
		programming = le.transform(['programming'])[0]
		prob_tech = prob_multimedia + prob_internet + prob[:,gadgets]+prob[:,programming]
		return prob_tech
	elif (keyword == 'Biology'):
		prob_med = super_classes('Medicine',targets,le,prob)
		food = le.transform(['food and nutrition'])[0]
		zoology = le.transform(['zoological'])[0]
		prob_bio = prob_med + prob[:,food]+prob[:,zoology]
		return prob_bio
	else:
		return None

def super_super_super_classes(keyword,targets,le,prob):
	if (keyword == 'Science'):
		prob_bio = super_super_classes('Biology',targets,le,prob)
		prob_env = super_classes('Environment',targets,le,prob)
		astronomy = le.transform(['astronomy'])[0]
		chem = le.transform(['chemistry'])[0]
		prob_sci = prob_bio+prob_env+prob[:,astronomy]+prob[:,chem]
		return prob_sci
	else:
		return None


def Predict(keyword):
	d = pd.read_csv('/home/indu/sem7/ML1/temp/Machine-Learning-Projects/data/datagov.csv')
	#d = pd.read_csv('/home/indu/sem7/ML1/temp/Machine-Learning-Projects/data/kaggle-data-set.csv')
	data_raw = d['Name']+d['Description']
	#data_raw = d['Name']+d['Big Description']+d['Summary']
	data = data_raw.apply(stemming)
	loaded_vectorizer = pickle.load(open('/home/indu/sem7/ML1/temp/Machine-Learning-Projects/src/vectorizer.pkl', 'rb'))
	X_test = loaded_vectorizer.transform(data)
	le = preprocessing.LabelEncoder()
	le_classes = np.load('/home/indu/sem7/ML1/temp/Machine-Learning-Projects/src/classes.npy')
	le.fit(le_classes)
	filename = '/home/indu/sem7/ML1/temp/Machine-Learning-Projects/src/finalized_model_LRl2.sav'
	loaded_model = pickle.load(open(filename, 'rb'))
	pred = loaded_model.predict(X_test)
	prob = loaded_model.predict_proba(X_test)
	#print (keyword in ['Law and Police', 'Governance', 'Finance','Education','Transport'])
	if (keyword in ['Law and Police', 'Governance', 'Finance','Education','Transport','Medicine','Environment','Languages','Multimedia','Sports','Internet']):
		print keyword
		prob_curr_label = super_classes(keyword, le_classes,le,prob)
	elif (keyword in ['Social Sciences','Economics','Technology','Biology']):
		prob_curr_label = super_super_classes(keyword, le_classes,le,prob)
	elif (keyword in ['Science']):
		prob_curr_label = super_super_super_classes(keyword,le_classes,le,prob)
	else:
		numerical_label = le.transform([keyword])[0]
		prob_curr_label = prob[:,numerical_label]
	max_probs = heapq.nlargest(100,prob_curr_label)
	thresh = np.mean(max_probs)
	se = pd.Series(pred)
	d['Predicted'] = se.values
	prob_series = pd.Series(prob_curr_label)
	d['Probability'] = prob_series.values
	#return 0
	#all_results =  d[d['Predicted'] == le.transform([keyword])[0]]
	#greater_than_thresh = all_results[all_results['Probability']>thresh]
	greater_than_thresh = d[d['Probability']>thresh]
	return greater_than_thresh

x = Predict('crime')