import pickle
import sqlalchemy
import heapq
import pandas as pd
import string
import sys
import numpy as np
import codecs

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

from fuzzywuzzy import process

#!/usr/bin/env python

import os, lucene

from java.nio.file import Paths
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.queryparser.classic import MultiFieldQueryParser
from org.apache.lucene.store import SimpleFSDirectory
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.queryparser.complexPhrase import ComplexPhraseQueryParser
from org.apache.lucene.queryparser.ext import ExtendableQueryParser
from org.apache.lucene.search import BooleanClause 
from org.apache.lucene.queries import CustomScoreQuery
from org.apache.lucene.queries import CustomScoreProvider
from org.apache.lucene.util import Version

# import sys
# reload(sys)
# sys.setdefaultencoding('utf-8')

from flask import Flask, render_template, request
from flaskext.mysql import MySQL
app = Flask(__name__)
mysql = MySQL()
mysql.init_app(app)

database_username = 'root'
database_password = 'Shayaan7'  #Please replace **** with your password
database_ip       = '127.0.0.1'
database_name     = 'huduku'

app.config['MYSQL_DATABASE_USER'] = 'root'
app.config['MYSQL_DATABASE_PASSWORD'] = 'Shayaan7'
app.config['MYSQL_DATABASE_DB'] = 'huduku'
app.config['MYSQL_DATABASE_HOST'] = 'localhost'

database_connection = sqlalchemy.create_engine('mysql+mysqlconnector://{0}:{1}@{2}/{3}'. format(database_username, database_password, database_ip, database_name), echo=False, encoding="utf8")


INDEX_DIR = "IndexFiles.index"


ps = PorterStemmer()
loaded_vectorizer = pickle.load(open('../models/vectorizer.pkl', 'rb'))
le = preprocessing.LabelEncoder()
le_classes = np.load('../models/classes.npy')
le.fit(le_classes)
filename = '../models/finalized_model_LRl2.sav'
loaded_model = pickle.load(open(filename, 'rb'))

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
	#classes_list = ['Science']
	#new_key = process.extractOne(keyword, classes_list)
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
	d = pd.read_csv('../../data/new_data/check_data.csv')
	d.fillna('',inplace=True)
	d = d.drop(d.columns[d.columns.str.contains('unnamed',case = False)],axis = 1)

	data_raw = d['Title']+d['Description']+d['Summary']

	data = data_raw.apply(stemming)
	X_test = loaded_vectorizer.transform(data)
	
	pred = loaded_model.predict(X_test)
	prob = loaded_model.predict_proba(X_test)
	keys_top_level = ['Science'] 
	keys_first_level = ['Social Sciences', 'Economics' ,'Technology', 'Biology']
	keys_second_level = ['Law and Police', 'Governance', 'Finance', 'Education', 'Transport', 'Medicine', 'Environment', 'Languages', 'Multimedia', 'Sports', 'Internet']
	all_keys = keys_top_level + keys_first_level + keys_second_level + list(le_classes)
	key_final = process.extractOne(keyword, all_keys)[0]
	#print key_final 
	if (key_final in keys_second_level):
		print keyword
		prob_curr_label = super_classes(key_final, le_classes,le,prob)
	elif (key_final in keys_first_level):
		prob_curr_label = super_super_classes(key_final, le_classes,le,prob)
	elif (key_final in keys_top_level):
		prob_curr_label = super_super_super_classes(key_final,le_classes,le,prob)
	elif (key_final in list(le_classes)):
		numerical_label = le.transform([key_final])[0]
		prob_curr_label = prob[:,numerical_label]
	else:
		return None
	max_probs = heapq.nlargest(100,prob_curr_label)
	thresh = np.mean(max_probs)
	se = pd.Series(pred)
	d['Predicted'] = se.values
	prob_series = pd.Series(prob_curr_label)
	d['Probability'] = prob_series.values
	greater_than_thresh = d[d['Probability']>thresh]
	return greater_than_thresh.drop(['Predicted','Probability'],axis=1)

def search(command, searcher, analyzer):
#	while True:
#		print
#		print "Hit enter with no input to quit."
#		command = raw_input("Query:")
#		if command == '':
#			return
#
#		print
#		print "Searching for:", command
		d = pd.read_csv('../../data/new_data/check_data.csv')
		d.fillna('',inplace=True)
		d = d.drop(d.columns[d.columns.str.contains('unnamed',case = False)],axis = 1)
		fields = ("description", "title", "summary", "keywords")
		parser = MultiFieldQueryParser(fields, analyzer)
		query = MultiFieldQueryParser.parse(parser, command)
		#query = QueryParser("description", analyzer).parse(command)
		#query = parser.parse(command)
		scoreDocs = searcher.search(query, 50).scoreDocs
		#print "%s total matching documents." % len(scoreDocs)
		topics = []
		for scoreDoc in scoreDocs:
			doc = searcher.doc(scoreDoc.doc)
			topics.append(doc.get("title"))
			#print 'Topic:', doc.get("title"), 'Score:', scoreDoc.score
		df = d.loc[d['Title'].isin(topics)]
		return df	


def algoSearch(command, searcher, analyzer):
#	while True:
#		print
#		print "Hit enter with no input to quit."
#		command = raw_input("Query:")
#		if command == '':
#			return
#
#		print
#		print "Searching for:", command
		d = pd.read_csv('../../data/new_data/check_data.csv')
		d.fillna('',inplace=True)
		d = d.drop(d.columns[d.columns.str.contains('unnamed',case = False)],axis = 1)
		#fields = ("algorithm")
		#parser = MultiFieldQueryParser(fields, analyzer)
		#query = MultiFieldQueryParser.parse(parser, command)
		query = QueryParser("algorithm", analyzer).parse(command)
		query = parser.parse(command)
		scoreDocs = searcher.search(query, 50).scoreDocs
		#print "%s total matching documents." % len(scoreDocs)
		topics = []
		for scoreDoc in scoreDocs:
			doc = searcher.doc(scoreDoc.doc)
			topics.append(doc.get("title"))
			#print 'Topic:', doc.get("title"), 'Score:', scoreDoc.score
		df = d.loc[d['Title'].isin(topics)]
		return df	

@app.route('/')
def student():
   return render_template('index.html')

@app.route('/result',methods = ['POST', 'GET'])
def result():
	if request.method == 'POST':
		result = request.form
		keyword = result['Name']
		criteria = request.form['criteria']
		print criteria
		if criteria == 'domain':
			df = Predict(keyword)
		elif criteria == 'keyword':
			vm_env = lucene.getVMEnv()
			vm_env.attachCurrentThread()
			base_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
			directory = SimpleFSDirectory(Paths.get(os.path.join(base_dir, INDEX_DIR)))
			searcher = IndexSearcher(DirectoryReader.open(directory))
			analyzer = StandardAnalyzer()
		  	df = search(keyword, searcher, analyzer)
		  	del searcher
		# df.to_html('templates/test.html' , na_rep='NaN', decimal='.')
		df.to_sql(con=database_connection, name='result', if_exists='replace')
		# df['Author'] =df['Author'].apply(stemming)
		conn = mysql.connect()
		cursor =conn.cursor()
		cursor.execute("SELECT * from result;")
		data = cursor.fetchall()
		# print df.iloc[0]
		# print data[0]
		return render_template("test.html",result=data)

if __name__ == '__main__':
	lucene.initVM(vmargs=['-Djava.awt.headless=true'])
	app.run(threaded=True,debug = True,host= '0.0.0.0')
