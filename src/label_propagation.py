import pandas as pd
import numpy as np
import string 

from sklearn.preprocessing import LabelEncoder
from sklearn.semi_supervised import LabelPropagation
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
	
ps = PorterStemmer()
def stemming(sentence):
	s = sentence.translate(None, string.punctuation)
	str1 = ''.join([i if ord(i) < 128 else ' ' for i in s])
	words = word_tokenize(str1)
	words_stemmed = map(lambda x : ps.stem(x), words)
	return ' '.join([s for s in words_stemmed])
	
if __name__ == "__main__":
	data = pd.read_csv('kaggle-data-set.csv')
	labels = data['Label']
	le = LabelEncoder()
	le.fit(labels.dropna().unique())
	print(labels.dropna().unique())
	label_vector = labels.apply(lambda x:(-1 if pd.isnull([x]) else le.transform([x])[0]))
	#label_vector = label_vector.fillna(-1)
	#print(label_vector)
	text_vector_raw = data['Name']+data['Big Description']+data['Summary']
	text_vector = text_vector_raw.apply(stemming)
	vectorizer = TfidfVectorizer(sublinear_tf=True,max_df=0.5,stop_words='english')
	X_train = vectorizer.fit_transform(text_vector)
	label_prop_model = LabelPropagation()
	label_prop_model.fit(X_train.todense(), label_vector)
	labels_series = pd.Series(le.inverse_transform(label_prop_model.predict(X_train.todense())))
	data_labelled = pd.concat([data,labels_series],axis=1)
	#print(label_prop_model.predict_proba(X_train[600].todense()))
	data_labelled.to_csv('kaggle-data-set-labelled2.csv')
