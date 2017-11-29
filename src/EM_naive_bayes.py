import numpy as np
import pandas as pd
import string

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


ps = PorterStemmer()
def stemming(sentence):
	s = sentence.translate(None, string.punctuation)
	str1 = ''.join([i if ord(i) < 128 else ' ' for i in s])
	words = word_tokenize(str1)
	words_stemmed = map(lambda x : ps.stem(x), words)
	return ' '.join([s for s in words_stemmed])


df = pd.read_csv('kaggle-data-set.csv')
data = df.dropna().copy() 
labels = data['Label']
le = LabelEncoder()
le.fit(labels.dropna().unique())
print(labels.dropna().unique())
#label_vector = labels.apply(lambda x:(-1 if pd.isnull([x]) else le.transform([x])[0]))
#label_vector = label_vector.fillna(-1)
#print(label_vector)
text_vector_raw = data['Name']+data['Big Description']+data['Summary']
text_vector = text_vector_raw.apply(stemming)
vectorizer = TfidfVectorizer(sublinear_tf=True,max_df=0.5,stop_words='english')
X = vectorizer.fit_transform(text_vector)
#print X.shape

def NaiveBayes(X_train, y_train):
	numClasses = len(y_train.unique()) #c
	(numSamples,numAttr) = X_train.shape
	mus = pd.DataFrame(np.zeros((numClasses,numAttr)))
	sigmas = pd.Series(np.zeros((numAttr,)))
	labels = pd.Series(y_train.unique())
	X_cs = labels.apply(lambda c: X_train[y_train==c])
	series = pd.Series(np.arange(len(X_cs)))
	series_sizes = series.apply(lambda x: X_cs[x].shape[0])
	#print series_sizes
	ucs = series.apply(lambda c: (X_cs[c].mean(axis=0)))
	u = pd.concat([pd.DataFrame(ucs[i]) for i in range(len(ucs))],axis=0).reset_index(drop=True)
	#temp = pd.concat([pd.DataFrame(ucs.loc[0])]*(series_sizes.loc[0]))
	#print temp.shape
	#print series_sizes.loc[0]
	sigma = series.apply(lambda x: np.square(X_cs[x] - pd.concat([pd.DataFrame(ucs.loc[x])]*series_sizes.loc[x])))
	sigma_mat = pd.concat([pd.DataFrame(sigma[i]) for i in range(len(sigma))],axis=0).reset_index(drop=True)
	sigma_vec = (sigma_mat.sum(axis=0))/numSamples
	return (u,sigma_vec)	

#def ExpectationMaximization(numClasses, numIterations):
	#c(numClasses) random values summing up to 1
	#for i in np.arange(len(numIterations)):
		#psi_mat =  
		#psi_mat_normalised  = 
	#	pi = psi_mat_normalised.sum(axis=0)
	#	mu = psi_mat_normalised.transpose() * X
	#	sigma = psi_mat_normalised.transpose() * sigma_vec (sigma_vec calculated as in previous function). nSamples to be replaced by npi...
	#

initialmodel = NaiveBayes(X.todense(),labels)
	

