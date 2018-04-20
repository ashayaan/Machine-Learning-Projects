import pandas as pd
import numpy as np
import pickle
import string
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

from sklearn.model_selection import train_test_split
from sklearn import metrics


def listify(x):
	y = x.translate(None, string.punctuation)
	return y.split()

flatten = lambda l: [item for sublist in l for item in sublist]

ps = PorterStemmer()
loaded_vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))

def stemming(sentence):
	if (type(sentence) is float):
		sentence = ''
	s = sentence.translate(None, string.punctuation)
	str1 = ''.join([i if ord(i) < 128 else ' ' for i in s])
	words = word_tokenize(str1)
	words_stemmed = map(lambda x : ps.stem(x), words)
	return ' '.join([s for s in words_stemmed])



df = pd.read_csv('../data/new_data/final_data1.csv')
d_algo = df['Algorithm'].dropna()
d_algo_list = d_algo.apply(listify)
algo_list = d_algo_list.tolist()
label_list = flatten(algo_list)
labels = np.unique(label_list)

le = preprocessing.LabelEncoder()
le.fit(labels)
encoded_algo = d_algo_list.apply(lambda x : le.transform(x))
y = MultiLabelBinarizer().fit_transform(encoded_algo)


df = df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1)
df_algo = df.loc[d_algo.index]
data_raw = df_algo['Title']+df_algo['Description']+df_algo['Summary']
data = data_raw.apply(stemming)
data_train,data_test,y_train,y_test = train_test_split(data,y,test_size=0.2)
X_train = loaded_vectorizer.transform(data_train)
X_test = loaded_vectorizer.transform(data_test)
clf = MLPClassifier(activation='relu',solver='sgd', alpha=1e-5, hidden_layer_sizes=(8, 2), random_state=8)

clf.fit(X_train, y_train)
pred = clf.predict(X_test)
score = metrics.accuracy_score(y_test,pred)


clf2 = SGDClassifier(loss='log',alpha=.0001, n_iter=50,class_weight="balanced")
clf2.fit(X_train, y_train)
# pred2 = clf2.predict(X_test,y_test)

print score
