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
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split
from sklearn import metrics

from collections import Counter
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

from imblearn.combine import SMOTEENN
from imblearn.under_sampling import ClusterCentroids

import matplotlib.pyplot as plt

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

df = pd.read_csv('../data/new_data/test.csv')
label_list = df['Classification'].dropna()
d_algo = df['Algorithm'].dropna()
d_algo_list = d_algo.apply(listify)
algo_list = d_algo_list.tolist()
label_list_2 = flatten(algo_list)
labels = np.unique(label_list)
# print len(labels)

print('Overall class distributions summary: {}'.format(Counter(label_list_2)))
# print('Test class distributions summary: {}'.format(Counter(y_test)))

# le = preprocessing.LabelEncoder()
# le.fit(labels)
# encoded_algo = d_algo_list.apply(lambda x : le.transform(x))
# m = MultiLabelBinarizer()
# y = m.fit_transform(encoded_algo)


df = df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1)
df_algo = df.loc[label_list.index]
df_algo['Title'].fillna(' ',inplace=True)
df_algo['Description'].fillna(' ',inplace=True)
df_algo['Summary'].fillna(' ',inplace=True)
df_algo['File_Type'].fillna(' ',inplace=True)
data_raw = df_algo['Title']+df_algo['Description']+df_algo['Summary']+df_algo['File_Type']
data = data_raw.apply(stemming)
tfidf = loaded_vectorizer.transform(data)
X_train,X_test,y_train,y_test = train_test_split(tfidf,label_list,test_size=0.2)
#X_train = loaded_vectorizer.transform(data_train)
#X_test = loaded_vectorizer.transform(data_test)

#smote_enn = SMOTEENN(random_state=0)
#X_resampled, y_resampled = smote_enn.fit_sample(X_train, y_train)
#cc = ClusterCentroids(random_state=0)
#X_resampled, y_resampled = cc.fit_sample(X_train, y_train)
#classif = OneVsRestClassifier(SVC(kernel='rbf'))
#classif = LogisticRegression(class_weight='balanced')
classif =  MLPClassifier(hidden_layer_sizes=(100,50), max_iter=100, alpha=1e-4,solver='adam', verbose=10, tol=1e-4, random_state=1, learning_rate_init=1e-3)
#classif = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),algorithm="SAMME",n_estimators=200)
classif.fit(X_train, y_train)
pred = classif.predict(X_test)
score = metrics.accuracy_score(y_test,pred)
cm = metrics.confusion_matrix(y_test,pred)
# print (le.inverse_transform(m.inverse_transform(pred)))
print ("Accuracy: "+str(score))
print ('Confusion matrix:')
print (cm)

#### ROC curves ####
fpr, tpr, thresh = metrics.roc_curve(y_test, pred)
plt.figure()
lw=2
plt.plot(fpr, tpr, color='red')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.show()
'''
clf = MLPClassifier(activation='relu',solver='sgd', alpha=1e-5, hidden_layer_sizes=(100, 50), random_state=8)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
score = metrics.accuracy_score(y_test,pred)
hamming_score = metrics.hamming_loss(y_test,pred)
print (le.inverse_transform(m.inverse_transform(pred)))
print ("Exact matching score: "+str(score))
print ("Hamming loss: "+str(hamming_score))

clf2 = RandomForestClassifier(max_depth=2, random_state=0)
clf2.fit(X_train, y_train)
pred2 = clf2.predict(X_test)
score2 = metrics.accuracy_score(y_test,pred2)
hamming_score2 = metrics.hamming_loss(y_test,pred2)
print ("Exact matching score: "+str(score2))
print ("Hamming loss: "+str(hamming_score2))


#clf2 = SGDClassifier(loss='log',alpha=.0001, n_iter=50,class_weight="balanced")
#clf2.fit(X_train, y_train)
# pred2 = clf2.predict(X_test,y_test)
'''


