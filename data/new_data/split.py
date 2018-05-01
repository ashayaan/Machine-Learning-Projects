import pandas as pd
import numpy as np
import string
#def sperateAlgorithm(x):
#	if type(x) == type("S"):
#		return x.strip('[]').split(",")
#	else:
#		return 0

def listify(x):
	if (type(x) == float):
		return []
	y = x.translate(None, string.punctuation)
	return y.split()
	

flatten = lambda l : [item for sublist in l for item in sublist]

if __name__ == '__main__':
	df = pd.read_csv('check_data.csv')
	df = df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1)
	d_algo = df['Algorithm'].dropna()
	d_algo_list = d_algo.apply(listify)
	algo_list = d_algo_list.tolist()
	label_list = flatten(algo_list)
	labels = np.unique(label_list)
	print(labels)
	generate_labels = ['BigQuery','CausalDiscovery','Classification','Clustering','DataCompression','DataMining','DeepLearning','Detection','Events','FaceDetection','FaceRecognition','FunctionLearning','GeospatialAnalysis','GraphAnalysis','HandwritingRecognition','ImageClassification','NLP','ObjectDetection','ObjectRecognition','Optimization','RecommenderSystems','Regression','RelationalLearning','Segmentation','SentimentAnalysis','SpeechRecognition','Summarization','TimeSeries','TopicModeling','TransferLearning','UnsupervisedClassification','VideoClassification','Visualization']
		
	for label in generate_labels:
		df[label] = None
	
	
	for label in generate_labels:
		count = 0
		for x in df['Algorithm']:
			l = listify(x)
			count+=1
			if l == 0:
				continue
			else:
				for t in l:
					if t.strip() == label:
						df.iloc[count-1, df.columns.get_loc(label)]=1
					else:
						df.iloc[count-1, df.columns.get_loc(label)]=0

	df.to_csv('test.csv',index=False)
