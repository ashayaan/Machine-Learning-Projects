import pandas as pd
import numpy as np


def sperateAlgorithm(x):
	if type(x) == type("S"):
		return x.strip('[]').split(",")
	else:
		return 0



if __name__ == '__main__':
	df = pd.read_csv('final_data1.csv')
	df = df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1)
	generate_labels = ['Regression','DataMining','Classification','Clustering','Visualization','TimeSeries','GeospatialAnalysis','NLP','SentimentAnalysis']
		
	for label in generate_labels:
		df[label] = None
	
	
	for label in generate_labels:
		count = 0
		for x in df['Algorithm']:
			l = sperateAlgorithm(x)
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