import pandas as pd
import numpy as np


def sperateAlgorithm(x):
	if type(x) == type("S"):
		return x.strip('[]').split(",")
	else:
		return 0

if __name__ == '__main__':
	df = pd.read_csv('final_data1.csv')
	df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1)
	df['Label'] = 0
	# print df['Label']
	count = 0
	for x in df['Algorithm']:
		l = sperateAlgorithm(x)
		count+=1
		if l == 0:
			continue
		else:
			for t in l:
				if t.strip() == "Regression" or t.strip() == "regression" :
					df.iloc[count-1, df.columns.get_loc('Label')]=1
					# df.iloc[count-1, 'Label']=1

	print df
	