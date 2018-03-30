import pandas as pd

df = pd.read_csv('all.csv')

for i in range(df.shape[0]):
	print df.iloc[i]['Keywords']
	# print type(df.iloc[i]['Keywords'])