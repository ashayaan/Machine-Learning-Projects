import pandas as pd

df_kaggle = pd.read_csv('kaggle2.csv')
df_datagov = pd.read_csv('datagov2.csv')
df_merged = pd.concat([df_kaggle, df_datagov], ignore_index=True)
df_merged.to_csv('kaggle_datagov1.csv', encoding='utf8')

