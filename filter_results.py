import pandas as pd

# import data
df = pd.read_csv('output/out.csv')

# filter data based on threshold from low to high and save to disk
df_small = df[df['avg_score'] > 1.0]
df_small.to_csv('output/small.csv', index=False)

df_medium = df[df['avg_score'] > 1.1]
df_medium.to_csv('output/medium.csv', index=False)

df_high= df[df['avg_score'] > 1.2]
df_high.to_csv('output/high.csv', index=False)