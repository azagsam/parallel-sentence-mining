import pandas as pd

# import data
df = pd.read_csv('output/final.csv')

# filter data based on threshold from low to high and save to disk
df_low = df[df['avg_score'] > 1.1]
df_low.to_csv('output/low.csv', index=False)

df_medium = df[df['avg_score'] > 1.2]
df_medium.to_csv('output/medium.csv', index=False)

df_high= df[df['avg_score'] > 1.3]
df_high.to_csv('output/high.csv', index=False)