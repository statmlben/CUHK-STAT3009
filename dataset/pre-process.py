import os
import pandas as pd

if not os.path.isfile('data.csv'):
    data = open('data.csv', mode='w')

file = 'combined_data_1.txt'

# Remove the line with movie_id: and add a new column of movie_id
print("Opening file: {}".format(file))
with open(file) as f:
	for line in f:
		line = line.strip()
		if line.endswith(':'):
			movie_id = line.replace(':', '')
		else:
			data.write(movie_id + ',' + line)
			data.write('\n')
data.close()

# Read all data into a pd dataframe
df = pd.read_csv('data.csv', names=['movie_id', 'user_id','rating','date'])


from sklearn import preprocessing
le_movie = preprocessing.LabelEncoder()
df['movie_id'] = le_movie.fit_transform(df['movie_id'])
le_user = preprocessing.LabelEncoder()
df['user_id'] = le_user.fit_transform(df['user_id'])

df = df[df['user_id'] < 2000]

le_movie = preprocessing.LabelEncoder()
df['movie_id'] = le_movie.fit_transform(df['movie_id'])
le_user = preprocessing.LabelEncoder()
df['user_id'] = le_user.fit_transform(df['user_id'])

# save the pre-processed data
df.to_csv('data.csv')

# train-test splitting
from sklearn.model_selection import train_test_split
dtrain, dtest = train_test_split(df, test_size=0.5, random_state=42)

dtrain.to_csv('train.csv', index=False)
dtest.to_csv('test.csv', index=False)
