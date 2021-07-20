import numpy as np
import pandas as pd

## simulate data from latant factor model
n, m, K, n_obs = 1000, 500, 5, 5000
P, Q = np.random.randn(n, K), np.random.randn(m, K)
mean_user, mean_item = np.random.randn(n), np.random.randn(m)

ratings = []
for i in range(n_obs):
	user_tmp = np.random.randint(n)
	item_tmp = np.random.randint(m)
	rating_tmp = mean_user[user_tmp] + mean_item[item_tmp] + np.dot(P[user_tmp], Q[item_tmp])
	ratings.append([(user_tmp, item_tmp), rating_tmp])
df = pd.DataFrame(ratings, columns =['(user_id, item_id)', 'ratings'])
df = df.drop_duplicates()
## add some noise
n_obs = len(df)
df['ratings'] = df['ratings'] + .1*np.random.randn(n_obs)
## generate solution
from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.33, random_state=42)
train, test = train.reset_index(drop=True), test.reset_index(drop=True)
test_pub = test[['(user_id, item_id)']]
sub = test.copy()
sub['ratings'] = 1.
sub = sub.reset_index(drop=True)
## save data
train.to_csv('train.csv',index=False)
test_pub.to_csv('test.csv',index=False)
test.to_csv('sol.csv',index=False)
sub.to_csv('sample_submission.csv',index=False)
