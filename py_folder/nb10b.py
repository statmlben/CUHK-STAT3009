from sklearn.metrics import ndcg_score
import itertools

def ndcg_rs(test_pair, true_rating, pred_rating, k=10):
    ndcg = []
    user_lst = list(set(test_pair[:,0]))
    user_index = [np.where(test_pair[:,0] == user_tmp)[0] for user_tmp in user_lst]
    for user_tmp in user_lst:
        true_rating_tmp = true_rating[user_index[user_tmp]]
        pred_rating_tmp = pred_rating[user_index[user_tmp]]
        ndcg_tmp = ndcg_score([true_rating_tmp], [pred_rating_tmp], k=k)
        ndcg.append(ndcg_tmp)
    return np.mean(ndcg)

def PR_loss(test_pair, true_rating, pred_rating):
    PR_loss_lst = []
    user_lst = list(set(test_pair[:,0]))
    user_index = [np.where(test_pair[:,0] == user_tmp)[0] for user_tmp in user_lst]
    for user_tmp in user_lst:
        record_idx_tmp = user_index[user_tmp]
        for pair_tmp in itertools.combinations(record_idx_tmp, 2):
            diff_true = true_rating[pair_tmp[0]] - true_rating[pair_tmp[1]]
            diff_pred = pred_rating[pair_tmp[0]] - pred_rating[pair_tmp[1]]
            if diff_true != 0:
            	PR_loss_lst.append( 1*(diff_true*diff_pred <= 0) )
    return np.mean(PR_loss_lst)

import numpy as np
import pandas as pd
# load rating
df = pd.read_csv('./dataset/ml-latest-small/ratings.csv')
del df['timestamp']



## joint encoding for userId and movieId
# !!! all dfs should share the same encoding for userId and movieId, respecitively!!!
le_movie = preprocessing.LabelEncoder()
le_user = preprocessing.LabelEncoder()

df['movieId'] = le_movie.fit_transform(df['movieId'])
df['userId'] = le_user.fit_transform(df['userId'])


# tran_pair, train_rating
train_pair = dtrain[['userId', 'movieId']].values
train_rating = dtrain['rating'].values
# test_pair
test_pair = dtest[['userId', 'movieId']].values
n_user = max(train_pair[:,0].max(), test_pair[:,0].max())+1
n_item = max(train_pair[:,1].max(), test_pair[:,1].max())+1

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Embedding, Flatten, Input, Dropout, Dense, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

### fit the data with PR_loss

## Step 1: pre-processing dataset

# 1.1 get triple data
def gen_triple(pair, rating):
    triple, diff = [], []
    ## user list
    user_lst = list(set(pair[:,0]))
    user_index = [np.where(pair[:,0] == user_tmp)[0] for user_tmp in user_lst]
    for user_tmp in user_lst:
        record_idx_tmp = user_index[user_tmp]
        ## find all possible pairwise comparison of observed items under the users
        for pair_idx_tmp in itertools.combinations(record_idx_tmp, 2):
            diff_tmp = np.sign(rating[pair_idx_tmp[0]] - rating[pair_idx_tmp[1]])
            ## if diff is zero; no information; remove this triple
            if diff_tmp != 0:
                triple.append([user_tmp, pair[pair_idx_tmp[0], 1], pair[pair_idx_tmp[1], 1]])
                diff.append(diff_tmp)
    return np.array(triple), np.array(diff)

train_triple, train_diff = gen_triple(pair=train_pair, rating=train_rating)
train_diff = (.5*(train_diff+1)).astype(int)

# define model
class RankNCF(keras.Model):
    def __init__(self, num_users, num_movies, embedding_size, **kwargs):
        super(RankNCF, self).__init__(**kwargs)
        self.num_users = num_users
        self.num_movies = num_movies
        self.embedding_size = embedding_size
        self.user_embedding = layers.Embedding(
            num_users,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-3),
        )
        self.movie_embedding = layers.Embedding(
            num_movies,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-3),
        )
        self.concatenate = layers.Concatenate()
        self.dense1 = layers.Dense(16, name='fc-1', activation='relu')
        self.dense2 = layers.Dense(8, name='fc-2', activation='relu')
        self.dense3 = layers.Dense(1, name='fc-3', activation='linear')

    def scorer(self, user_id, movie_id):
        user_vector = self.user_embedding(user_id)
        movie_vector = self.movie_embedding(movie_id)
        concatted_vec = self.concatenate([user_vector, movie_vector])
        fc_1 = self.dense1(concatted_vec)
        fc_2 = self.dense2(fc_1)
        fc_3 = self.dense3(fc_2)
        return fc_3

    def call(self, inputs):
        user_id = inputs[:, 0]
        movie1_id = inputs[:, 1]
        movie2_id = inputs[:, 2]
        score1 = self.scorer(user_id, movie1_id)
        score2 = self.scorer(user_id, movie2_id)
        return score1 - score2

model = RankNCF(num_users=n_user, num_movies=n_item, embedding_size=20)
## Step 3: train the model

metrics = [
    keras.metrics.BinaryAccuracy(name='binary_acc')
]

model.compile(
    optimizer=keras.optimizers.Adam(1e-4), 
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), 
    metrics=metrics
)

history = model.fit(
    x=train_triple,
    y=train_diff,
    batch_size=128,
    epochs=2,
    verbose=1,
    validation_split=.2,
)

pred_rating = model.scorer(user_id = test_pair[:,0], movie_id = test_pair[:,1])
pred_rating = pred_rating.numpy().flatten()

print('NDCG: RankNCF: %.3f' %ndcg_rs(test_pair, true_rating=test_rating, pred_rating=pred_rating))
print('PR_loss: RankNCF: %.3f' %PR_loss(test_pair, true_rating=test_rating, pred_rating=pred_rating))