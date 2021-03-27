import pandas as pd
import numpy as np
from math import sqrt
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import Ridge
from sklearn import linear_model

#Read user file:
u_cols=['user_id', 'age', 'sex', 'occupation', 'zip_code']
users=pd.read_csv('ml-100k/u.user',sep="|",names=u_cols,encoding='latin-1')
n_users=users.shape[0]
print(users.head())

#Read ratings file:
r_cols=['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings_base = pd.read_csv('ml-100k/ua.base', sep='\t', names=r_cols, encoding='latin-1')
ratings_test = pd.read_csv('ml-100k/ua.test', sep='\t', names=r_cols, encoding='latin-1')
rate_train = ratings_base.to_numpy()
rate_test = ratings_test.to_numpy()

#Reading items file:
i_cols = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

items = pd.read_csv('ml-100k/u.item', sep='|', names=i_cols, encoding='latin-1')
X0=items.to_numpy()
X_train_counts=X0[:,-19:]
transformer = TfidfTransformer(smooth_idf=True, norm ='l2')
tfidf = transformer.fit_transform(X_train_counts.tolist()).toarray()

def get_items_rated_by_users(rate_matrix, user_id):
    y=rate_matrix[:,0]
    ids = np.where(y == user_id +1)[0] 
    item_ids = rate_matrix[ids, 1] - 1 # index starts from 0 
    scores = rate_matrix[ids, 2]
    return (item_ids, scores)

d=tfidf.shape[1] #data dimension
W = np.zeros((d, n_users))
b = np.zeros((1, n_users))
for n in range(n_users):
    ids,scores=get_items_rated_by_users(rate_train,n)
    clf=Ridge(alpha=0.01, fit_intercept=True)
    Xhat=tfidf[ids,:]
    clf.fit(Xhat,scores)
    W[:,n]=clf.coef_
    b[0,n]=clf.intercept_

#predicted scores
Yhat=tfidf.dot(W)+b

n=11
np.set_printoptions(precision=2)
ids,scores=get_items_rated_by_users(rate_test,n)
print('Rated movies ids :', ids )
print('True ratings     :', scores)
print('Predicted ratings:', Yhat[ids, n])

def evaluate(Yhat,rates,W,b):
    se=0
    cnt=0
    for n in range(n_users):
        ids,scores_truth=get_items_rated_by_users(rates,n)
        scores_pred=Yhat[ids,n]
        e=scores_truth-scores_pred
        se+=np.linalg.norm(e)
        cnt+=e.size
    return sqrt(se/cnt)

print('RMSE for training:', evaluate(Yhat, rate_train, W, b))
print('RMSE for testing:', evaluate(Yhat, rate_test, W, b))