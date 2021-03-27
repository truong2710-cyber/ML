import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

class MF(object):
    def __init__(self,Y_data,K,lam=0.1,Xinit=None,Winit=None,learning_rate=.5,max_iter=100,print_every=10,user_based=1):
        self.Y_raw_data=Y_data
        self.K=K
        self.lam=lam
        self.learning_rate=learning_rate
        self.max_iter=max_iter
        self.print_every=print_every
        self.user_based=user_based
        self.n_users=int(np.max(Y_data[:,0]))+1
        self.n_items=int(np.max(Y_data[:,1]))+1
        self.n_ratings=Y_data.shape[0]
        if Xinit is None:
            self.X=np.random.randn(self.n_items,K)
        else:
            self.X=Xinit
        if Winit is None:
            self.W=np.random.randn(K,self.n_users)
        else:
            self.W=Winit
        # normalized data, update later in normalized_Y function
        self.Y_data_n=self.Y_raw_data.copy()
    def normalize_Y(self):
        if self.user_based:
            user_col=0
            item_col=1
            n_objects=self.n_users
        else:
            user_col=1
            item_col=0
            n_objects=self.n_items
        users=self.Y_raw_data[:,user_col]
        self.mu=np.zeros((n_objects,))
        for n in range(n_objects):
            ids=np.where(users==n)[0]
            item_ids=self.Y_data_n[ids,item_col]
            ratings=self.Y_data_n[ids,2]
            m=np.mean(ratings)
            if np.isnan(m):
                m=0
            self.mu[n]=m
            self.Y_data_n[ids,2]=ratings-self.mu[n]
    def loss(self):
        L=0
        for i in range(self.n_ratings):
            n,m,rate=int(self.Y_data_n[i,0]),int(self.Y_data_n[i,1]),int(self.Y_data_n[i,2])
            L+=0.5*(rate-np.dot(self.X[m,:],self.W[:,n]))**2
        L/=self.n_ratings
        L+=0.5*self.lam*(np.linalg.norm(self.W,'fro')+np.linalg.norm(self.X,'fro'))
        return L
    def get_items_rated_by_user(self,user_id):
        ids=np.where(self.Y_data_n[:,0]==user_id)[0]
        item_ids=self.Y_data_n[ids,1].astype(int)
        ratings=self.Y_data_n[ids,2]
        return (item_ids,ratings)
    def get_users_who_rated_item(self,item_id):
        ids=np.where(self.Y_data_n[:,1]==item_id)[0]
        user_ids=self.Y_data_n[ids,0].astype(int)
        ratings=self.Y_data_n[ids,2]
        return (user_ids,ratings)
    def updateX(self):
        for m in range(self.n_items):
            user_ids,ratings=self.get_users_who_rated_item(m)
            What_m=self.W[:,user_ids]
            grad_xm=-(ratings-self.X[m,:].dot(What_m)).dot(What_m.T)/self.n_ratings+self.lam*self.X[m,:]
            self.X[m,:]-=self.learning_rate*grad_xm
    def updateW(self):
        for n in range(self.n_users):
            item_ids,ratings=self.get_items_rated_by_user(n)
            Xhat_n=self.X[item_ids,:]
            grad_wn=-Xhat_n.T.dot(ratings-Xhat_n.dot(self.W[:,n]))/self.n_ratings+self.lam*self.W[:,n]
            self.W[:,n]-=self.learning_rate*grad_wn
    def fit(self):
        self.normalize_Y()
        for it in range(self.max_iter):
            self.updateW()
            self.updateX()
            if (it + 1) % self.print_every == 0:
                rmse_train = self.RMSE(self.Y_raw_data)
                print('iter =', it + 1, ', loss =', self.loss(), ', RMSE train =', rmse_train)

    def pred(self,u,i):
        """
        pred the rating of user u for item i
        """
        if self.user_based:
            bias=self.mu[u]
        else:
            bias=self.mu[i]
        pred=self.X[i,:].dot(self.W[:,u])+bias
        if pred<0:
            return 0
        if pred>5:
            return 5
        return pred
    def pred_for_user(self,u):
        """
        predict rating of user u for all unrated items
        """
        ids=np.where(self.Y_data_n[:,0]==u)[0]
        items_rated_by_u=Y_data_n[ids,1].tolist()
        pred_ratings=[]
        for i in range(self.n_items):
            if i not in items_rated_by_u:
                pred_ratings.append(self.pred(u,i))
        return pred_ratings

    def RMSE(self,rate_test):
        n_tests=rate_test.shape[0]
        se=0
        for n in range(n_tests):
            se+=(rate_test[n,2]-self.pred(rate_test[n,0],rate_test[n,1]))**2
        rmse=np.sqrt(se/n_tests)
        return rmse

r_cols=['user_id', 'movie_id', 'rating', 'unix_timestamp']
rating_base=pd.read_csv('ml-100k/ub.base', sep='\t', names=r_cols, encoding='latin-1')
rating_test=pd.read_csv('ml-100k/ub.test', sep='\t', names=r_cols, encoding='latin-1')
rate_train=rating_base.to_numpy()
rate_test=rating_test.to_numpy()
rate_train[:, :2] -= 1
rate_test[:, :2] -= 1

rs=MF(rate_train,K=10,user_based=1)
rs.fit()
print(rs.RMSE(rate_test))
