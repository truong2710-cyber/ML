import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

class CF(object):
    def __init__(self,Y_data,k,dist_func=cosine_similarity, uuCF=1):
        self.uuCF=uuCF
        self.Y_data=Y_data if uuCF==1 else Y_data[:,[1,0,2]]
        self.k=k # number of neighbor points
        self.dist_func=dist_func
        self.Ybar_data=None
        self.n_users = int(np.max(self.Y_data[:, 0])) + 1 
        self.n_items = int(np.max(self.Y_data[:, 1])) + 1
    def add(self, new_data):
        """
        Update Y_data matrix when new ratings come.
        For simplicity, suppose that there is no new user or item.
        """
        self.Y_data = np.concatenate((self.Y_data, new_data), axis = 0)
    def normalize_Y(self):
        users=self.Y_data[:,0] # all users - first col of Y_data
        self.Ybar_data=self.Y_data.copy()
        self.mu = np.zeros((self.n_users,))
        for n in range(self.n_users):
            ids=np.where(users==n)
            item_ids=self.Y_data[ids,1]
            ratings=self.Y_data[ids,2]
            m=np.mean(ratings)
            if np.isnan(m):
                m=0
            self.mu[n]=m
            # normalize
            self.Ybar_data[ids,2]=ratings-self.mu[n]
        self.Ybar=sparse.coo_matrix((self.Ybar_data[:,2],(self.Ybar_data[:,1],self.Ybar_data[:,0])),(self.n_items,self.n_users))
        self.Ybar = self.Ybar.tocsr()

    def similarity(self):
        self.S=self.dist_func(self.Ybar.T,self.Ybar.T)
    
    def refresh(self):
        self.normalize_Y()
        self.similarity()
    def fit(self):
        self.refresh()

    def __pred(self,u,i, normalized=1):
        """
        predict the rating of user u for item i (normalized)
        """
        # find all users who rated i
        ids=np.where(self.Y_data[:,1]==i)[0]
        users_rated_i=(self.Y_data[ids,0]).astype(int)
        # find similarity between u and users_rated_i
        sim=self.S[u,users_rated_i]
        # find the k-most similar users
        a=np.argsort(sim)[-self.k:]
        # and the corresponding similarity level
        nearest_s=sim[a]

        r=self.Ybar[i,users_rated_i[a]]
        if normalized:
            # add a small number, for instance, 1e-8, to avoid dividing by 0
            return (r*nearest_s)[0]/(np.abs(nearest_s).sum() + 1e-8)

        return (r*nearest_s)[0]/(np.abs(nearest_s).sum() + 1e-8) + self.mu[u]
    
    def pred(self,u,i,normalized=1):
        """
        predict the rating of user u for item i (normalized)
        """
        if self.uuCF:
            return self.__pred(u,i,normalized)
        return self.__pred(i,u,normalized)
    def recommend(self,u,normalized=1):
        """
        Determine all items should be recommended for user u. (uuCF =1)
        or all users who might have interest on item u (uuCF = 0)
        The decision is made based on all i such that:
        self.pred(u, i) > 0. Suppose we are considering items which 
        have not been rated by u yet. 
        """
        ids=np.where(self.Y_data[:,0]==u)[0]
        items_rated_by_u=self.Y_data[ids,1].tolist()
        recommend_items=[]
        for i in range(self.n_items):
            if i not in items_rated_by_u:
                rating=self.__pred(u,i)
                if rating>0:
                    recommend_items.append(i)
        return recommend_items
    def print_recommendation(self):
        for u in range(self.n_users):
            if self.uuCF==1:
                print("Recommend items ",self.recommend(u)," for user ",u)
            else:
                print("Recommed item ",u," for user ",self.recommend(u))
# data file 
r_cols = ['user_id', 'item_id', 'rating']
ratings = pd.read_csv('ex.dat', sep = ' ', names = r_cols, encoding='latin-1')
Y_data = ratings.to_numpy()

rs = CF(Y_data, k = 2, uuCF = 1)
rs.fit()

rs.print_recommendation()
    
r_cols=["user_id","movie_id","rating","unix_timestamp"]
ratings_base=pd.read_csv("ml-100k/ub.base",sep='\t', names=r_cols, encoding='latin-1')
ratings_test=pd.read_csv("ml-100k/ub.test",sep='\t', names=r_cols, encoding='latin-1')
rate_train = ratings_base.to_numpy()
rate_test = ratings_test.to_numpy()
rate_train[:,:2]-=1
rate_test[:,:2]-=1

rs=CF(rate_train,k=20,uuCF=1)
rs.fit()
n_tests=rate_test.shape[0]
SE=0
for n in range(n_tests):
    pred=rs.pred(rate_test[n,0],rate_test[n,1],normalized=0) 
    SE+=(pred-rate_test[n,2])**2
RMSE=np.sqrt(SE/n_tests)
print("User-user CF, RMSE = ",RMSE)   

