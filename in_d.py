import sklearn
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression as LR
from sklearn.linear_model import RANSACRegressor as RANSAC
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits, fetch_olivetti_faces

# your code here

def correlation_dimension(X, metric = 'euclidean' ):
    '''
    X : array [n_samples_a, n_samples_a] if metric == “precomputed”, or, 
            [n_samples_a, n_features] otherwise Array of pairwise distances between samples, or a feature array
    
    metric : The metric to use when calculating distance between instances in a feature array. 
             If metric is a string, it must be one of the options allowed by scipy.spatial.distance.pdist
             for its metric parameter, or a metric listed in pairwise.PAIRWISE_DISTANCE_FUNCTIONS. 
             If metric is “precomputed”, X is assumed to be a distance matrix.
    
    return list of C
    '''
    
    M = sklearn.metrics.pairwise_distances(X, metric = metric, n_jobs = -1)
    
    n = len(X)
    
    mini = M[M!= 0].min()
    maxi = M[M!= 0].max()
    list_or_r = np.exp(np.arange(np.log(mini), np.log(maxi), (np.log(maxi)-np.log(mini))/1000))
    list_of_C = []
    for r in list_or_r:
        list_of_C.append((np.sum(M < r) - n)/(n*(n-1)) )
    C = np.array(list_of_C)
    i = np.argwhere(C > 0)[:,0]
    C = C[i]
    R = list_or_r[i]
    return C, R

def estimate_corr_dimensional(X, metric = 'euclidean', plot = True):
    
    '''
    Calculated the correlation dimension and plot the figure
    X : array [n_samples_a, n_samples_a] if metric == “precomputed”, or, 
            [n_samples_a, n_features] otherwise Array of pairwise distances between samples, or a feature array
    
    metric : The metric to use when calculating distance between instances in a feature array. 
             If metric is a string, it must be one of the options allowed by scipy.spatial.distance.pdist
             for its metric parameter, or a metric listed in pairwise.PAIRWISE_DISTANCE_FUNCTIONS. 
             If metric is “precomputed”, X is assumed to be a distance matrix.
             
    
    '''
    C, R = correlation_dimension(X, metric = metric)
    Linear = LR()
    X = np.log(R)
    y = np.log(C)

    RC = RANSAC(base_estimator= Linear,  residual_threshold= 0.1*np.mean(np.absolute(y - np.mean(y))))
    RC.fit(X.reshape(-1,1), y)
    coef, inter = RC.estimator_.coef_, RC.estimator_.intercept_
    if plot:
        plt.figure(figsize = (6,6))
        plt.title("Correlation dimension estimation, ")
        plt.plot(X, y, c = 'b', label = 'correlation dimension')
        plt.plot(X, X*coef + inter, c = 'r', label = 'linear approximation, coef = {:.2f}'.format(coef[0]) )
        plt.xlabel('Log(r)')
        plt.ylabel('Log(C)')
        plt.legend()
        plt.grid()
        plt.show()
    return coef[0]


def intrinsic_dim_sample_wise(X, k=5):
    neighb = NearestNeighbors(n_neighbors=k+1).fit(X)
    dist, ind = neighb.kneighbors(X)
    dist = dist[:, 1:]
    dist = dist[:, 0:k]
    assert dist.shape == (X.shape[0], k)
    assert np.all(dist > 0)
    d = np.log(dist[:, k - 1: k] / dist[:, 0:k-1])
    d = d.sum(axis=1) / (k - 2)
    d = 1. / d
    intdim_sample = d
    return intdim_sample
 
def intrinsic_dim_scale_interval(X, k1=10, k2=20):
    X = pd.DataFrame(X).drop_duplicates().values # remove duplicates in case you use bootstrapping
    intdim_k = []
    for k in range(k1, k2 + 1):
        m = intrinsic_dim_sample_wise(X, k).mean()
        intdim_k.append(m)
    return intdim_k
 
def repeated(func, X, nb_iter=100, random_state=None, mode='bootstrap', **func_kw):
    if random_state is None:
        rng = np.random
    else:
        rng = np.random.RandomState(random_state)
    nb_examples = X.shape[0]
    results = []
 
    iters = range(nb_iter) 
    for i in iters:
        if mode == 'bootstrap':
            Xr = X[rng.randint(0, nb_examples, size=nb_examples)]
        elif mode == 'shuffle':
            ind = np.arange(nb_examples)
            rng.shuffle(ind)
            Xr = X[ind]
        elif mode == 'same':
            Xr = X
        else:
            raise ValueError('unknown mode : {}'.format(mode))
        results.append(func(Xr, **func_kw))
    return results

def estimate_mle_dim(data, k1 = 3, k2 = 50, nb_iter = 20, plot = True):
#     k1 = 3 # start of interval(included)
#     k2 = 50 # end of interval(included)
#     nb_iter = 20
    intdim_k_repeated = repeated(intrinsic_dim_scale_interval, 
                                 data, 
                                 mode='bootstrap', 
                                 nb_iter=nb_iter, # nb_iter for bootstrapping 
                                 k1=k1, k2=k2)
    intdim_k_repeated = np.array(intdim_k_repeated)
    if plot:
        fig = plt.figure(figsize=(12,5.25))

        plt.suptitle("Intrinsic dimension estimate via MLE")

        plt.subplot(121)
        plt.xlabel("Neighborhood cardinality")
        plt.ylabel("Intrinsic dimension")
        plt.grid(linestyle='dotted')

        plt.plot(range(k1, k2 + 1), np.mean(intdim_k_repeated, axis=0), 'b')
        plt.plot(range(k1, k2 + 1), np.mean(intdim_k_repeated, axis=0) + np.std(intdim_k_repeated, axis=0), 'r')
        plt.plot(range(k1, k2 + 1), np.mean(intdim_k_repeated, axis=0) - np.std(intdim_k_repeated, axis=0), 'r')

        plt.subplot(122)
        plt.xlabel("Intrinsic dimension")
        plt.grid(linestyle="dotted")

        plt.hist(intdim_k_repeated.mean(axis=0))
        plt.show()
        
    values = np.histogram(intdim_k_repeated.mean(axis=0))
    num  = values[0]
    bins = values[1]
    i = np.argmax(num)
    coef = bins[i]
    plt.show()
    return coef

def estimate_local_pca(data, k = 15, plot = True):
    X = data

    # select neighborhood for each point
    nn = NearestNeighbors(n_neighbors=k).fit(X)
    idx = nn.kneighbors(return_distance=False)
    U = X[idx]

    eigenvalues = np.zeros((X.shape[-1], X.shape[1]), np.float)
    for i in range(X.shape[0]):

        # center the data in neighborhood by subtracting x_i
        data_centered = U[i] - X[i]

        # estimate covariance matrix
        covariance = np.dot(data_centered.T, data_centered) / (k - 1)

        # SVD decomposition
        _, eigenvalues_i, _ = np.linalg.svd(covariance)
#         print(eigenvalues_i.shape)
        # add local eigenvalues to array of eigenvalues
        eigenvalues = np.vstack((eigenvalues, eigenvalues_i))
    eigenvalues_mean = np.mean(eigenvalues, axis=0)
    eigenvalues_mean
    EV = eigenvalues_mean / eigenvalues_mean.sum()
    CEV = np.cumsum(EV)
    
    if plot:
        fig = plt.figure(figsize=(12,5.25))

        plt.subplot(121)
        plt.title("Explained variance")
        plt.xlabel("# PCs")
        plt.grid(linestyle="dotted")
        plt.loglog(EV, "o-")

        plt.subplot(122)
        plt.title("Cumulative explained variance")
        plt.axhline(linewidth=1, y=0.99, color='r')
        plt.axhline(linewidth=1, y=0.95, color='r')
        plt.axhline(linewidth=1, y=0.9, color='r')
        plt.axhline(linewidth=1, y=0.8, color='r')
        plt.xlabel("# PCs")
        plt.grid(linestyle="dotted")
        plt.plot(CEV, "o-")
        plt.show()
    return np.sum(CEV< 0.95)

def EV_i(i, eigenvalues):
    return eigenvalues[i]/eigenvalues.sum()
def CEV_d(d, eigenvalues):
    return eigenvalues[:d].sum()/eigenvalues.sum()

def estimate_pca(data, plot = True):
    # apply PCA
    X = data
    pca = PCA()
    pca.fit(X)

    EV = []
    CEV = []
#     print(pca.explained_variance_.shape)
    for i in range(pca.explained_variance_.shape[0]):
        EV.append(EV_i(i, pca.explained_variance_))
        CEV.append(CEV_d(i, pca.explained_variance_))
    # plot EV/CEVs
    if plot:
        fig = plt.figure(figsize=(12,5.25))

        plt.subplot(121)
        plt.title("Explained variance")
        plt.xlabel("# PCs")
        plt.grid(linestyle="dotted")
        plt.loglog(EV, "o-")

        plt.subplot(122)
        plt.title("Cumulative explained variance")
        plt.axhline(linewidth=1, y=0.99, color='r')
        plt.axhline(linewidth=1, y=0.95, color='r')
        plt.axhline(linewidth=1, y=0.9, color='r')
        plt.axhline(linewidth=1, y=0.8, color='r')
        plt.xlabel("# PCs")
        plt.grid(linestyle="dotted")
        plt.plot(CEV, "o-")
        plt.show()
#     print('The number of variables explaining 80% is', (np.array(CEV)<0.8).sum())
    
#     print('The number of variables explaining 95% is', (np.array(CEV)<0.95).sum())
    return (np.array(CEV)<0.95).sum()


class SFS():
#         self.set = [0]*size
    def fit(self, model, X, y, cv = 5):
        self.history_train = []
        self.history_test = []
        self.history_index = []
        n_features = X.shape[-1]
        for rep in range(12):
#             print(rep)
            dct_te = []
            dct_tr = []
            for j in range(X.shape[-1]):
                if j in self.history_index:
                    dct_te.append(np.inf)
                    dct_tr.append(np.inf)
                else:
                    scores = {
                         'fit_time'   : [],
                         'score_time' : [],
                         'test_score' : [],
                         'train_score': []
                    }
                    for col in range(y.shape[-1]):
                        if col in self.history_index:
                            continue
                        if col == j:
                            continue
                        score = cross_validate(model, X[:,  self.history_index + [j]], y[:, col], cv=cv,
                                                scoring = 'neg_root_mean_squared_error', return_train_score = True, n_jobs = -1)
                        for tar in score:
                            scores[tar].append(score[tar])
                    dct_tr.append(-np.mean(scores['train_score']))
                    dct_te.append(-np.mean(scores['test_score']))
            i = np.argmin(dct_te)
            self.history_index.append(i)
            self.history_test.append(dct_te[i])
            self.history_train.append(dct_tr[i])
            j = np.argmin(dct_te)
            if abs(i - j) > 10:
                break
#             self.set.append(i)
        return {'set' : self.history_index,
                'test': self.history_test,
                'train' : self.history_train
               }

class oldSFS():
#         self.set = [0]*size
    def fit(self, model, X, y, cv = 5):
        self.history_train = []
        self.history_test = []
        self.history_index = []
        n_features = X.shape[-1]
        for rep in range(12):
#             print(rep)
            dct_te = []
            dct_tr = []
            for j in range(X.shape[-1]):
                if j in self.history_index:
                    dct_te.append(np.inf)
                    dct_tr.append(np.inf)
                else:
                    predicted_indexes = np.ones(n_features, dtype= np.bool)
                    predicted_indexes[self.history_index + [j]] = 0
                    scores = cross_validate(model, X[:, self.history_index + [j]], y[:,  predicted_indexes], cv=cv,
                                            scoring = 'neg_root_mean_squared_error', return_train_score = True, n_jobs = -1)
                    dct_tr.append(-np.mean(scores['train_score']))
                    dct_te.append(-np.mean(scores['test_score']))
            i = np.argmin(dct_te)
            self.history_index.append(i)
            self.history_test.append(dct_te[i])
            self.history_train.append(dct_tr[i])
            j = np.argmin(dct_te)
            if abs(i - j) > 10:
                break
#             self.set.append(i)
        return {'set' : self.history_index,
                'test': self.history_test,
                'train' : self.history_train
               }

import datetime
def today_date():
    today = datetime.date.today()
    month =str(today.month) 
    day   =str(today.day) 
    if len(day) <2:
        day = '0' + day
    if len(month) <2:
        month = '0' + month
    return month+'_'+day