# Various methods for getting outliers

import numpy as np
import scipy
import pandas as pd
from scipy import linalg, stats
from sklearn.neighbors import KDTree
from sklearn.ensemble import IsolationForest as IF

'''
Local Outlier Factor 
Reference: Breunig MM, Kriegel HP, Ng RT, Sander J. LOF: Identifying Density-Based 
Local Outliers. In: Proceedings of the 2000 ACM SIGMOD International Conference on 
Management of Data. Dallas, Texas, USA: ACM Press; 2000. p. 93–104.

This function will not work if the majority rows in X are identical. 
Args:
    X: NxD numpy array
    leaf_size: KDTree leaf size
    k: number of nearest neighbor
Return:
    lof: Nx1 numpy array, the local outlier factor for each data point
'''
def LOF(X, leaf_size=2, k=100):
    N = np.shape(X)[0]
    kdt = KDTree(X, leaf_size=2, metric='euclidean')
    dist, idx = kdt.query(X, k)
    # print(dist[:20])
    lrd = k / (np.sum(dist, axis=1))
    sum_lrd_o = np.zeros(N)
    for i in range(N):
        sum_lrd_o[i] = np.sum(lrd[idx[i]])
    
    lof = sum_lrd_o / lrd / k
    return lof

def LOF2prob(lof):
    P = 1 / lof
    # P[P > 1] = 1
    return P
    
 
"""
Use Mahalanobis distance and chi-square to find outliers.
Reference: Markus Goldstein. "Anomaly Detection in Large Datasets", 
PhD-Thesis, Pages 248, Technische Universitaet Kaiserslautern, 
Dr. Hut Verlag Muenchen, 2/2014. ISBN: 978-3-8439-1572-4.
Args:
    X: NxD numpy array, the observations
    mu: D numpy array, the center for each gaussian. 
    sigma: DxD numpy array, the covariance matrix of each gaussian.
Return:
    P: N numpy array, the probability that P is in the distribution
"""
def gaussian_prob(X, p_n):
    mu = np.mean(X, axis=0)
    sigma = np.cov(X, rowvar=0)
    v = (X - mu)
    '''
    Mahalanobis_dist = np.zeros((np.shape(X)[0]))
    for i in range(np.shape(X)[0]):
        Mahalanobis_dist[i] = scipy.spatial.distance.mahalanobis(X[i], mu, linalg.pinv(sigma))
    print(Mahalanobis_dist[:10])   
    '''
    Mahalanobis_dist = np.matmul(v, linalg.pinv(sigma))
    Mahalanobis_dist = np.einsum('ij,ji->i', Mahalanobis_dist, v.T)
    Mahalanobis_dist = np.sqrt(Mahalanobis_dist)
    # print(Mahalanobis_dist[:10])
    k = np.shape(X)[1]
    
    CMGOS = Mahalanobis_dist / stats.chi2.ppf(p_n, k)
    
    # "CMGOS will be lower than 1.0 for normal instances and larger for outliers"
    # convert form CMGOS to probability
    P = 1 / (CMGOS + 1e-20)
    P = P / np.max(P)
    return P


''' isolation forest for finding outliers '''
class IsolationForest:
    def __init__(self, X, trainSubset = 50, trainCount = 10, threshold = 0.6):
        self.X = X
        print('X shape:', self.X.shape)

        self.trainSubset = trainSubset
        self.trainCount  = trainCount
        self.threshold   = threshold
        self.inlier_X = []
        self.inlier_idx = []
        self.outlier_X = []

    def train(self):
        mask = None
        for i in range(self.trainCount):
            if i % 10 == 0:
                print('select a random subset of entries as training data for the', i + 1, 'time')   
            testX = self.X[np.random.choice(self.X.shape[0], self.trainSubset, replace=False), :]
            clf = IF(behaviour='new', contamination='auto')
            clf.fit(testX)
            pred = clf.predict(self.X)
            if mask is None:
                mask = pred 
            else:
                mask = mask + pred 
        threshold = self.threshold
        bo = mask >= threshold * 1 + (1 - threshold) * -1
        self.inlier_X  = self.X[bo]
        self.outlier_X = self.X[bo == False]
        self.inlier_idx = bo

    def getInlierX(self):
        return self.inlier_X

    def getOutlierX(self):
        return self.outlier_X
    
    def getInlierIndex(self):
        return self.inlier_idx

'''
# load data
df = pd.read_csv('data/Hospital.csv')
X = np.array(df)
col_names = np.array(df.columns)

# randomly picked two columns for outlier detection
temp = IsolationForest(X[:,[0,5]])
temp.train()
X_outlier = temp.getOutlierX()

print("number of outliers: " + str(np.shape(X_outlier)[0]))
'''
