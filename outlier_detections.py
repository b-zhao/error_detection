# Various methods for getting outliers

import numpy as np
from scipy import linalg, stats
from sklearn.neighbors import KDTree

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
    print(dist[:20])
    lrd = k / (np.sum(dist, axis=1))
    sum_lrd_o = np.zeros(N)
    for i in range(N):
        sum_lrd_o[i] = np.sum(lrd[idx[i]])
    
    lof = sum_lrd_o / lrd / k
    return lof

def LOF2prob(lof):
    '''P = lof
    P[P <= 1] = 1
    P[P > 1] = 0'''
    P = 1 / lof
    return P
    
 
"""
Use Mahalanobis distance and chi-square to find outliers.
Reference: https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Interval
Args:
    X: NxD numpy array, the observations
    mu: D numpy array, the center for each gaussian. 
    sigma: DxD numpy array, the covariance matrix of each gaussian.
Return:
    P: N numpy array, the probability that P is in the distribution
"""
def gaussian_prob(X, mu, sigma):
    v = (X - mu)
    Mahalanobis_dist = np.matmul(v, linalg.pinv(sigma))
    Mahalanobis_dist = np.einsum('ij,ji->i', Mahalanobis_dist, v.T)
    # print(Mahalanobis_dist[:10])
    k = np.shape(X)[1]
    
    chi2_p = 1 - stats.chi2.cdf(Mahalanobis_dist, k)
    # print(chi2_p[:10])
    return chi2_p

