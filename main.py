import numpy as np
from GMM import *
from get_affinity_matrices import *
from label_assignment import *
from isolationForest import *
from outlier_detections import *

def main():
    X = affinity_matrix_binary("data/employee.csv")
    num_rows, _, num_cols = np.shape(X)
    print("affinity matrix dimension: ", np.shape(X))
    X = X.reshape((num_rows**2, num_cols))
    
    # Gaussian for outlier
    normalized_X = np.sqrt(X / np.sum(X, axis=0)) # normalize by column
    gamma, (pi, mu, sigma) = GMM()(normalized_X, K=1, max_iters=100)
    mu = mu.reshape(np.shape(mu)[1])
    sigma = sigma.reshape((np.shape(sigma)[1:]))
    P = gaussian_prob(normalized_X, mu, sigma)
    P = P.reshape((num_rows, num_rows))

    
    '''# Local Outlier Factor
    P = LOF(X)
    P = LOF2prob(P)
    P = P.reshape((num_rows, num_rows))'''

    '''# Isoforest for outlier
    IF = IsolationForest(X, trainSubset=100, trainCount=200, threshold=0.99)
    IF.train()
    inlier_idx = IF.getInlierIndex()    
    P = np.zeros((num_rows**2))
    P[inlier_idx] = 1
    P = P.reshape((num_rows, num_rows))'''
 
    # assign labels
    A = assign_label(P, lr=0.01, max_iter=200)
    print("First 20 entries in A: ", A[:20])



if __name__ == '__main__':
    main()
    