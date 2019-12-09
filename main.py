import numpy as np
from get_affinity_matrices import affinity_matrix_binary, affinity_matrix_similarity
from label_assignment import assign_label
from outlier_detections import gaussian_prob, LOF, LOF2prob, IsolationForest
from get_features import get_feature_vecs

'''
filename: a csv file 
upper_only: only use distinct tuple pairs (above diagonal in affinity matrix) for Gaussian distribution 
returns a vector A, A[i]=Prob(tuple i has no error)
'''
def affinity_approach(filename, upper_only):
    X1 = affinity_matrix_binary(filename)
    X2 = affinity_matrix_similarity(filename)
    X = np.concatenate((X1, X2), axis=2)
    num_rows, _, num_cols = np.shape(X)
    
    if upper_only:
        triu_idx = np.triu_indices(num_rows, 1)
        X = X[triu_idx]
    else:
        X = X.reshape((num_rows**2, num_cols))
    
    # now each row of X is a similarity vector of a tuple pair
    # get probability of containing no error, for every tuple pair
    
    # Gaussian for outlier detection
    p_n = 0.99 # estimated probability of non-outlier instances
    P = gaussian_prob(X, p_n)
    
    '''# LOF for outlier detection
    lof = LOF(X, leaf_size=2, k=100)
    P = LOF2prob(lof)'''
    
    '''# Isoforest for outlier detection
    IF = IsolationForest(X, trainSubset=100, trainCount=10, threshold=0.99)
    IF.train()
    inlier_idx = IF.getInlierIndex()    
    P = np.zeros((np.shape(X)[0]))
    P[inlier_idx] = 1'''
    
    # if upper_only, restore full size matrix P by seting diagonal and lower trianger to 0
    if upper_only:
        restored_P = np.zeros((num_rows, num_rows))
        counter = 0
        for r in range(num_rows):
            for c in range(r + 1, num_rows):
                restored_P[r][c] = P[counter]
                counter += 1
        P = restored_P
    else:
        P = P.reshape((num_rows, num_rows))
 
    # assign labels
    # use loss_func='method2' if method1 does not work
    A = assign_label(P, upper_only, loss_func='method1', lr=0.001, max_iter=800)
    # A = assign_label(P, upper_only, loss_func='method2', lr=0.00001, max_iter=2000)

    return A

'''
filename: a csv file 
returns a vector P, P[i]=Prob(tuple i has no error)
'''
def single_row_feature_approach(filename):
    X = get_feature_vecs(filename)
    
    # Gaussian for outlier detection
    p_n = 0.99 # estimated probability of non-outlier instances
    P = gaussian_prob(X, p_n)
    
    '''# LOF for outlier detection
    lof = LOF(X, leaf_size=2, k=100)
    P = LOF2prob(lof)'''
    
    '''# Isoforest for outlier detection
    IF = IsolationForest(X, trainSubset=100, trainCount=10, threshold=0.99)
    IF.train()
    inlier_idx = IF.getInlierIndex()    
    P = np.zeros((np.shape(X)[0]))
    P[inlier_idx] = 1'''
    
    return P


if __name__ == '__main__':
    A = affinity_approach("data/employee_err.csv", True)
#     A = single_row_feature_approach("data/employee_err.csv")
    print("First 20 entries in A: ", A[:10])
    