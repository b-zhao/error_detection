import numpy as np
from get_affinity_matrices import affinity_matrix_binary
from label_assignment import assign_label
from outlier_detections import gaussian_prob  

'''
filename: a csv file 
upper_only: only use distinct tuple pairs (above diagonal in affinity matrix) for Gaussian distribution 
'''
def main(filename, upper_only):
    X = affinity_matrix_binary(filename)
    num_rows, _, num_cols = np.shape(X)
    
    if upper_only:
        triu_idx = np.triu_indices(num_rows, 1)
        X = X[triu_idx]
    else:
        X = X.reshape((num_rows**2, num_cols))
    
    # now each row of X is a similarity vector of a tuple pair
    # describe X using a Gaussian
    mu = np.mean(X, axis=0)
    sigma = np.cov(X, rowvar=0)
    
    # get probability of containing no error, for every tuple pair
    P = gaussian_prob(X, mu, sigma)
    
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
    A = assign_label(P, upper_only, loss_func='method2', lr=0.01, max_iter=200)
    print("First 20 entries in A: ", A[:10])



if __name__ == '__main__':
    main("data/employee_err.csv", True)
    