import pandas as pd
import numpy as np
from get_features import *
import time
import py_entitymatching as em

'''
Return affinity matrix A. A[i][j][f] = (t_i[f] == t_j[f])
'''
def affinity_matrix_binary(filename):
    df = pd.read_csv(filename)
    df, _, _ = remove_na(df)
    X = np.array(df)

    num_rows, num_cols = np.shape(X)
    affinity_matrix = np.zeros((num_rows, num_rows, num_cols))
    
    for i in range(int(num_rows)):
        row_i_tile = np.tile(np.array(X[i]), (num_rows, 1))
        affinity_matrix[i] = np.equal(X, row_i_tile)

    return affinity_matrix

'''
Return affinity matrix A. A[i][j][f] = abs(t_i[f] - t_j[f]) if numerical, 
lev_dist(t_i[f], t_j[f]) otherwise
'''
def affinity_matrix_similarity(filename):
    df = pd.read_csv(filename)
    df, _, _ = remove_na(df)
    X = np.array(df)

    num_rows, num_cols = np.shape(X)
    affinity_matrix = np.zeros((num_rows, num_rows, num_cols))
    
    for i in range(int(num_rows)):
        row_i_tile = np.tile(np.array(X[i]), (num_rows, 1))
        for j in range(int(num_rows)):
            for k in range(int(num_cols)):
                if np.issubdtype(type(X[i][k]), np.number): 
                    affinity_matrix[i][j][k] = abs(X[i][k] - X[j][k])
                else:
                    affinity_matrix[i][j][k] = em.lev_dist(X[j][k], row_i_tile[j][k])
    return affinity_matrix

# start_time = time.time()
# affinity_matrix = affinity_matrix_binary("data/hospital.csv")
# print("Affinity matrix shape: ", np.shape(affinity_matrix))
# print("runtime: %s seconds" % (time.time() - start_time))
# print(affinity_matrix[2][2])
# print(affinity_matrix[2][1])
