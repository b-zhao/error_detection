import numpy as np

# loss function on the report
def loss1(A, P, N, upper_only):
    loss = (np.linalg.norm(np.outer(A, A) - P))**2 - np.inner(A, A)
    return loss

# gradients on the report
def gradient1(A, P, N, upper_only):
    grad = np.zeros(N)
    for i in range(N):
        for j in range(N):
            grad[i] = grad[i] + (A[i] * A[j] - P[i][j]) * A[j]
        grad[i] = grad[i] - (A[i] * A[i] - P[i][i]) * A[i]
    return grad


# alternative loss function (see label_assignment doc on Overleaf)
def loss2(A, P, N, upper_only):
    if not upper_only:
        p_ = 1 - 2 * P
        product = 0
        for i in range(N):
            for j in range(i, N):
                product = product + (p_[i][j] * A[i] * A[j])
        return product + np.sum(P)
    else:
        p_ = 1 - 2 * P
        np.fill_diagonal(p_, 0)
        product = 0
        for i in range(N):
            for j in range(i+1, N):
                product = product + (p_[i][j] * A[i] * A[j])
        return product + np.sum(P) / 2

# gradients for alternative loss function
def gradient2(A, P, N, upper_only):
    if not upper_only:
        grad = np.zeros(N)
        p_ = 1 - 2 * P
        grad = np.dot(p_, A)
        for i in range(N):
            grad[i] = np.dot(p_[i], A)
        return grad
    else:
        grad = np.zeros(N)
        p_ = 1 - 2 * P
        np.fill_diagonal(p_, 0)
        i_lower = np.tril_indices(np.shape(P)[0], -1)
        p_[i_lower] = p_.T[i_lower]
        # print(np.allclose(p_.T, p_))
        
        grad = np.dot(p_, A)
        for i in range(N):
            grad[i] = np.dot(p_[i], A)
        return grad 

def mask(A):
    A[A>1] = 1
    A[A<0] = 0
    return A

'''
Args:
    P: NxN numpy array, where N is # tuples in the original dataset. P[i][j] = probability 
       that both tuple i and tuple j have no error
    upper_only: if true, the lower triangle and diagonal of P is all zero
    loss_func: one of 'method1' and 'method2', the loss function and corresponding gradient
Return:
    A: Nx1 array, A[i] = probability that tuple i has no error
'''
def assign_label(P, upper_only=True, loss_func='method1', lr=0.01, max_iter=200):
    # initialize A
    N = np.shape(P)[0]
    A = np.random.random(N)
    
    # make P.T = P (fill lower triangle)
    i_lower = np.tril_indices(np.shape(P)[0], -1)
    P[i_lower] = P.T[i_lower]
    # print(np.allclose(P.T, P))
    
    # gradient descent
    l = loss1(A, P, N, upper_only) if loss_func == 'method1' else loss2(A, P, N, upper_only)
    min_loss = l
    for t in range(max_iter):
        grad = gradient1(A, P, N, upper_only) if loss_func == 'method1' else gradient2(A, P, N, upper_only)
        A = A - lr * grad
        A = mask(A)        
        new_loss = loss1(A, P, N, upper_only) if loss_func == 'method1' else loss2(A, P, N, upper_only)

        # if converged, stop
        # if abs(new_loss - l) < 0.01 and abs(new_loss - min_loss) < 0.01:
        if np.linalg.norm(grad) / np.shape(A)[0] < 1e-8:
            break
            
        if new_loss < min_loss:
            min_loss = new_loss
        l = new_loss
        
        if t == max_iter - 1:
            print("Max iteration reached: label assignment did not converge. ")

    return A


# test: the second tuple contains an error
# P = np.array([  [1, 0, 1, 1, 1],
#                 [0, 0, 0, 0, 0],
#                 [1, 0, 1, 1, 1],
#                 [1, 0, 1, 1, 1],
#                 [1, 0, 1, 1, 1]  ])
# # P = np.random.random((1000, 1000))
# A = assign_label(P, upper_only=True, loss_func='method2')
# print(A)

