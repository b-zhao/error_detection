import numpy as np

def loss(A, P, N):
    p_ = 1 - 2 * P
    product = 0
    for i in range(N):
        for j in range(N):
            product = product + (p_[i][j] * A[i] * A[j])
    return product + np.sum(P)


def gradient(A, P, N):
    grad = np.zeros(N)
    p_ = 1 - 2 * P
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
Return:
    A: Nx1 array, A[i] = probability that tuple i has no error
'''
def assign_label(P, lr=0.01, max_epoch=200):
    N = np.shape(P)[0]
    A = np.random.random(N)

    l = loss(A, P, N)
    min_loss = l
    for t in range(max_epoch):
        grad = gradient(A, P, N)
        A = A - lr * grad
        A = mask(A)
        new_loss = loss(A, P, N)

        if abs(new_loss - l) < 0.01 and abs(new_loss - min_loss) < 0.01:
            break
        else:
            l = new_loss
        if new_loss < min_loss:
            min_loss = new_loss
    return A


# test: the second tuple contains an error
P = np.array([  [1, 0, 1, 1, 1],
                [0, 0, 0, 0, 0],
                [1, 0, 1, 1, 1],
                [1, 0, 1, 1, 1],
                [1, 0, 1, 1, 1]  ])
# P = np.random.random((1000, 1000))
A = assign_label(P)
print(A)

