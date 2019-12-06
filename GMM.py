# GMM class is modified from CS 7641 Fall 2019 HW2

import numpy as np
from tqdm import tqdm_notebook as tqdm
from get_features import *


def softmax(logits):
    """
    Args:
        logits: N x D numpy array
    """
    logits_exp = np.exp((logits.T - logits.max(axis=1)).T)
    logits_exp_sum = np.sum(logits_exp, axis=1)
    return (logits_exp.T / logits_exp_sum).T

def logsumexp(logits):
    """
    Args:
        logits: N x D numpy array
    Return:
        s: N x 1 array where s[i,0] = logsumexp(logits[i,:])
    """
    exp_ = np.exp((logits.T - logits.max(axis=1)).T)
    log_ = np.log(np.sum(exp_, axis=1))
    return (log_.T + logits.max(axis=1)).T.reshape(np.shape(logits)[0], 1)

class GMM(object):
    def __init__(self): # No need to implement
        pass
        
    def _init_components(self, points, K, **kwargs):
        """
        Args:
            points: NxD numpy array, the observations
            K: number of components
            kwargs: any other args you want
        Return:
            pi: numpy array of length K, prior
            mu: KxD numpy array, the center for each gaussian. 
            sigma: KxDxD numpy array, the covariance matrix of each gaussian.
        """
        mu = np.average(points, axis=0)
        
        pi = np.empty(K)
        pi.fill(1.0/K)
        noise = np.random.random(K)
        pi = pi + (noise-np.average(noise))/1000000000
        
        D = np.shape(points)[1] 
        sigma = np.random.random((K, D, D)) * 2 - 1
        for k in range(K):
            sigma[k] = np.dot(sigma[k], sigma[k].T)
        return pi, mu, sigma
        

    def _ll_joint(self, points, pi, mu, sigma, **kwargs):
        """
        Args:
            points: NxD numpy array, the observations
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian. 
            sigma: KxDxD numpy array, the covariance matrix of each gaussian.
        Return:
            ll(log-likelihood): NxK array
        """
        N, D = np.shape(points)
        K = np.shape(pi)[0]
        ll = np.zeros((N, K))

        for k in range(K):
            temp = np.matmul((points - mu[k]), np.linalg.pinv(sigma[k]))
            ll[:,k] = -0.5 * np.einsum('ij,ij->i', temp, (points - mu[k]))
            ll[:,k] += np.log( pi[k] / ((2*np.pi)**(D/2)*np.sqrt(np.linalg.det(sigma[k]) + 1e-18)))
        return ll
    
    def _E_step(self, points, pi, mu, sigma, **kwargs):
        """
        Args:
            points: NxD numpy array, the observations
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian. 
            sigma: KxDxD numpy array, the covariance matrix of each gaussian.
        Return:
            gamma: NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
        """
        ll = self._ll_joint(points, pi, mu, sigma)
        gamma = softmax(ll)
        return gamma
        

    def _M_step(self, points, gamma, **kwargs):
        """
        Args:
            points: NxD numpy array, the observations
            gamma: NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
        Return:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian. 
            sigma: KxDxD numpy array, the covariance matrix of each gaussian.
        """
        # raise NotImplementedError
        N, D = np.shape(points)
        K = np.shape(gamma)[1]
        N_k = np.sum(gamma, axis=0)
        
        pi = N_k / N
        mu = (np.matmul(gamma.T, points).T / N_k).T
        sigma = np.zeros((K, D, D))
        for k in range(K):
            temp = np.matmul((points - mu[k]).reshape((N, D, 1)), (points - mu[k]).reshape((N, 1, D)))
            sigma[k] = np.dot(np.transpose(temp, (1, 2, 0)), gamma[:,k])
            sigma[k] /= N_k[k]
        return pi, mu, sigma

    def __call__(self, points, K, max_iters=100, abs_tol=1e-16, rel_tol=1e-16, **kwargs):
        """
        Args:
            points: NxD numpy array, where N is # points and D is the dimensionality
            K: number of clusters
            max_iters: maximum number of iterations
            abs_tol: convergence criteria w.r.t absolute change of loss
            rel_tol: convergence criteria w.r.t relative change of loss
            kwargs: any additional arguments you want
        Return:
            gamma: NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
            (pi, mu, sigma): (1xK np array, KxD numpy array, KxD numpy array), mu and sigma.
        """        
        pi, mu, sigma = self._init_components(points, K, **kwargs)
        pbar = tqdm(range(max_iters))
        for it in pbar:
            gamma = self._E_step(points, pi, mu, sigma)
            pi, mu, sigma = self._M_step(points, gamma)
            
            # calculate the negative log-likelihood of observation
            joint_ll = self._ll_joint(points, pi, mu, sigma)
            loss = -np.sum(logsumexp(joint_ll))
            if it:
                diff = np.abs(prev_loss - loss)
                if diff < abs_tol and diff / prev_loss < rel_tol:
                    break
            prev_loss = loss
            pbar.set_description('iter %d, loss: %.4f' % (it, loss))
        return gamma, (pi, mu, sigma)

    
# X = get_feature_vecs('data/hospital.csv')
# np.save("data/hospital_features.npy", X)

# X = np.load("data/hospital_features.npy")
# X_normalized = np.empty_like(X)

# normalize each column
# for i in range(np.shape(X)[1]):
#     r = (np.amax(X[:,i]) - np.amin(X[:,i]))
#     if r != 0:
#         X_normalized[:,i] = (X[:,i] - np.amin(X[:,i])) / r
#     else:
#         X_normalized[:,i] = (X[:,i] - np.amin(X[:,i]))
        
# gamma, (pi, mu, sigma) = GMM()(X_normalized, K=1, max_iters=100)
# print(gamma)
# print(pi)
# print(mu)
# print(sigma)

    