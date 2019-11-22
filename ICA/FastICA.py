# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 21:53:56 2019

@author: Bohui Zhang
"""

import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
from scipy.io import loadmat

# load EEG data
print('Loading data...')
train_data_mat = loadmat('eeg_data.mat')
train_data = train_data_mat['EEG_data'][:,:]
print('EEG_data loaded: ', train_data.shape)

def FastICA(x):
    """
    X: signal (space)
    A: mixing matrix
    W: unmixing matrix
    A_estimate:
    """
    # centering
    # m -> dimension of data
    # n -> number of samples
    [m, n] = x.shape
    avg = np.mean(x, axis=1)
    for i in range(1, m):
        x[i, :] = x[i, :] - avg[i]
        
    # whitening / balling
    # cov_x -> the covariance matrix of x
    cov_x = np.cov(x)
    # eigen-value decomposition (EVD)
    eig_value, eig_vector = linalg.eig(cov_x) 
    whiten = np.diag(eig_value**(-1/2)).dot(eig_vector.T)
    x = whiten.dot(x)
    
    # iterating
    max_iter = 10000
    tolerance = 0.00001
    
    w = np.random.rand(m, m)
    for i in range(1, m):
        # initial weight vector: random
        w_plus = w[:, i].reshape(m, 1)
        count = 0
        last_w_plus = np.zeros((m, 1))
        w_plus = w_plus / linalg.norm(w_plus) 
        criterion_m = linalg.norm(w_plus - last_w_plus) > tolerance
        criterion_p = linalg.norm(w_plus + last_w_plus) > tolerance
        while criterion_m & criterion_p:
            # number of iterations
            count = count + 1
            # value of the last iteration 
            last_w_plus = np.copy(w_plus) 
            for j in range(1, m):
                # g_1(u) = tanh(a_1*u) which a_1 = 1
                # g_2(u) = u * exp(-u^2 / 2) 
                g = np.tanh(last_w_plus.T.dot(x)) 
                term_1 = np.mean(x[j,:] * g) 
                term_2 = np.mean(1 - g ** 2) * last_w_plus[j] 
                w_plus[j] = term_1 - term_2 
            w_projections = np.zeros(m)
            for k in range(1, i):
                w_projections = w_projections + (w_plus.T.dot(w[:,k])) * w[:,k] 
            w_plus = w_plus.reshape(1, m)
            w_plus = w_plus - w_projections
            w_plus = w_plus.reshape(m, 1)
            w_plus = w_plus / (linalg.norm(w_plus))
            if count == max_iter:
                print('Exit loop', linalg.norm(w_plus - last_w_plus, 1))
                break
        print('loop count:', count)
        w[:, i] = w_plus.reshape(m,)
    x = w.T.dot(x)
    
# test

x = train_data[:30, :1000]
z = FastICA(x)
print(z.shape)
plt.figure(figsize=(5,20))

plt.subplot(2,1,1)
plt.plot(z[1,:])
plt.subplot(2,1,2)
plt.plot(z[2,:])