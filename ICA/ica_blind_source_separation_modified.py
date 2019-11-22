# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 10:06:33 2019

@author: Bohui Zhang
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import loadmat

from sklearn.decomposition import FastICA, PCA

# #############################################################################
# load EEG data
print('Loading data...')
train_data_mat = loadmat('eeg_data.mat')
train_data = train_data_mat['EEG_data'][:30,:1000]
print('EEG_data loaded:', train_data.shape)

X = train_data.T  # Observations 

# Compute ICA
ica = FastICA(n_components=30)
S = ica.fit_transform(X)  # Reconstruct signals
A = ica.mixing_  # Get estimated mixing matrix

# We can `prove` that the ICA model applies by reverting the unmixing.
assert np.allclose(X, np.dot(S, A.T) + ica.mean_)

# For comparison, compute PCA
pca = PCA(n_components=30)
H = pca.fit_transform(X)  # Reconstruct signals based on orthogonal components

# #############################################################################
# Plot results

plt.figure(figsize=(20, 10))

models = [X, S, H, A]
names = ['Observations (mixed signal)',
         'True Sources',
         'ICA recovered signals', 
         'PCA recovered signals']
colors = ['blue', 'red', 'white', 'white']

for ii, (model, name) in enumerate(zip(models, names), 1):
    plt.subplot(4, 1, ii)
    plt.title(name)
    for sig, color in zip(model.T, colors):
        plt.plot(sig, color=color)

plt.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.46)
plt.savefig('ica.png')