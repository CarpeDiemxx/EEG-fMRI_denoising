<<<<<<< HEAD
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 2019

@author: Bohui Zhang
=======
"""
>>>>>>> 2a7585c176806692418b800359f60238ba7dd104
"""

import random
import numpy as np
import pandas as pd
from scipy.io import loadmat

# global variables
<<<<<<< HEAD
eeg_data_length = 1200
segment_length = 50
segment_number = 100 
=======
eeg_data_length = 10000
segment_length = 50
segment_number = 2000 
>>>>>>> 2a7585c176806692418b800359f60238ba7dd104

# load EEG data
# train_data.mat -> x -> (1, 1200)
# eeg_data.mat -> EEG_data -> (32, 332416)
<<<<<<< HEAD
data_set = 3
=======
data_set = 2
>>>>>>> 2a7585c176806692418b800359f60238ba7dd104
print('Loading data...')
if data_set == 1:
	train_data_mat = loadmat('train_data.mat')
	# print(train_data_mat.keys())
	train_data = train_data_mat['x'][0, :eeg_data_length]
	# print(train_data, train_data.shape)
<<<<<<< HEAD
elif data_set == 2:
	train_data_mat = loadmat('eeg_data.mat')
	train_data = train_data_mat['EEG_data'].sum(axis=0).reshape(1, 332416)
	train_data = train_data[0, 20000:20000+eeg_data_length]
else:
	train_data_mat = loadmat('SimuEEG.mat') 
	train_data = train_data_mat['x'][0, :eeg_data_length]

=======
else:
	train_data_mat = loadmat('eeg_data.mat')
	train_data = train_data_mat['EEG_data'].sum(axis=0).reshape(1, 332416)
	train_data = train_data[0, 20000:20000+eeg_data_length]
>>>>>>> 2a7585c176806692418b800359f60238ba7dd104

def seg_extract(x:'original EEG data', m:'segment number', n:'segment length'):
	"""
	Segment the original signal x (x.shape = (eeg_data_length, 1)) into m smaller signal of length n.
	Note that these m segments can have any percentage **overlap** with each other.
	
	Returns:
		x_seg: segmented eeg data (x_seg.shape = (n, m))
		R[i]: an nxN binary (0,1) matrix that extracts the i-th segment from x
	"""
	N = x.size
	# j -> the index of the start of the i-th segment
	j = 0	
	x_seg = np.zeros((n, m))
	R = np.zeros((m, n, N))
	for i in range(0, m):
		if j > N - n:
			for k in range(0, n):
				R[i][k][N-n+k] = 1
		else:
			for k in range(0, n):
				R[i][k][j+k] = 1
		x_seg[:, i] = (R[i].dot(x.T)).reshape(50)
		# move back the index randomly to implement overlapping
<<<<<<< HEAD
		t = random.randint(n*2/5, n)
		j = j + n - t
	return x_seg, R

def dict_initialize(x_seg:'segmented EEG data', n_components:'dimension of dictionary'):
	"""
	Initialize the dictionary by taking the first n_components column vectors of 
	left sigular matrix of original EEG data as atoms of the initial dictionary.
=======
		t = random.randint(n*1/5, n)
		j = j + n - t
	return x_seg, R

def dict_initialize(x_seg:'segmented EEG data', n_components:'dimension of dictionary'=50):
	"""
	Initialize the dictionary by taking the first n_components column vectors of 
	left sigular matrix of original EEG data as atoms of the initial dictionary.

>>>>>>> 2a7585c176806692418b800359f60238ba7dd104
	Returns:
		dict_data: initialized dictionary (dict_data.shape = (n, n_components))
	"""
	u, sigma, v = np.linalg.svd(x_seg)
<<<<<<< HEAD
	print('x_seg.shape =', x_seg.shape)
	print('u.shape =', u.shape) 
	dict_data = u[:, :n_components]  # dict_data.shape = (50, 50) ! 
	print('dict_shape =', dict_data.shape)
=======
	dict_data = u[:, :n_components]
>>>>>>> 2a7585c176806692418b800359f60238ba7dd104
	return dict_data

def dict_update(x:'data', D:'dictionary', s, n_components):
	"""
	Apply K-SVD to find s and D.
	Update the dictionary column by column.
	"""
	for i in range(n_components):
		index = np.nonzero(s[i, :])[0]
		if len(index) == 0:
			continue
		# update the i-th column of dictionary
		D[:, i] = 0
		# compute the error matrix
		err = (x - np.dot(D, s))[:, index]
		# svd -> u: dictionary; s*v: sparse matrix
		u, sigma, v = np.linalg.svd(err, full_matrices=False)
		# 0-th column of left singular matrix -> new i-th column of the dictionary
		D[:, i] = u[:, 0]
		# 0-th singular value * 0-th row of right singular matrix -> sparse coeff matrix
		for j, k in enumerate(index):
			s[i, k] = sigma[0] * v[0, j]
	return D, s

<<<<<<< HEAD
def x_estimate(y:'original EEG data', D, s, m, n, const_lambda=0.01):
=======
def x_estimate(y:'original EEG data', D, s, m, n, const_lambda=0.5):
>>>>>>> 2a7585c176806692418b800359f60238ba7dd104
	"""
	Estimate x (BCG artifact) using Eq.(5) in (Abolghasemi et al., 2015).
	"""
	N = y.size
	first_term = const_lambda * np.identity(N)
	for i in range(0, m):
		first_term += (R[i].T).dot(R[i])
	first_term = np.mat(first_term).I
	second_term = const_lambda * y
	for i in range(0, m):
		second_term += ((R[i].T).dot(D).dot(s[:, i])).reshape(N, 1)
	return (first_term * second_term)

from sklearn import linear_model

y = train_data.reshape(1, eeg_data_length).T
print('Extracting segments...')
x_seg, R = seg_extract(train_data, m=segment_number, n=segment_length)
<<<<<<< HEAD
dictionary = dict_initialize(x_seg, segment_number)
=======
dictionary = dict_initialize(x_seg)
>>>>>>> 2a7585c176806692418b800359f60238ba7dd104
max_iter = 5
tolerance = 0.15

print('Learning dictionary...')
for i in range(max_iter):
	# sparse coding stage
	s = linear_model.orthogonal_mp(dictionary, x_seg)
	e = np.linalg.norm(x_seg - np.dot(dictionary, s))
	if e < tolerance:
		break
	# dictionary update stage
	dict_update(x_seg, dictionary, s, n_components=50)

print('Estimating BCG...')
x = x_estimate(y, dictionary, s, m=segment_number, n=segment_length)

print('Computing clean EEG...')
v = y - x

# save data
import scipy.io as scio

scio.savemat('y.mat', {'y':y})
scio.savemat('v.mat', {'v':v})

np.save('y.npy', y)
np.save('v.npy', v)
<<<<<<< HEAD
np.save('x.npy', x)
np.save('dictionary.npy', dictionary) 
=======
>>>>>>> 2a7585c176806692418b800359f60238ba7dd104

import matplotlib.pyplot as plt 

def eeg_plot(y:'original signal', v:'clean EEG', x:'BCG'):
	"""
	Plot original EEG data and clean EEG data.
	"""
	t = np.linspace(0, 20, eeg_data_length)
	plt.figure(figsize=(5, 20))
	# plt.ylim(-1.5, 1.5)
	plt.plot(t, y, color='b')
<<<<<<< HEAD
	#plt.plot(t, v, color='r')
	plt.plot(t, x, color='r')
	plt.show()

print('Plotting...')
#eeg_plot(y, v, x)

real_EEG = loadmat('SimuEEG.mat')['n'][0, :eeg_data_length].reshape(eeg_data_length, 1)
print(max(abs(real_EEG-v)))
print(min(abs(real_EEG-v)))
print(abs(real_EEG-v).mean())
print(dictionary.shape)


print('s.shape =', s.shape)


#eeg_plot(real_EEG, v, x)
eeg_plot(y, v, x)
=======
	plt.plot(t, v, color='r')
	# plt.plot(t, x, color='c')
	plt.show()

print('Plotting...')
eeg_plot(y, v, x)
>>>>>>> 2a7585c176806692418b800359f60238ba7dd104
