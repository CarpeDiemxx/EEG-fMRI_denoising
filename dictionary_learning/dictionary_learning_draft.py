# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 2019

@author: Bohui Zhang
"""
import random
import numpy as np
import pandas as pd
from scipy.io import loadmat

from dictionary_modification import dictionary_modification as dictmod 

# global variables
eeg_data_length = 1200 
segment_length = 50
segment_number = 50
n_components = 50

# load EEG data
# train_data.mat -> eeg -> (32, 332416)
# eeg_data.mat -> EEG_data -> (32, 332416)
data_set = 2
print('Loading data...')
if data_set == 1:
	train_data_mat = loadmat('train_data.mat')  # FMRIB data
	train_data = train_data_mat['eeg'].sum(axis=0).reshape(1, 332416)
	train_data = train_data[0, 1000:1000+eeg_data_length]
elif data_set == 2:
	train_data_mat = loadmat('eeg_data.mat')  # Leixu data
	train_data = train_data_mat['eeg'].sum(axis=0).reshape(1, 332416)
	train_data = train_data[0, 2000:2000+eeg_data_length]
else:
	train_data_mat = loadmat('SimuEEG.mat')  # Synthetic data
	train_data = train_data_mat['x'][0, :eeg_data_length]

def seg_extract(x:'original EEG data', m:'segment number', n:'segment length'):
	"""
	Segment the original signal x (x.shape = (eeg_data_length, 1)) into m smaller signal of length n.
	Note that these m segments can have any percentage **overlap** with each other.
	
	Returns:
		x_seg: segmented eeg data (x_seg.shape = (n, m))
		R[i]: an n x N binary (0,1) matrix that extracts the i-th segment from x
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
		x_seg[:, i] = (R[i].dot(x.T)).reshape(segment_length)
		# move back the index randomly to implement overlapping
		t = random.randint(0, n)  
		#t = random.randint(0, n*27/50)
		j = j + n - t
	return x_seg, R

def dict_initialize(x_seg:'segmented EEG data', n_components:'dimension of dictionary'):
	"""
	Initialize the dictionary by taking the first n_components column vectors of 
	left sigular matrix of original EEG data as atoms of the initial dictionary.

	Returns:
		dict_data: initialized dictionary (dict_data.shape = (n, n_components))
	"""
	u, sigma, v = np.linalg.svd(x_seg)
	dict_data = u[:, :n_components]  # dict_data.shape = (50, 50) ! 
	#dict_data = np.random.rand(x_seg.shape[0], n_components)
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
		'''
		for j, k in enumerate(index):
			s[i, k] = sigma[0] * v[0, j]
		'''
		s[i, index] = sigma[0] * v[0, :]
	return D, s

def x_estimate(y:'original EEG data', D, s, m, n, const_lambda=0.01):   

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

def atom_rearrange(D, index_list):
	"""
	index_list should be a list
	"""
	copy_D = D
	for i in index_list:
		copy_D[:, i] = np.zeros((1, D.shape[0]))
	return copy_D

from sklearn import linear_model

y = train_data.reshape(1, eeg_data_length).T
print('Extracting segments...')
x_seg, R = seg_extract(train_data, m=segment_number, n=segment_length)

dictionary = dict_initialize(x_seg, n_components)
#print('dict shape', dictionary.shape)

max_iter = 10
tolerance = 0.20

print('Learning dictionary...')
for i in range(max_iter):
	# sparse coding stage
	s = linear_model.orthogonal_mp(dictionary, x_seg)
	e = np.linalg.norm(x_seg - np.dot(dictionary, s))
	if e < tolerance:
		break
	# dictionary update stage
	dict_update(x_seg, dictionary, s, n_components)

print('Estimating BCG...')
x = x_estimate(y, dictionary, s, m=segment_number, n=segment_length)

dict_prime = dictmod(dictionary)
index_list = dict_prime.clustering()
x_prime = x_estimate(y, atom_rearrange(dictionary, index_list), s, m=segment_number, n=segment_length)

print('Computing clean EEG...')
v = y - x
v_prime = y - x_prime

# save data
import scipy.io as scio

scio.savemat('results\\y.mat', {'y':y})
scio.savemat('results\\v.mat', {'v':v})

np.save('results\\y.npy', y)
np.save('results\\v.npy', v)

np.save('results\\x.npy', x)
np.save('results\\dictionary.npy', dictionary) 

np.savetxt('results\\s.txt', s)

import matplotlib.pyplot as plt 

def eeg_plot(y:'original signal', v:'clean EEG', x:'BCG', y_label):
	"""
	Plot original EEG data and clean EEG data.
	"""
	t = np.linspace(0, 20, eeg_data_length)
	plt.figure(figsize=(5, 20))
	#plt.ylim(-1.2, 1.5)
	#plt.ylim(-10000, 12000)
	plt.plot(t, y, color='b', label=y_label)
	plt.plot(t, v, color='r', label='Cleaned EEG (before)')
	plt.plot(t, x, color='y', label='Cleaned EEG (after)')
	plt.legend(loc='0')
	plt.show()

print('Plotting...')
real_EEG = loadmat('SimuEEG.mat')['n'][0, :eeg_data_length].reshape(eeg_data_length, 1)
#print(max(abs(real_EEG-v)))
#print(min(abs(real_EEG-v)))
#print(abs(real_EEG-v).mean())

if data_set == 3:
	eeg_plot(real_EEG, v, v_prime, "Real EEG")
else:
	eeg_plot(y, v, v_prime, "Original Signal")

