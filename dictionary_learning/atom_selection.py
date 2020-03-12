import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import math

D = np.load('results\\dictionary.npy')
x = np.load('results\\x.npy')

def correlation_coefficient(x, y):
	numerator = ((x-x.mean()) * (y-y.mean())).sum()
	var_x = np.sqrt(((x - x.mean()) ** 2).sum())
	var_y = np.sqrt(((y - y.mean()) ** 2).sum())
	return abs(numerator / (var_x * var_y))

def correlation_coefficient_avg(D, x):
	corr_list = np.zeros((D.shape[1], 1), dtype=float)
	corr_list_avg = np.zeros((D.shape[1], 1), dtype=float)
	for test_times in range(1, 5):
		start_point = random.randint(0, x.shape[0]-D.shape[1])
		end_point = start_point + D.shape[1]
		x_segs = x[start_point:end_point, ]
		for i in range(0, D.shape[1]):
			corr_list[i, 0] = correlation_coefficient(x_segs, D[:, i])
		corr_list_avg += corr_list
		corr_list_avg = corr_list_avg / test_times

	x_axis = np.linspace(1, D.shape[1], D.shape[1]) 
	plt.scatter(x_axis, corr_list_avg, marker = 'o', color = 'c', s = 50)
	plt.plot(x_axis, np.zeros((D.shape[1], 1)), color = 'grey')
	plt.show()

#correlation_coefficient_avg(D, D[:, 50])

from matplotlib.collections import LineCollection
from matplotlib.ticker import MultipleLocator

def plot_dictionary(data):
	# Plot the EEG
	n_components = data.shape[0]
	t = np.arange(n_components)
	ticklocs = []
	ax2 = plt.figure('D').add_subplot(1, 1, 1)
	print(type(ax2))
	ax2.set_xlim(0, n_components)
	ax2.set_xticks(np.arange(n_components))
	dmin = data.min()
	dmax = data.max()
	dr = (dmax - dmin) * 0.7  # Crowd them a bit.
	y0 = dmin
	n_rows = n_components
	y1 = (n_rows - 1) * dr + dmax
	ax2.set_ylim(y0, y1)
	segs = []
	for i in range(n_rows):
	    segs.append(np.column_stack((t, data[:, i])))
	    ticklocs.append(i * dr)
	offsets = np.zeros((n_rows, 2), dtype=float)
	offsets[:, 1] = ticklocs
	lines = LineCollection(segs, offsets=offsets, transOffset=None)
	ax2.add_collection(lines)
	# Set the yticks to use axes coordinates on the y axis
	ax2.set_yticks(ticklocs)
	y_list = []
	for i in range(1, n_components+1):
		y_list.append('%d' % i)
	ax2.set_yticklabels(y_list)
	ax2.set_xlabel('Time (s)')
	plt.tight_layout()
	plt.show()

#plot_dictionary(D)

# clustering

def dictionary_pca(D):
	mean = np.mean(D, axis=0)
	X = D - mean
	X = np.dot(X.T, X)
	U, S, V = np.linalg.svd(X)

	S = S / S.sum()
	xx = np.arange(1, 51)
	plt.figure(figsize=(8, 6), dpi=200)
	plt.xlim(1, 50)
	#plt.ylim(0.00, 0.50)
	plt.xlabel("l", fontsize=18, color='black', horizontalalignment='center', fontname='Times New Roman')
	plt.ylabel(r'$\bar {\lambda}_{l}$', fontsize=18, color='black', horizontalalignment='center',
		 	   fontname='Times New Roman')  
	plt.plot(xx, S[xx-1], linestyle="-", marker="o", color="b", linewidth=2)
	plt.show()

#dictionary_pca(D)

def mean_clustering(D):
	"""
	Based on mean.
	"""
	atom_number = D.shape[0]
	# axis = 0 -> add up rows
	atoms_mean_list = np.mean(D, 0)
	print(atoms_mean_list)
	plt.figure(figsize=(7, 5))
	x_axis = np.linspace(1, atom_number, atom_number) 
	plt.scatter(x_axis, atoms_mean_list)
	plt.show()

def var_clustering(D):
	"""
	Based on std var, log2(std). 
	"""
	atoms_number = D.shape[1]
	std_list = list(map(lambda x: np.std(x, ddof=1), (D[:, i] for i in range(atoms_number))))
	std_log_list = list(map(lambda x: -1 * math.log2(x), std_list))
	print(std_log_list)
	plt.figure(figsize=(7, 5))
	x_axis = np.linspace(1, atoms_number, atoms_number) 
	plt.scatter(x_axis, std_log_list, color='b')
	plt.show()

#var_clustering(D)


