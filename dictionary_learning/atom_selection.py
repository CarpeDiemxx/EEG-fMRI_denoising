from sklearn.decomposition import PCA
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

def correlation_coefficient(x, y):
	numerator = ((x-x.mean()) * (y-y.mean())).sum()
	var_x = np.sqrt(((x - x.mean()) ** 2).sum())
	var_y = np.sqrt(((y - y.mean()) ** 2).sum())
	return abs(numerator / (var_x * var_y))

def corr_coef(x, y):
	return x.corr(y)

A = np.array([1, 2, 3, 4, 5, 6, 7])
B = np.array([2, 4, 6, 8, 10, 12, 14])


D = np.load('dictionary.npy')
x = np.load('x.npy')
print(x.shape)
corr_list = np.zeros((D.shape[1], 1), dtype=float)
corr_list_avg = np.zeros((D.shape[1], 1), dtype=float)
for test_times in range(1, 501):
	start_point = random.randint(0, x.shape[0]-D.shape[1])
	end_point = start_point + D.shape[1]
	x_segs = x[start_point:end_point, 0]
	for i in range(0, D.shape[1]):
		corr_list[i, 0] = correlation_coefficient(x_segs, D[:, i])
	corr_list_avg += corr_list
	corr_list_avg = corr_list_avg / test_times

x_axis = np.linspace(1, 50, 50) 
plt.scatter(x_axis, corr_list_avg, marker = 'o', color = 'c', s = 50)
plt.plot(x_axis, np.zeros((D.shape[1], 1)), color = 'grey')
plt.show()

from matplotlib.collections import LineCollection
from matplotlib.ticker import MultipleLocator

def plot_eeg(data):
	# Plot the EEG
	n_samples = 50
	t = np.arange(n_samples)
	ticklocs = []
	ax2 = plt.figure('D').add_subplot(1, 1, 1)
	print(type(ax2))
	ax2.set_xlim(0, 50)
	ax2.set_xticks(np.arange(50))
	dmin = data.min()
	dmax = data.max()
	dr = (dmax - dmin) * 0.7  # Crowd them a bit.
	y0 = dmin
	n_rows = 50
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
	for i in range(1, 51):
		y_list.append('%d' % i)
	ax2.set_yticklabels(y_list)

	ax2.set_xlabel('Time (s)')


	plt.tight_layout()
	plt.show()

#plot_eeg(D)

#blablabla
