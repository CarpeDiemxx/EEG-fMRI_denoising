# -*- coding: utf-8 -*-
"""
@author: Bohui Zhang
@e-mail: bhzhang97@gmail.com
"""

import numpy as np
import matplotlib.pyplot as plt 

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans 
from sklearn.linear_model import LogisticRegression
from mpl_toolkits.mplot3d import Axes3D

class dictionary_modification(object):
	def __init__(self, D, reduced=None, label=None):
		self.dictionary = D
		self.reduced = self.pca()
		self.label = self.k_means()

	def pca(self):
		""" PCA """
		D_reduced = PCA(n_components=3).fit_transform(self.dictionary)
		return D_reduced

	def k_means(self):
		""" k-means """
		D_kmeans = KMeans(n_clusters=3, random_state=0).fit(self.reduced)
		return D_kmeans.labels_
		
	def clustering(self):
		index_list = []
		#mode = int(np.sum(self.label[:5]) / 3)
		mode = np.random.randint(0, 3)
		for i in range(len(self.label)):
			if self.label[i] == mode:
				index_list.append(i)
		print(len(index_list))
		return index_list

	def plot(self):
		""" plot """
		fig = plt.figure(1, figsize=(8, 6))
		ax = Axes3D(fig, elev=-150, azim=110)
		ax.scatter(self.reduced[:, 0], self.reduced[:, 1], self.reduced[:, 2], 
				   c=self.label, cmap=plt.cm.Set1, edgecolor='k', s=50)
		plt.show()


if __name__ == "__main__":
	
	D = np.load('results\\dictionary.npy')
	mod_dict = dictionary_modification(D)
	print(mod_dict.clustering())
