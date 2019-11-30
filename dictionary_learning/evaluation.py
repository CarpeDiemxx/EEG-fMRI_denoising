import numpy as np
import matplotlib.pyplot as plt 
from math import sqrt

# import data
original = np.load('y.npy')
clean = np.load('v.npy') 

# Averaged root mean square error (RMSE) 
# self-deriviate percent RMSE
def rmse_averaged(clean, original):
	return sqrt(((clean - original) ** 2).mean()) * 100

print('formula_1:', rmse_averaged(clean, original))

# imported percent RMSE
from sklearn.metrics import mean_squared_error

def rmse_imported(clean, original):
	return sqrt(mean_squared_error(original, clean)) * 100

# print('result_2:', rmse_imported(clean, original))

# RMSE (S.S.Oh et al., 2014)
def rmse_compared(clean, original):
	return (sqrt(((clean - original) ** 2).sum()) / ((original ** 2).sum())) * 100

print('formula_2:', rmse_compared(clean, original))