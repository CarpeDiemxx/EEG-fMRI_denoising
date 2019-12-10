# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 2019

@author: Bohui Zhang
"""

import numpy as np
import matplotlib.pyplot as plt 
from math import sqrt

# import data
original = np.load('y.npy')
clean = np.load('v.npy') 

class ClassName(object):
	"""docstring for ClassName"""
	def __init__(self, arg):
		super(ClassName, self).__init__()
		self.arg = arg
		

'''
# Averaged root mean square error (%RMSE) 
# self-deriviate percent RMSE

from sklearn.metrics import mean_squared_error

def rmse_averaged(clean, original):
	# return sqrt(mean_squared_error(original, clean)) * 100  # the same result 
	return sqrt(((clean - original) ** 2).mean()) * 100

print('%RMSE_1:', rmse_averaged(clean, original))

# RMSE (S.S.Oh et al., 2014)
def rmse_compared(clean, original):
	return (sqrt(((clean - original) ** 2).sum()) / ((original ** 2).sum())) * 100

print('%RMSE_2:', rmse_compared(clean, original))
'''
# (INPS)

from scipy.fftpack import fft, ifft

def fft_plot(clean, origin): 
	clean_fft = fft(clean)
	# clean_fft_real = clean_fft.real 
	# clean_fft_imag = clean_fft.imag 
	clean_fft_abs = abs(clean_fft) 
	clean_fft_normalized = clean_fft_abs / (len(clean) / 2)

	original_fft = fft(original)
	# original_fft_real = original_fft.real 
	# original_fft_imag = original_fft.imag 
	original_fft_abs = abs(original_fft) 
	original_fft_normalized = original_fft_abs / (len(original) / 2)

	x_axis = np.linspace(0, 1, len(clean)) 
	
	# original waveform
	plt.subplot(211)
	plt.plot(x_axis, clean) 
	plt.plot(x_axis, original) 
	plt.title('Original waveform') 
	# FFT (normalized)
	plt.subplot(212)
	plt.plot(x_axis, clean_fft_normalized) 
	plt.plot(x_axis, original_fft_normalized) 
	plt.title('FFT of signal (normalization)')

	plt.show()

import matplotlib.mlab as mlab 

# Pxx, freqs = plt.psd(clean, NFFT=, Fs=250, window=mlab.window_none, 
# 					 noverlap=, pad_to=, scale_by_freq=True)
