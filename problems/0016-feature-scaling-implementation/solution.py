import numpy as np

def feature_scaling(data: np.ndarray) -> (np.ndarray, np.ndarray):
	# Your code here
	z = (data-np.mean(data, axis=0))/(np.std(data, axis=0))
	x_mark = (data-np.min(data, axis=0))/(np.max(data, axis=0)-np.min(data, axis=0))
	return z, x_mark
