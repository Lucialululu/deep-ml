import numpy as np

def to_categorical(x, n_col=None):
	# Your code here
	if n_col == None:
		n_col = np.max(x) + 1
	
	one_hot = np.zeros((len(x), n_col))
	row_indices = np.arange(len(x))
	one_hot[row_indices, x] = 1

	return one_hot