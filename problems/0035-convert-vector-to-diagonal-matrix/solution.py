import numpy as np

def make_diagonal(x):
	# Your code here
	empty_matrix = np.zeros((len(x), len(x)))
	#return  empty_matrix
	for i in range(len(x)):
		empty_matrix[i][i] = x[i]
		
	return empty_matrix

	


	# return 