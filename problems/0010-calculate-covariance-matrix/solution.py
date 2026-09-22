def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
	import numpy as np
	V = np.asarray(vectors)
	means = np.mean(V, axis = 1)
	mean_matrix = np.reshape((np.repeat(means, len(V[0]))), np.shape(V))
	diffs = V - mean_matrix
	covs = (diffs @ diffs.T)/(len(V[0])-1)
	return covs
	

		
		
	# Your code here
	# return []