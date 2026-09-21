import numpy as np

def reshape_matrix(a: list[list[int|float]], new_shape: tuple[int, int]) -> list[list[int|float]]:
	#Write your code here and return a python list after reshaping by using numpy's tolist() method

	# Compare the product rows * cols of the new_shape with the total number of elements in the original matrix a. You can get the total count using np.asarray(a).size. If they differ, return [] early.
	arr = (np.asarray(a)).flatten()
	rows = new_shape[0]
	cols = new_shape[1]

	# validation check
	tot_elements = (np.asarray(a)).size
	if tot_elements != rows * cols:
		return []

	reshaped_matrix = np.zeros(new_shape)
	for i in range(rows):
		reshaped_matrix[i, :] = arr[i*cols:(i+1)*cols]

	return reshaped_matrix.tolist()