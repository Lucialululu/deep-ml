def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
	import numpy as np
	matrix = np.asarray(matrix)
	means = []
	if mode == 'column':
		for i in range(len(matrix[0])):
			col_mean = np.mean([element[i] for element in matrix])
			means.append(col_mean)
	else:
		for i in range(len(matrix)):
			row_mean= np.mean(matrix[i])
			means.append(row_mean)
	return means


	