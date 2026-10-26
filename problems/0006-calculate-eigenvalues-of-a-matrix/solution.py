def calculate_eigenvalues(matrix: list[list[float|int]]) -> list[float]:
	import numpy as np
	trace_A = np.trace(matrix)
	det_A = np.linalg.det(matrix)
	coefficients = [1, -trace_A, det_A]
	eigenvals = np.roots(coefficients)
	return eigenvals