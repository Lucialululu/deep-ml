def scalar_multiply(matrix: list[list[int|float]], scalar: int|float) -> list[list[int|float]]:
	import numpy as np
	M = np.asarray(matrix)
	return M * scalar
	