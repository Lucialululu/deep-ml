def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
	import numpy as np
	# Return a list where each element is the dot product of a row of 'a' with 'b'.
	# If the number of columns in 'a' does not match the length of 'b', return -1.
	col_elms_a = np.size(a, 1)
	length_b = np.size(b)
	if col_elms_a == length_b:
		return np.dot(a, b)
	else:
		return -1
	