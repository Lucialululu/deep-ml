def relu(z: float) -> float:
	# Your code here
	import numpy as np
	if z >= 0:
		relu_z = np.maximum(0,z)
	else:
		relu_z = 0
	return relu_z
