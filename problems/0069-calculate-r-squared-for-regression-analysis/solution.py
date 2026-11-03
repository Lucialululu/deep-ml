
import numpy as np

def r_squared(y_true, y_pred):
	# Write your code here
	SSR = np.sum((y_true - y_pred)**2)
	SST = np.sum((y_pred - np.mean(y_true))**2)
	
	if SST == 0:
		return 0
	else:
		R2 = 1 - SSR/SST

	return R2
