
import numpy as np

def rmse(y_true, y_pred):
	# Write your code here
	n = np.size(y_true)
	rmse_res = np.sqrt(1/n * np.sum(np.abs(y_true-y_pred)**2))
	return round(rmse_res,3)
