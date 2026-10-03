import numpy as np

def accuracy_score(y_true, y_pred):
	n = len(y_true)
	accuracy = np.sum(y_true == y_pred)/n
	return accuracy
			
			