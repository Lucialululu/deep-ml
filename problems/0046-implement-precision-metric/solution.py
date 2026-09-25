import numpy as np
from sklearn.metrics import confusion_matrix

def precision(y_true, y_pred):
	if (y_pred==y_true).all():
		return 0
	else:
		conf_matrix = confusion_matrix(y_true, y_pred)
		TN, FP, FN, TP = conf_matrix.ravel()

		precision = TP / (TP + FP)

		return precision
	
