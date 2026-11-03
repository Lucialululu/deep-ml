import numpy as np

def f_score(y_true, y_pred, beta):
	"""
	Calculate F-Score for a binary classification task.

	:param y_true: Numpy array of true labels
	:param y_pred: Numpy array of predicted labels
	:param beta: The weight of precision in the harmonic mean
	:return: F-Score rounded to three decimal places
	"""
	TP = np.sum(np.logical_and(y_pred == 1, y_true == 1))
	FP = np.sum(np.logical_and(y_pred == 1, y_true == 0))
	TN = np.sum(np.logical_and(y_pred == 0, y_true == 0))
	FN = np.sum(np.logical_and(y_pred == 0, y_true == 1))

	precision = TP/(TP+FP)
	recall = TP/(TP+FN)

	F_score = (1+beta**2) * (precision * recall)/((beta**2 * precision) + recall)

	return F_score.round(3)

	
