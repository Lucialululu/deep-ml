
from collections import Counter
import numpy as np

def confusion_matrix(data):
	# Implement the function here
	data = np.array(data)
	labels = data[:, 0]
	preds = data[:, 1]

	tp = np.sum((preds == 1) & (labels == 1))
	fp = np.sum((preds == 1) & (labels == 0))
	fn = np.sum((preds == 0) & (labels == 1))
	tn = np.sum((preds == 0) & (labels == 0))

	matrix = [[tp, fn], [fp, tn]]

	return matrix
