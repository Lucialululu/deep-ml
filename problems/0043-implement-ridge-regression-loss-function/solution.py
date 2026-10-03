import numpy as np

def ridge_loss(X: np.ndarray, w: np.ndarray, y_true: np.ndarray, alpha: float) -> float:
	# Your code here
	n = np.shape(X)[0]
	y_pred = np.dot(X, w)

	MSE = 1/n * np.sum((y_pred - y_true)**2)
	pen = alpha * np.sum(w**2)

	return MSE + pen
