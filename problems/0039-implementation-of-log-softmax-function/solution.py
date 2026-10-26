import numpy as np

def log_softmax(scores: list) -> np.ndarray:
	# Your code here
	log_softmax = []
	for i in range(len(scores)):
		log_softmax_val = scores[i] - np.max(scores) - np.log(np.sum(np.exp(scores - np.max(scores))))
		log_softmax.append(log_softmax_val)

	return log_softmax