import numpy as np
def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):
	# Your code here
	weights = initial_weights
	bias = initial_bias
	mse_values = []

	for i in range(epochs):
		z = np.sum(features * weights, axis=1) + bias
		preds = 1/(1+np.exp(-z))

		n = len(features)

		mse = 1/n * np.sum((preds - labels)**2)

		mse_values.append(mse)

		sigmoid_diff_z = np.exp(-z)/(1+np.exp(-z))**2
		mse_w = 2/n * (preds - labels) * sigmoid_diff_z @ features
		mse_b = 2/n * np.sum((preds - labels) * sigmoid_diff_z)

		weights = weights - learning_rate * mse_w
		bias = bias - learning_rate * mse_b

	updated_weights = weights
	updated_bias = bias

	return updated_weights, updated_bias, mse_values