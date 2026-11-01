import math
import numpy as np

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
	# Your code here
	features = np.array(features)
	weights = np.array(weights)
	labels = np.array(labels)
	bias = np.array(bias)
	
	z = np.sum(weights * features, axis=1) + bias
	
	predicted = 1/(1+np.exp(-z))

	n = len(features)

	mse = round(1/n * np.sum((predicted - labels)**2), 4)
	
	return predicted, mse