import numpy as np

def divide_on_feature(X, feature_i, threshold):
	chosen_col = X[:, feature_i]
	mask = chosen_col >= threshold
	X_kept = X[mask]
	X_unkept = X[~mask]
	return [X_kept, X_unkept]