def k_means_clustering(points: list[tuple[float, ...]], k: int, initial_centroids: list[tuple[float, ...]], max_iterations: int) -> list[tuple[float, ...]]:
	import numpy as np
	# Your code here
	points = np.array(points, dtype=float)
	centroids = np.array(initial_centroids, dtype=float)

	for i in range(max_iterations):
		labels = []
		for p in points:
			distances = np.linalg.norm(centroids - p, axis=1)
			labels.append(np.argmin(distances))
		labels = np.array(labels)

		for i in range(k):
			cluster_points = points[labels == i]
			if len(cluster_points) > 0:
				cluster_mean = np.mean(cluster_points, axis=0)
				centroids[i] = cluster_mean 
	
	return [tuple(float(x) for x in np.round(c, 4)) for c in centroids]
		


	#return labels