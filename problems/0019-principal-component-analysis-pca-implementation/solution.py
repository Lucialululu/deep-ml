import numpy as np

def pca(data: np.ndarray, k: int) -> np.ndarray:
    """
    Perform PCA and return the top k principal components.
    
    Args:
        data: Input array of shape (n_samples, n_features)
        k: Number of principal components to return
    
    Returns:
        Principal components of shape (n_features, k), rounded to 4 decimals.
        Each eigenvector's sign is fixed so its first non-zero element is positive.
    """
    # Your code here
    # standardize:
    sdata = (data - np.mean(data, axis=0))/(np.std(data, axis=0))

    # cov matrix:
    cov = np.cov(sdata, rowvar=False)

    # eigenvalues:
    eigenvals, eigenvects = np.linalg.eigh(cov)
    sorted_idx = np.argsort(eigenvals)[::-1]
    top_k = eigenvects[:, sorted_idx[:k]]

    # if sign needs to be flipped:
    for i in range(top_k.shape[1]):
        column = top_k[:, i]
        first_nonzero = column[np.abs(column )> 1e-10][0]
        if first_nonzero < 0:
            top_k[:, i] = -1 * column

    return np.round(top_k, 4)

