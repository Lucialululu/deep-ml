def transpose_matrix(a: list[list[int|float]]) -> list[list[int|float]]:
    """
    Transpose a 2D matrix by swapping rows and columns.
    
    Args:
        a: A 2D matrix of shape (m, n)
    
    Returns:
        The transposed matrix of shape (n, m)
    """
    import numpy as np
    rows = len(a)
    columns = len(a[0])
    shape = (columns, rows)
    transposed = np.zeros(shape)

    for i in range(rows):
        for j in range(columns):
            transposed[j][i] = a[i][j]

    return transposed