def matrixmul(a:list[list[int|float]], b:list[list[int|float]])-> list[list[int|float]]:
    import numpy as np
    A = np.asarray(a)
    B = np.asarray(b)
    if A.shape[1] == B.shape[0]:
       matrix_mul = A @ B
       return matrix_mul
    else:
       return -1
        