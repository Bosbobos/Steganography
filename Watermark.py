import numpy as np


def arnold(matrix):
    m, n = matrix.shape
    new_matrix = matrix.copy()
    k = np.array([[1, 1], [1, 2]])
    for i in range(m):
        for j in range(m):
            ij = np.array([[i],[j]])
            new_ij = np.matmul(k, ij)
            new_i, new_j = new_ij[0][0], new_ij[1][0]
            new_matrix[new_i % m][new_j % m] = matrix[i][j]

    return new_matrix


def reverse_arnold(matrix):
    m, n = matrix.shape
    new_matrix = matrix.copy()
    k = np.array([[2, -1], [-1, 1]])
    for i in range(m):
        for j in range(m):
            ij = np.array([[i],[j]])
            new_ij = np.matmul(k, ij)
            new_i, new_j = new_ij[0][0], new_ij[1][0]
            new_matrix[new_i % m][new_j % m] = matrix[i][j]

    return new_matrix
