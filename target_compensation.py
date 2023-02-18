import numpy as np
import copy

# target compensation
def target_compensation(matrix, matrix_target_raw, types):
    matrix_target = copy.deepcopy(matrix_target_raw)
    # Compensation is performed if the performance of alternatives is better than the target.
    ind_profit = np.where(types == 1)[0]
    ind_cost = np.where(types == -1)[0]
    for i in range(matrix.shape[0]):
        for j in ind_profit:
            if matrix_target[i, j] < matrix[i, j]:
                matrix_target[i, j] = matrix[i, j]
        for j in ind_cost:
            if matrix_target[i, j] > matrix[i, j]:
                matrix_target[i, j] = matrix[i, j]

    return matrix_target
