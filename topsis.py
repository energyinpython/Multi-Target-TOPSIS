import numpy as np

from pyrepo_mcda import normalizations as norms
from mcdm_method import MCDM_method, MCDM_method_targeted


class TOPSIS(MCDM_method):
    def __init__(self, normalization_method = norms.minmax_normalization):
        self.normalization_method = normalization_method

    def __call__(self, matrix, weights, types):
        TOPSIS._verify_input_data(matrix, weights, types)
        return TOPSIS._topsis(self, matrix, weights, types, self.normalization_method)

    @staticmethod
    def _topsis(self, matrix, weights, types, normalization_method):
        # Normalize matrix using chosen normalization (for example linear normalization)
        norm_matrix = normalization_method(matrix, types)

        # Multiply all rows of normalized matrix by weights
        weighted_matrix = norm_matrix * weights

        # Calculate vectors of PIS (ideal solution) and NIS (anti-ideal solution)
        pis = np.max(weighted_matrix, axis=0)
        nis = np.min(weighted_matrix, axis=0)

        # Calculate chosen distance of every alternative from PIS and NIS
        Dp = (np.sum((weighted_matrix - pis)**2, axis = 1))**0.5
        Dm = (np.sum((weighted_matrix - nis)**2, axis = 1))**0.5

        return Dm / (Dm + Dp)


class TOPSIS_TARGETED(MCDM_method_targeted):
    def __init__(self, normalization_method = norms.minmax_normalization):
        self.normalization_method = normalization_method

    def __call__(self, matrix, matrix_target, weights, types):
        TOPSIS_TARGETED._verify_input_data(matrix, matrix_target, weights, types)
        return TOPSIS_TARGETED._topsis_targeted(self, matrix, matrix_target, weights, types, self.normalization_method)

    @staticmethod
    def _topsis_targeted(self, matrix, matrix_target, weights, types, normalization_method):
        # Normalize matrix using chosen normalization (for example linear normalization)
        nmatrix = normalization_method(matrix, matrix_target, types)
        nmatrix_target = normalization_method(matrix_target, matrix, types)
    
        # Multiply all rows of normalized matrix by weights
        weighted_matrix = nmatrix * weights
        weighted_matrix_target = nmatrix_target * weights

        # Calculate vectors of PIS (ideal solution) and NIS (anti-ideal solution)
        # here pis is weighted_matrix_target
        nis = np.min(weighted_matrix, axis = 0)
    
        # Calculate chosen distance of every alternative from PIS (weighted_matrix_target) and NIS
        Dp = (np.sum((weighted_matrix - weighted_matrix_target)**2, axis = 1))**0.5
        Dm = (np.sum((weighted_matrix - nis)**2, axis = 1))**0.5
    

        return Dm / (Dm + Dp)