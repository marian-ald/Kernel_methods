import numpy as np


class Models(object):
    """
    docstring
    """
    def __init__(self):
        pass

    def compute_K_matrix(self, train_data):

        return K


    def solve_linear_system(self, K, n, lam, y):
        """
            K = nxn size matrix
            y = n size vector
        """
        I = np.identity(n)
        mat_coef = K + n * lam * I
        alpha = np.linalg.solve(mat_coef, y)

        return alpha


    def predict_labels(self, alpha, K_matrix, test_data):

        return labels



    def KRR(self, x_tr, x_te, y_tr, y_te):
        pass
