import numpy as np


class Models(object):
    """
    docstring
    """
    def __init__(self):
        pass
    
    def gaussian_kernel(self,x1, x2, sigma = 0.01):
        return np.exp(-0.5 * np.linalg.norm(x1 - x2) ** 2 / np.square(sigma)) #/ (sigma * np.sqrt(2*np.pi))


    def kernel_matrix(self, X1, X2):
        X1_count = X1.shape[0]
        X2_count = X2.shape[0]

        K = np.zeros((X1_count, X2_count))
        for i in range(X1_count):
            for j in range(X2_count):
                K[i,j] = self.gaussian_kernel(X1[i], X2[j], 0.1)
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


    def predict_labels(self, alpha, K):

        return np.dot(K, alpha)


    def rbf(x1, x2, gamma = None):
        '''
        RBF: Radial basis function or guassian kernel
        ----------------------------------------------
        :param: x1: NxD transposed feature space
        :param: x2: NxD feature space
        :param: gamma: 1/2(sigma-square)
        :return type: kernel(Gram) matrix
        '''
        if not gamma:
            gamma = 1
        else:
            gamma = gamma
        if x1.ndim == 1 and x2.ndim == 1:
            return np.exp(-gamma * np.linalg.norm(x1 - x2)**2)
        elif (x1.ndim > 1 and x2.ndim == 1) or (x1.ndim == 1 and x2.ndim > 1):
            return np.exp(-gamma * np.linalg.norm(x1 - x2, axis = 1)**2)
        elif x1.ndim > 1 and x2.ndim > 1:
            return np.exp(-gamma * np.linalg.norm(x1[:, np.newaxis] - x2[np.newaxis, :], axis = 2)**2)


    def KRR(self, x_tr, x_te, y_tr, y_te):
        pass
