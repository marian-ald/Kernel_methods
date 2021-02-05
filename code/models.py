import numpy as np
from functools import partial
from utils import *
import itertools as it


class Models(object):
    """
    docstring
    """
    def __init__(self):
        pass
    
    def gaussian_kernel(self, sigma, x1, x2):
        return np.exp(-0.5 * np.linalg.norm(x1 - x2) ** 2 / np.square(sigma)) #/ (sigma * np.sqrt(2*np.pi))


    def rbf_kernel(self, x1, x2, gamma = 1):
        '''
        RBF: Radial basis function or guassian kernel
        ----------------------------------------------
        :param: x1: NxD transposed feature space
        :param: x2: NxD feature space
        :param: gamma: 1/2(sigma-square)
        :return type: kernel(Gram) matrix
        '''
        return np.exp(-gamma * np.linalg.norm(x1[:, np.newaxis] - x2[np.newaxis, :], axis = 2)**2)


    def kernel_matrix_training(self, X, kernel):
        """ 
            Compute the kernel matrix for the training data.
        """
        X_count = len(X)

        K = np.zeros((X_count, X_count))
        for i in range(X_count):
            K[i,i] = kernel(X[i], X[i])

        for i, j in it.combinations(range(X_count), 2):
            K[i,j] = K[j,i] = kernel(X[i], X[j])
        return K


    def kernel_matrix_test(self, X1, X2, kernel):
        X1_count = X1.shape[0]
        X2_count = X2.shape[0]

        K = np.zeros((X1_count, X2_count))
        for i in range(X1_count):
            for j in range(X2_count):
                K[i,j] = kernel(X1[i], X2[j])
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


    def train_folds(self, data, labels, folds):
        """
        docstring
        """

        len_data = len(data)
        data = np.array(list(zip(data, labels)))
        len_fold = int(len(data) / folds)

        lambda_values = [0.5, 0.9]
        sigma_values = [0.5]

        for lam in lambda_values:
            accuracy_values = []
            for sigma in sigma_values:
                # Build a partial gaussian function with the current 'sigma' value
                kernel_func = partial(self.gaussian_kernel, sigma)
                print('Processing sigma value={}'.format(sigma))

                fold_accuracy = 0
                for i in range(folds):
                    print('Fold: {}'.format(i))
                    # Training data is obtained by concatenating the 2 subsets: at the right + at the left
                    # of the current fold
                    train_data = [*data[0:i*len_fold], *data[(i+1)*len_fold:len_data]]

                    # The current fold is used to test the model
                    test_data = [*data[i*len_fold:(i+1)*len_fold]]
                    
                    x_train = np.array([x[0] for x in train_data])
                    y_train = np.array([x[1] for x in train_data])

                    x_test = np.array([x[0] for x in test_data])
                    y_test = np.array([x[1] for x in test_data])

                    # Build the Gram matrix
                    gram_matrix = self.kernel_matrix_training(x_train, kernel_func)

                    # Solve the linear system in order to find the vector weights
                    alpha = self.solve_linear_system(gram_matrix, len(x_train), lam, y_train)
                    alpha = alpha.reshape(len(x_train),1)

                    # Build the Gram matrix for the test data
                    gram_mat_test = self.kernel_matrix_test(x_train, x_test, kernel_func)

                    # Compute predictions over the test data
                    pred = self.predict_labels(alpha, np.matrix.transpose(gram_mat_test))

                    # Convert predictions to labels
                    pred = array_to_labels(pred)

                    fold_accuracy += accuracy_score(pred, y_test)
                
                # Compute average accuracy for all folds
                average_accuracy = fold_accuracy / folds
                accuracy_values.append(average_accuracy)

            print('lambda={}'.format(lam))
            print('For the sigma values: {}'.format(sigma_values))
            print('Accuracies: {}'.format(accuracy_values))
