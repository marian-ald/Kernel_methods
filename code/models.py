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
        return np.exp(-0.5 * (np.linalg.norm(x1 - x2) ** 2) / sigma**2) #/ (sigma * np.sqrt(2*np.pi))


    def miss(self, s, t):
        """ Count the number of mismatches between two strings."""
        return sum((si != sj for si, sj in zip(s, t)))


    def mismatch_kernel(self, k, delta, m, gamma, s, t):
        """ String kernel with displacement, mismatches and exponential decay. """
        L = len(s)
        return sum(((np.exp(-gamma * d**2) \
                    * np.exp(-gamma * self.miss(s[i:i + k], t[d + i:d + i + k])) \
                    * (self.miss(s[i:i + k], t[d + i:d + i + k]) <= m) 
                    for i, d in it.product(range(L - k + 1), range(-delta, delta + 1))
                    if i + d + k <= L and i + d >= 0)))


    def poly_kernel(self, power, x1, x2):
        """
        polynominal function
        k(x1,x2) = (1+x1 * x2) ^i
        """
        return pow((1 + np.dot(x1, x2)), power)


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



    def spectrum_histogram(self, X, k, distribution):
        # Build the kmers dictionary for the training sequences
        kmer_dict = build_kmers_dict(X, k, distribution)

        # Or load the kmers dictionary if computed before
        # kmer_dict = load_object('kmer_dict_k={}'.format(k))

        conv = Converter(k)

        # List which stores the kmers frequencies for each sequence
        histogram_X = []

        for seq in X:
            # Set all values in the dictionary to 0
            kmer_dict = dict.fromkeys(kmer_dict, 0)

            # For each kmer in the current seq, increment its occurence nb in the frequency dictionary
            for kmer in conv.all_kmers_as_ints(seq):
                kmer_dict[kmer] += 1

            # Get a snapshot of the kmer_dic and insert as a list of kmer frequencies in the histogram
            histogram_X.append(list(kmer_dict.values()))

        save_object(histogram_X, 'spectrum_histogram_distrib={}_k={}'.format(distribution, k))
        return histogram_X


    def spectrum_matrix(self, X, k, distribution):
        """
            Compute the spectrum kernel for a list of sequences X.
            X can be omitted(with []) if the spectrum histogram was computed before
        """
        histograms_X = []

        if len(X) == 0:
            histograms_X = load_object('spectrum_histogram_distrib={}_k={}'.format(distribution, k))
        else:
            print('Computing the spectrum histogram')
            histograms_X = self.spectrum_histogram(X, k, distribution)

        K = self.kernel_matrix_training(histograms_X, partial(np.dot))

        save_object(K, 'spectrum_kernel_distribution={}_k={}'.format(distribution, k))
        return K


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


    def compute_alpha_KRR(self, train_x, train_y, lam, sigma, distrib):
        kernel = partial(self.gaussian_kernel, sigma)

        # Compute training Gram matrix
        K_train = self.kernel_matrix_training(train_x, kernel)

        # Solve the linear system in order to find the vector weights
        alpha = self.solve_linear_system(K_train, len(train_x), lam, train_y)
        alpha = alpha.reshape(len(train_x), 1)

        save_object(alpha, 'alpha_velues_sigma={}_lambda={}_distrib={}'.format(sigma, lam, distrib))

        return alpha


    def run_KRR(self, x_train, y_train, x_test):
        test_labels = []
        for i in range(3):
            # Compute alpha coefficients using the training set
            alpha = m.compute_alpha_KRR(x_train[i], y_train[i], 0.001, 0.1, i)
            
            # Define the gaussian kernel
            kernel = partial(m.gaussian_kernel, 0.1)
            
            # Predict the labels over the test set
            labels = m.do_predictions(x_train[i], y_train[i], x_test[i], alpha, kernel)
            test_labels = test_labels + labels

            write_labels_csv(test_labels)


    def do_predictions(self, train_x, train_y, test_x, alpha, kernel):

        # Compute test Gram matrix
        K_test = self.kernel_matrix_test(train_x, test_x, kernel)
        labels = self.predict_labels(alpha, np.matrix.transpose(K_test))

        labels = array_to_labels(labels, 0)
        return labels


    def train_folds(self, data, labels, folds):
        """
        docstring
        """

        len_data = len(data)
        data = np.array(list(zip(data, labels)))
        len_fold = int(len(data) / folds)

        lambda_values = [0.1, 0.3, 0.6, 0.9]
        sigma_values = [0.0001, 0.001, 0.01, 0.1, 0.5]

        for lam in lambda_values:
            accuracy_values = []
            for sigma in sigma_values:
                # Build a partial gaussian function with the current 'sigma' value
                kernel_func = partial(self.gaussian_kernel, sigma)
                print('Processing sigma value={}'.format(sigma))

                # TODO Compute the whole gram matrix here only once
                # each fold will extract

                fold_accuracy = 0
                for i in range(folds):
                    # print('Fold: {}'.format(i))
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
                    pred = array_to_labels(pred, -1)

                    fold_accuracy += accuracy_score(pred, y_test)
                
                # Compute average accuracy for all folds
                average_accuracy = fold_accuracy / folds
                accuracy_values.append(average_accuracy)

            print('lambda={}'.format(lam))
            print('For the sigma values: {}'.format(sigma_values))
            print('Accuracies: {}\n'.format(accuracy_values))


    def run_and_save_kernels(self, train):

        kernel_mat = self.kernel_matrix_training(train, partial(self.mismatch_kernel, 8, 1, 1, 1))
        save_object(kernel_mat, 'mismatch_k=8_delta=1_m=1_gamma=1')

        kernel_mat = self.kernel_matrix_training(train, partial(self.mismatch_kernel, 7, 1, 1, 1))
        save_object(kernel_mat, 'mismatch_k=7_delta=1_m=1_gamma=1')

