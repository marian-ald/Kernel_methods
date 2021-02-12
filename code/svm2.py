import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import cvxopt
import cvxopt.solvers
from utils import *
from models import *

class C_SVM():
    """
    Implementation of C-SVM algorithm
    """
    def __init__(self, K, ID, C=10, eps=1e-5, solver='CVX', print_callbacks=True):
        """
        :param K: np.array, kernel
        :param ID: np.array, Ids (for ordering)
        :param C: float, regularization constant
        :param eps: float, threshold determining whether alpha is a support vector or not
        :param solver: int, choose between 'CVX' or 'BFGS'
        :param print_callbacks: Bool, print evolution of gradient descent when using 'L-BFGS-B' solver (suggested)
        """
        self.K = K
        self.ID = ID
        self.C = C
        self.eps = eps
        self.solver = solver
        self.print_callbacks = print_callbacks
        self.Nfeval = 1

    def loss(self, a):
        """
        :param a: np.array, alphas
        :return: float, loss function
        """
        return -(2 * np.dot(a, self.y_fit) - np.dot(a.T, np.dot(self.K_fit, a)))

    def jac(self, a):
        """
        :param a: np.array, alphas
        :return: np.array, loss Jacobian
        """
        return -(2 * self.y_fit - 2*np.dot(self.K_fit, a))

    def callbackF(self, Xi, Yi=0):
        """
        Print useful information about gradient descent evolution.
        :param Xi: np.array, values returned by scipy.minimize at each iteration
        :return: None, update print
        """
        if self.print_callbacks:
            if self.Nfeval == 1:
                self.L = self.loss(Xi)
                print('Iteration {0:2.0f} : loss={1:8.4f}'.format(self.Nfeval, self.L))
            else:
                l_next = self.loss(Xi)
                print('Iteration {0:2.0f} : loss={1:8.4f}, tol={2:8.4f}'
                      .format(self.Nfeval, l_next, abs(self.L - l_next)))
                self.L = l_next
            self.Nfeval += 1
        else:
            self.Nfeval += 1

    def fit(self, X, y):
        """
        Train C-SVM on X and y.
        :param X: pd.DataFrame, training features
        :param y: pd.DataFrame, training labels
        """
        # self.Id_fit = np.array(X.loc[:, 'Id'])
        # self.idx_fit = np.array([np.where(self.ID == self.Id_fit[i])[0] for i in range(len(self.Id_fit))]).squeeze()
        # self.K_fit = self.K[self.idx_fit][:, self.idx_fit]
        self.idx_fit = np.array(list(range(len(X))))
        self.K_fit = self.K
        self.y_fit = y
        self.n = self.K_fit.shape[0]
        print('n is {}'.format(self.n))

        if self.solver == 'BFGS':
            # initialization
            a0 = np.random.randn(self.n)
            # Gradient descent
            bounds_down = [-self.C if self.y_fit[i] <= 0 else 0 for i in range(self.n)]
            bounds_up = [+self.C if self.y_fit[i] >= 0 else 0 for i in range(self.n)]
            bounds = [[bounds_down[i], bounds_up[i]] for i in range(self.n)]
            res = fmin_l_bfgs_b(self.loss, a0, fprime=self.jac, bounds=bounds, callback=self.callbackF)
            self.a = res[0]
        elif self.solver == 'CVX':
            r, o, z = np.arange(self.n), np.ones(self.n), np.zeros(self.n)
            P = cvxopt.matrix(self.K_fit.astype(float), tc='d')
            q = cvxopt.matrix(-self.y_fit, tc='d')
            # print(' 1 {}'.format(len(np.r_[self.y_fit, -self.y_fit])))
            # print(' 2 {}'.format(len(np.r_[r, r + self.n])))
            # print(' 3 {}'.format(len(np.r_[r, r])))

            G = cvxopt.spmatrix(np.r_[self.y_fit, -self.y_fit], np.r_[r, r + self.n], np.r_[r, r], tc='d')
            h = cvxopt.matrix(np.r_[o * self.C, z], tc='d')
            cvxopt.solvers.options['show_progress'] = False
            sol = cvxopt.solvers.qp(P, q, G, h)
            self.a = np.ravel(sol['x'])
        # Align support vectors index with index from fit set
        self.idx_sv = np.where(np.abs(self.a) > self.eps)
        self.y_fit = self.y_fit[self.idx_sv]
        self.a = self.a[self.idx_sv]
        self.idx_sv = self.idx_fit[self.idx_sv]
        # Intercept
        self.y_hat = np.array([np.dot(self.a, self.K[self.idx_sv, i]).squeeze() for i in self.idx_sv])
        self.b = np.mean(self.y_fit - self.y_hat)


    def predict(self, K_test):
        """
        Make predictions for features in X
        :param X: pd.DataFrame, features
        :return: np.array, predictions (-1/1)
        """
        # Align prediction IDs with index in kernel K
        # self.Id_pred = np.array(X.loc[:, 'Id'])
        # self.idx_pred = np.array([np.where(self.ID == self.Id_pred[i])[0] for i in range(len(self.Id_pred))]).squeeze()
        pred = []
        for i in range(len(K_test[0])):
            pred.append(np.sign(np.dot(self.a, K_test[self.idx_sv, i].squeeze()) + self.b))
        return np.array(pred)

def main():
    distribution = -1

    # Check if a certain dataset is selected, if no argument is provided, all 3
    # will be processed
    if len(sys.argv) > 1:
        if not(sys.argv[1] == '0' or sys.argv[1] == '1' or sys.argv[1] == '2'):
            sys.exit('Error: Expected arguments:0/1/2')
        distribution = int(sys.argv[1])
        print('Processing distribution: {}'.format(distribution))

    m = Models()

    x_train = np.array(read_x_data(train=True, raw=True))
    y_train = np.array(read_y_data())
    x_test = np.array(read_x_data(train=False, raw=True))

    # x_train = x_train[1]
    # y_train = y_train[1]
    # x_test = x_test[1]
    
    # x_train = x_train[:1000]
    # y_train = y_train[:1000]
    
    # x_test = x_train[:200]
    # x_train = x_train[200:]

    # y_test = y_train[:200]
    # y_train = y_train[200:]


    # Build a partial spectrum function
    kernel_func = partial(np.dot)

    all_labels = []

    
    # Build the Gram matrix for the spectrum kernel
    histograms_X_train = m.spectrum_histogram(x_train[distribution], x_train[distribution], 8, 0)
    gram_matrix_train = m.kernel_matrix_training(histograms_X_train, kernel_func)


    # Build the Gram matrix for the test data
    histograms_X_test = m.spectrum_histogram(x_train[distribution], x_test[distribution], 8, 0)
    gram_mat_test = m.kernel_matrix_test(histograms_X_train, histograms_X_test, kernel_func)


    # gram_matrix_train = load_object('spectr_kernel_aug_k=7_train_distrib={}'.format(distrib))
    # gram_mat_test = load_object('spectr_kernel_aug_k=7_test_distrib={}'.format(distrib))

    model = C_SVM(gram_matrix_train, 1)

    model.fit(x_train[distribution], y_train[distribution])

    predicted_values = list(model.predict(gram_mat_test))
    
    all_labels += list(predicted_values)

    predicted_values = [0 if x == -1 else 1 for x in predicted_values]

    # print('accuracy = {}'.format(np.mean(y_test == predicted_values)))
    write_labels_csv_KRR_spectrum(predicted_values, 'test_results_SVM_spectr_distribution={}.csv'.format(distribution), distribution)

    # save_object(all_labels, 'labels_svm.pkl')
    # all_labels = [0 if x == -1 else 1 for x in all_labels]
    # write_labels_csv(all_labels)


if __name__ == "__main__":
    main()
