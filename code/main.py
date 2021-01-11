from utils import *
from models import *
import numpy as np
import sys
import random


if __name__ == '__main__':

    m = Models()

    x_train = np.array(read_x_data(train=True, raw=False))
    y_train = np.array(read_y_data())
    x_test = np.array(read_x_data(train=False, raw=False))

    random.seed(2)
    random.shuffle(x_train)
    random.seed(2)
    random.shuffle(y_train)

    # x_train = x_train[:9]
    # y_train = y_train[:9]
    m.train_folds(x_train, y_train, 6)

    sys.exit()

    # kernel_mat = m.kernel_matrix(x_train, x_train)
    # save_object(kernel_mat, 'gaussian_sigma_0.1')
    
    # kernel_mat_test = m.kernel_matrix(x_train, x_test)
    # save_object(kernel_mat_test, 'test_gaussian_sigma_0.1')
    

    # kernel_matrix = load_object('gaussian_sigma_0.1')

    # alpha = m.solve_linear_system(kernel_matrix, len(x), 1, y)
    # save_object(alpha, 'alpha_KRR_gaussian_sigma_0.1_lam_1')

    alpha = load_object('alpha_KRR_gaussian_sigma_0.1_lam_1')
    alpha = alpha.reshape(6000,1)

    test_kernel_matrix = load_object('test_gaussian_sigma_0.1')

    pred = m.predict_labels(alpha, np.matrix.transpose(test_kernel_matrix))

    pred = array_to_labels(pred)


    print(pred[:100])