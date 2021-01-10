from utils import *
from models import *
import numpy as np


if __name__ == '__main__':
    
    utils = Utils()

    x_train = np.array(utils.read_x_data(raw=False))
    y = np.array(utils.read_y_data())

    x_test = np.array(utils.read_x_data(train=False, raw=False))

    m = Models()

    # kernel_mat = m.kernel_matrix(x_train, x_train)
    # utils.save_object(kernel_mat, 'gaussian_sigma_0.1')
    
    # kernel_mat_test = m.kernel_matrix(x_train, x_test)
    # utils.save_object(kernel_mat_test, 'test_gaussian_sigma_0.1')
    

    # kernel_matrix = utils.load_object('gaussian_sigma_0.1')

    # alpha = m.solve_linear_system(kernel_matrix, len(x), 1, y)
    # utils.save_object(alpha, 'alpha_KRR_gaussian_sigma_0.1_lam_1')

    alpha = utils.load_object('alpha_KRR_gaussian_sigma_0.1_lam_1')
    alpha = alpha.reshape(6000,1)

    test_kernel_matrix = utils.load_object('test_gaussian_sigma_0.1')

    pred = m.predict_labels(alpha, np.matrix.transpose(test_kernel_matrix))

    pred = utils.array_to_labels(pred)


    print(pred[:100])