from utils import *
from models import *
import numpy as np


if __name__ == '__main__':
    
    utils = Utils()

    x = utils.read_x_data(raw=False)
    y = np.array(utils.read_y_data())

    m = Models()

    x = np.array(x)

    # kernel_mat = m.kernel_matrix(x)
    # utils.save_object(kernel_mat, 'gaussian_sigma_0.1')
    

    kernel_matrix = utils.load_object('gaussian_sigma_0.1')

    alpha = m.solve_linear_system(kernel_matrix, len(x), 1, y)
    utils.save_object(alpha, 'alpha_KRR_gaussian_sigma_0.1_lam_1')

    alpha = utils.load_object('alpha_KRR_gaussian_sigma_0.1_lam_1')
    print(alpha[:20])

    # print('Gaussian kernel')    
    # print(m.gaussian_kernel(x[0], x[0]))
    # print(m.kernel_matrix(x))
