from utils import *
from models import *
import numpy as np


if __name__ == '__main__':
    
    utils = Utils()

    x = utils.read_x_data(raw=False)
    y = utils.read_y_data()


    m = Models()

    x = np.array(x)
    
    a=[1,2,3,4]
    utils.save_object(a, 'aaa')
    print(utils.load_object('aaa'))

    kernel_mat = m.kernel_matrix(x)
    utils.save_object(kernel_mat, 'gaussian_sigma_0.1')
    

    # print('Gaussian kernel')    
    # print(m.gaussian_kernel(x[0], x[0]))
    # print(m.kernel_matrix(x))
