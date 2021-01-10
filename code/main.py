from utils import *
from models import *
import numpy as np


if __name__ == '__main__':
    
    utils = Utils()

    x = utils.read_x_data(raw=False)
    y = utils.read_y_data()


    m = Models()

    x = np.array(x)
    
    print(x[0])
    
    print('Gaussian kernel')    
    print(m.gaussian_kernel(x[0], x[0]))
    print(m.kernel_matrix(x))

