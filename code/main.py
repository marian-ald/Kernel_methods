from utils import *
from models import *
import numpy as np
import sys
import random
import math



if __name__ == '__main__':

    m = Models()

    # Each of the following lists contains 3 elements:
    # An element is a list which incorporate data for a single distribution
    # E.g. x_train = [train0, train1, train2]
    x_train = np.array(read_x_data(train=True, raw=False))
    y_train = np.array(read_y_data())
    x_test = np.array(read_x_data(train=False, raw=False))

    # random.seed(2)
    # random.shuffle(x_train)
    # random.seed(2)
    # random.shuffle(y_train)


    # kernel_mat = m.kernel_matrix_training(x_train, partial(m.gaussian_kernel, 0.01))
    # kernel_mat = m.kernel_matrix_training(x_train, partial(m.mismatch_kernel, 8, 1, 1, 1))
    # m.run_and_save_kernels(x_train)

    # build_kmers_dict(x_train, 7)

    # K_spectrum = m.spectrum_kernel(x_train, 7)


    # sys.exit()

    y_train = y_train[:1000]
    x_train = x_train[:1000]

    # y_train = y_train[2000:4000]
    # x_train = x_train[2000:4000]

    x_test = x_train[700:]
    x_train = x_train[:700]
    y_test = y_train[700:]
    y_train = y_train[:700]



    # x_test = x_train[5000:]
    # x_train = x_train[:5000]
    # y_test = y_train[5000:]
    # y_train = y_train[:5000]


    # krr = KernelRidge(alpha=0.001,kernel='rbf')
    # krr.fit(x_train,y_train)

    # new_y = krr.predict(x_test)
    # new_y = array_to_labels(new_y)

    # print(new_y[:10])
    # print(y_train[:10])

    # acc = accuracy_score(new_y, y_test)
    # print('############################')
    # print('Accuracy: %.3f' % acc)

    # sys.exit()





    # ################################################

    kernel_mat = m.kernel_matrix_training(x_train, partial(m.gaussian_kernel, 0.01))
    save_object(kernel_mat, 'gaussian_sigma_0.3')
    
    kernel_mat_test = m.kernel_matrix_test(x_train, x_test, partial(m.gaussian_kernel, 0.01))
    # print(kernel_mat)
    save_object(kernel_mat_test, 'test_gaussian_sigma_0.3')
    
    kernel_matrix = load_object('gaussian_sigma_0.3')

    alpha = m.solve_linear_system(kernel_matrix, len(x_train), 0.8, y_train)
    save_object(alpha, 'alpha_KRR_gaussian_sigma_0.3_lam_0.7')

    alpha = load_object('alpha_KRR_gaussian_sigma_0.3_lam_0.7')
    # alpha = alpha.reshape(5000,1)

    test_kernel_matrix = load_object('test_gaussian_sigma_0.3')

    pred = m.predict_labels(alpha, np.matrix.transpose(test_kernel_matrix))
    print(pred[:20])

    pred = array_to_labels(pred)

    print(pred[:20])
    print(y_train[:20])


    a = accuracy_score(pred, y_test)
    print('Accuracy = {} '.format(a))