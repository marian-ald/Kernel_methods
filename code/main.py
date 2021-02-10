from utils import *
from models import *
import numpy as np
import sys
import random
import math



if __name__ == '__main__':

    m = Models()
    # Variable to indicate which dataset distribution to use
    distribution = -1

    # Check if a certain dataset is selected, if no argument is provided, all 3
    # will be processed
    if len(sys.argv) > 1:
        if not(sys.argv[1] == '0' or sys.argv[1] == '1' or sys.argv[1] == '2'):
            sys.exit('Error: Expected arguments:0/1/2')
        distribution = int(sys.argv[1])
        print('Processing distribution: {}'.format(distribution))

    # Each of the following lists contains 3 elements:
    # An element is a list which incorporate data for a single distribution
    # E.g. x_train = [train0, train1, train2]
    x_train = np.array(read_x_data(train=True, raw=True))
    y_train = np.array(read_y_data())
    x_test = np.array(read_x_data(train=False, raw=True))


    # y_train = y_train[1][:40]
    # x_train = x_train[1][:40]


    # x_test = x_train[10:]
    # x_train = x_train[:10]
    # y_test = y_train[10:]
    # y_train = y_train[:10]



    # Run the KRR using gaussian kernel
    # m.run_KRR(x_train, y_train, x_test)

    # Run the KRR using the spectrum kernel
    m.run_KRR_spectrum(x_train[distribution], y_train[distribution], x_test[distribution], distribution)

    # for i in range(1):
    #     K_spectrum = m.spectrum_matrix(x_train, 7, i)
    # m.train_folds_spectrum(x_train, y_train, 5)
    # print(type(y_train[0]))
    # print(y_train[:10])
    # m.train_folds_spectrum(x_train, y_train, 5, 0)
    sys.exit()

    # Run k-cross validation for KRR+gaussian kernel
    # if distribution != -1:
    #     m.train_folds(x_train[distribution], y_train[distribution], 5)
    # else:
    #     for i in range(3):
    #         m.train_folds(x_train[i], y_train[i], 5)

    # Run k-cross validation for KRR+spectrum kernel
    if distribution != -1:
        m.train_folds_spectrum(x_train[distribution], y_train[distribution], 5, distribution)
    else:
        for i in range(3):
            m.train_folds_spectrum(x_train[i], y_train[i], 5, distribution)



    # x_test = x_train[5000:]
    # x_train = x_train[:5000]
    # y_test = y_train[5000:]
    # y_train = y_train[:5000]


    # krr = KernelRidge(alpha=0.001,kernel='rbf')
    # krr.fit(x_train,y_train)

    # new_y = krr.predict(x_test)
    # new_y = array_to_labels(new_y)


