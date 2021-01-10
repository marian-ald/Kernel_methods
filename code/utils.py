import csv 
import pandas as pd
import pickle as pkl
import sys


data_folder = '../data/'


class Utils:
    def __init__(self):
        pass


    def read_raw_file(self, file_name):
        """
        docstring
        """
        lines = []
        # opening the CSV file 
        with open(file_name, mode ='r')as file: 
            # reading the CSV file 
            csvFile = csv.reader(file) 

            # displaying the contents of the CSV file 
            for line in csvFile:
                lines.append(line[1])

            return lines[1:]


    def read_mat_file(self, file_name):
        """
        docstring
        """
        df = pd.read_csv(file_name, header=None, delimiter=r"\s+")
        return list(df.values)


    def read_x_data(self, train=True, raw=True):
        """
            Read the 3 files in raw/matrix format for train/test, depending on the flags(raw,train)
        """
        x_data = list([])
        raw_or_prep = '' if raw==True else '_mat100'
        tr_or_test = 'tr' if train==True else 'te'

        for k in range(0,3):
            x_file_name = '{}X{}{}{}.csv'.format(data_folder, tr_or_test, k, raw_or_prep)
            if raw == False:
                x_data_k = self.read_mat_file(x_file_name)
            else:
                x_data_k = self.read_raw_file(x_file_name)

            x_data = x_data+x_data_k
        return x_data


    def read_y_data(self):
        """
            Read the labels from all 3 files as a list of integers.
            Keep '1' as 1
            Transform '0' in -1
        """
        y_train = []

        for k in range(0,3):
            y_file = '{}Ytr{}.csv'.format(data_folder, k)
            y_train_k = self.read_raw_file(y_file)

            # Convert '0's into '1's
            y_train_k = [-1 if x == '0' else 1 for x in y_train_k]

            # Append the content of the current file to 'y_train'
            y_train.extend(y_train_k)
        
        return y_train


    def save_object(self, obj, name):
        file_path = '../data/kernel_mat/{}.pkl'.format(name)
        with open(file_path, 'wb') as output:
            pkl.dump(obj, output)


    def load_object(self, name):
        file_path = '../data/kernel_mat/{}.pkl'.format(name)
        try:
            with open(file_path, 'rb') as pkl_file:
                read_obj = pkl.load(pkl_file)
                return read_obj    
        except IOError:
            print('File {} does not exist.'.format(file_path))
            sys.exit()
    

    def array_to_labels(self, elements):
        """
            For each element in the list:
                if element > 0 => replace it with 1
                if element < 0 => replace it with 0
        """
        return [0 if x < 0 else 1 for x in elements]