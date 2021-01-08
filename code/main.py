from utils import *
from models import *


if __name__ == '__main__':
    
    utils = Utils()

    x = utils.read_x_data(raw=False)
    y = utils.read_y_data()


    m = Models()