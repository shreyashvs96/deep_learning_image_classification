import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
from helper_functions import *
from data_prep import *
from two_layer_NN import two_layer_model, plot_costs

if __name__ == '__main__':
    np.random.seed(1)
    train_x, train_y, test_x, test_y, classes = data_prep_cat()

    # 2 layer NN
    params, costs = two_layer_model(train_x, train_y, 7, 0.0075, 2500, True)
    plot_costs(costs, 0.0075)

