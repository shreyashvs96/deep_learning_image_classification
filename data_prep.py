from helper_functions import *


def data_prep_cat():
    # Load the datasets
    train_x_orig, train_y, test_x_orig, test_y, classes = load_cat_data()

    # Reshape the training and test datasets
    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

    # Standardize data to have values between 0 and 1
    train_x = train_x_flatten / 255
    test_x = test_x_flatten / 255

    print('Number of training examples:', train_x.shape[1])
    print('Number of features:', train_x.shape[0])
    return train_x, train_y, test_x, test_y, classes
