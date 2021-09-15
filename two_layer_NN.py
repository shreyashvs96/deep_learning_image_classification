import numpy as np
import matplotlib.pyplot as plt


def two_layer_model(X, Y, num_hidden_units=7, learning_rate=0.0075, num_iterations=3000, print_cost=False):
    # Linear -> ReLU -> Linear -> Sigmoid

    n_x = X.shape[0]
    n_h = num_hidden_units
    n_y = 1

    np.random.seed(1)
    params = initialize_parameters(n_x, n_h, n_y)
    grads = {}
    costs = []
    print('Number of iterations:', num_iterations)

    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']

    # Gradient descent
    for i in range(0, num_iterations):

        # Forward propagation
        A1, cache1 = linear_activation_forward(X, W1, b1, activation_function='ReLU')
        A2, cache2 = linear_activation_forward(A1, W2, b2, activation_function='Sigmoid')

        # Compute cost
        cost = compute_cost(A2, Y)

        # Backward propagation
        dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))
        dA1, dW2, db2 = linear_activation_backward(dA2, cache2, activation_function='Sigmoid')
        dA0, dW1, db1 = linear_activation_backward(dA1, cache1, activation_function='ReLU')

        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2

        # Update parameters
        params = update_parameters(params, grads, learning_rate)

        W1 = params['W1']
        b1 = params['b1']
        W2 = params['W2']
        b2 = params['b2']

        # Print cost every 100 iterations
        if print_cost and i % 100 == 0 or i == num_iterations - 1:
            print('Cost after iteration ', i, ':', cost)
        if i % 100 == 0 or i == num_iterations - 1:
            costs.append(cost)

    return params, costs


def initialize_parameters(n_x, n_h, n_y):
    """
    :param n_x: number of input units (features)
    :param n_h: number of units in the hidden layer
    :param n_y: number of output units
    :return: dictionary of parameters - W1, b1, W2 and b2
    """
    np.random.seed(1)
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }

    return parameters


def linear_forward(A, W, b):
    Z = W.dot(A) + b
    cache = (A, W, b)
    return Z, cache


def linear_activation_forward(A_prev, W, b, activation_function):
    Z, linear_cache = linear_forward(A_prev, W, b)
    if activation_function == 'ReLU':
        A, activation_cache = relu(Z)
    elif activation_function == 'Sigmoid':
        A, activation_cache = sigmoid(Z)

    cache = (linear_cache, activation_cache)
    return A, cache


def relu(Z):
    A = np.maximum(0, Z)
    cache = Z
    return A, cache


def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    cache = Z
    return A, cache


def compute_cost(AL, Y):
    m = Y.shape[1]
    # Compute loss from aL and y.
    cost = (1. / m) * (-np.dot(Y, np.log(AL).T) - np.dot(1 - Y, np.log(1 - AL).T))
    cost = np.squeeze(cost)
    return cost


def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ


def sigmoid_backward(dA, cache):
    Z = cache
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    return dZ


def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1. / m * np.dot(dZ, A_prev.T)
    db = 1. / m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation_function):
    linear_cache, activation_cache = cache
    if activation_function == 'ReLU':
        dZ = relu_backward(dA, activation_cache)
    elif activation_function == 'Sigmoid':
        dZ = sigmoid_backward(dA, activation_cache)
    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db


def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2  # Number of layers
    for i in range(L):
        parameters['W' + str(i + 1)] = parameters['W' + str(i + 1)] - grads['dW' + str(i + 1)] * learning_rate
        parameters['b' + str(i + 1)] = parameters['b' + str(i + 1)] - grads['db' + str(i + 1)] * learning_rate
    return parameters


def plot_costs(costs, learning_rate=0.0075):
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()