'''
    Unit test 2:
    This file includes unit tests for problem2.py.
    You could test the correctness of your code by typing `nosetests -v test2.py` in the terminal.
'''

from problem1 import *
from problem2 import *

import numpy as np

# -------------------------------------------------------------------------
def test_forward():
    ''' (10 points) NN:forward()'''

    X = np.asmatrix(np.array([[1,1], [-1,-1]]))
    input_dim = 2
    num_classes = 3
    # make a tiny network
    dimensions = [input_dim, 2, num_classes]

    activation_funcs = {1:Tanh, 2:Softmax}
    loss_func = CrossEntropyLoss
    nn = NN(dimensions, activation_funcs, loss_func, rand_seed = 42)

    output = nn.forward(X)
    assert np.allclose(output, np.array([[0.34959677, 0.34959677],
                                         [0.39983301, 0.39983301],
                                         [0.25057022, 0.25057022]]),
                       atol=1e-6)

# -------------------------------------------------------------------------
def test_bacward():
    ''' (10 points) NN:bacward()'''
    X = np.asmatrix(np.array([[1,1], [-1,-1]])).T
    Y = np.asmatrix(np.array([[1, 0, 0], [0, 1, 0]])).T
    input_dim = 2
    num_classes = 3
    # make a tiny network
    dimensions = [input_dim, 2, num_classes]

    activation_funcs = {1:Tanh, 2:Softmax}
    loss_func = CrossEntropyLoss
    nn = NN(dimensions, activation_funcs, loss_func, rand_seed = 42)

    output = nn.forward(X)
    nn.backward(Y)

    assert type(nn.dW) == dict
    assert type(nn.dW[2]) == np.matrixlib.defmatrix.matrix
    assert np.allclose(nn.dW[2], np.array([[-0.15813148, -0.58064749],
                                           [ 0.15921209,  0.5846154],
                                           [-0.00108061, -0.00396791]]),
                       atol=1e-6)
    assert type(nn.dW[1]) == np.matrixlib.defmatrix.matrix
    assert np.allclose(nn.dW[1], np.array([[0.69106193, 0.69106193],
                                           [0.06855901, 0.06855901]]),
                       atol=1e-6)

    assert type(nn.db) == dict
    assert type(nn.db[2]) == np.matrixlib.defmatrix.matrix
    assert np.allclose(nn.db[2], np.array([[-0.15770135], [-0.15593308], [0.31363443]]),
                       atol=1e-6)
    assert type(nn.db[1]) == np.matrixlib.defmatrix.matrix
    assert np.allclose(nn.db[1], np.array([[-0.21162467], [0.00937698]]),
                       atol=1e-6)

def test_update_parameters():
    ''' (5 points) NN:update_parameters()'''
    X = np.asmatrix(np.array([[1, 1], [-1, -1]])).T
    Y = np.asmatrix(np.array([[1, 0, 0], [0, 1, 0]])).T
    input_dim = 2
    num_classes = 3
    # make a tiny network
    dimensions = [input_dim, 2, num_classes]

    activation_funcs = {1: Tanh, 2: Softmax}
    loss_func = CrossEntropyLoss
    nn = NN(dimensions, activation_funcs, loss_func, rand_seed=42)

    output = nn.forward(X)
    nn.backward(Y)
    nn.update_parameters(lr=0.1, weight_decay=0.01)

    assert type(nn.W) == dict
    assert type(nn.W[2]) == np.ndarray
    assert np.allclose(nn.W[2], np.array([[-0.13079753, -0.08853565],
                                          [ 0.97287285, 0.42205312],
                                          [-0.2938444, 0.34011044]]),
                       atol=1e-6)
    assert type(nn.W[1]) == np.ndarray
    assert np.allclose(nn.W[1], np.array([[0.27861145, -0.16589614],
                                          [0.44654921, 1.05931939]]),
                       atol=1e-6)

    assert type(nn.b) == dict
    assert type(nn.b[2]) == np.ndarray
    assert np.allclose(nn.b[2], np.array([[ 0.01577014], [ 0.01559331], [-0.03136344]]),
                       atol=1e-6)
    assert type(nn.b[1]) == np.ndarray
    assert np.allclose(nn.b[1], np.array([[0.02116247], [-0.0009377]]),
                       atol=1e-6)

def test_explain():
    ''' (10 points) NN:explain() Required for graduate'''
    X = np.asmatrix(np.array([[1, 1], [-1, -1]])).T
    y = 1
    # Y = np.asmatrix(np.array([[1, 0, 0], [0, 1, 0]])).T
    input_dim = 2
    num_classes = 3
    # make a tiny network
    dimensions = [input_dim, 2, num_classes]

    activation_funcs = {1: Tanh, 2: Softmax}
    loss_func = CrossEntropyLoss
    nn = NN(dimensions, activation_funcs, loss_func, rand_seed=42)

    feature_map = nn.explain(X, y)

    assert type(feature_map) == np.matrixlib.defmatrix.matrix
    assert np.allclose(feature_map, np.array([[0.36689668, 0.36689668],
                                              [-0.00298354, -0.00298354]]),
                       atol=1e-6)


