# -------------------------------------------------------------------------
'''
    Problem 3: Train and test neural networks on MNIST classification.
'''
import pickle

import numpy as np
from sklearn import metrics

from problem1 import *
from problem2 import *

#--------------------------
def load_MNIST():
    """
    Load the MNIST handwritten digit recognition datasets for multi-class classification.
    :return: training X (columns are examples),
            training Y (columns are one-hot vectors indicating class labels),
            test X (columns are examples),
            test Y (columns are one-hot vectors indicating class labels)
    """

    with open('../data/X_train.pkl', 'rb') as f:
        X_train = pickle.load(f)
        X_train.reshape(-1, )
    with open('../data/Y_train.pkl', 'rb') as f:
        Y_train = pickle.load(f)
    with open('../data/X_test.pkl', 'rb') as f:
        X_test = pickle.load(f)
    with open('../data/Y_test.pkl', 'rb') as f:
        Y_test = pickle.load(f)

    num_classes = len(np.unique(Y_train))
    input_dim, n_samples = X_train.shape

    print(f'num_classes = {num_classes}')
    print(f'num_pixels = {input_dim}')
    print(f'num_training_samples = {n_samples}')

    tr_y_multi_class = np.zeros((num_classes, n_samples))
    for i in range(num_classes):
        tr_y_multi_class[i, np.where(Y_train == i)] = 1
    Y_train = np.asmatrix(tr_y_multi_class)

    input_dim, n_samples = X_test.shape
    te_y_multi_class = np.zeros((num_classes, n_samples))
    for i in range(num_classes):
        te_y_multi_class[i, np.where(Y_test == i)] = 1
    Y_test = np.asmatrix(te_y_multi_class)

    return X_train, Y_train, X_test, Y_test


if __name__ == '__main__':

    ## test multiple class prediction
    tr_X, tr_Y, te_X, te_Y = load_MNIST()
    print('===========MNIST===========')


    kwargs = {'Training X': tr_X,
              'Training Y': tr_Y,
              'Test X': te_X,
              'Test Y': te_Y,
              'max_iters': 201,
              'Learning rate': 0.001,
              'Weight decay': 0.0001,
              'Mini-batch size': 1024,
              'record_every': 10,
              'Test loss function name': '0-1 error',
              'Feature map filename': '../data/feature_maps.pkl'
              }

    input_dim, n_samples = tr_X.shape
    num_classes = tr_Y.shape[0]

    # input -> hidden -> output
    # you're encouraged to explore other architectures with more or less number of layers
    # Is more layers the better?
    # Will ReLU work better than Sigmoid/Tanh?
    dimensions = [input_dim, 128, 64, num_classes]
    activation_funcs = {1:ReLU, 2:ReLU, 3:Softmax}
    loss_func = CrossEntropyLoss

    nn = NN(dimensions, activation_funcs, loss_func)
    nn.train(**kwargs)

    # after training finishes, save the model parameters
    with open('../data/trained_model.pkl', 'wb') as f:
        pickle.dump(nn, f)

    # evaluate the model on the test set and report the detailed performance
    predicted = np.argmax(nn.forward(kwargs['Test X']), axis=0)
    ground_truth_labels = np.argmax(kwargs['Test Y'], axis=0)
    print("Classification report for the neural network \n%s\n"
          % metrics.classification_report(np.squeeze(np.asarray(ground_truth_labels)), np.squeeze(predicted), digits=3))
    print("Confusion matrix \n%s\n"
          % metrics.confusion_matrix(np.squeeze(np.asarray(ground_truth_labels)), np.squeeze(predicted)))
