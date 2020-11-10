# -------------------------------------------------------------------------
'''
    Problem 4: Explain a trained network
'''
import pickle

import numpy as np
from sklearn import metrics

from problem1 import *
from problem2 import *
from problem3 import *

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

    with open('../data/trained_model.pkl', 'rb') as f:
        nn = pickle.load(f)

    # then explain predictions of some images
    feature_maps = {}
    for c in range(num_classes):
        image_c = kwargs['Test X'][:, np.where(kwargs['Test Y'][c, :] == 1)[0]]
        feature_map = nn.explain(image_c, c)
        feature_maps[c] = feature_map
    pass
    with open(kwargs['Feature map filename'], 'wb') as f:
        pickle.dump(feature_maps, f)
