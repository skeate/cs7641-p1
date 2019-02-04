"""
Artificial Neural Network
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import neural_network
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from helpers import basicResults, scorer
from parse_data import read_wine, read_gestures
from plot import plot_learning_curve, plot_timing_curve, plot_iteration_curve

iter_adjust = {
    'ANN__max_iter': [2**x for x in range(12)] +
                     [2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000],
}


def run_ann(data, dataset, solved_params=None):
    x, y, pipeline = data
    print('Data size: ', x.shape)
    pipe = Pipeline([
        *pipeline,
        ('ANN', neural_network.MLPClassifier(max_iter=1000, early_stopping=True)),
    ])
    print('Splitting dataset for ' + dataset)
    x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y)
    if solved_params is None:
        print('Calculating hyperparameters for ' + dataset)
        dim = x.shape[1]
        params = {
            'ANN__hidden_layer_sizes': [(d,) for d in [dim, dim//2]],
            'ANN__solver': ['adam'],
            'ANN__alpha': 10.0 ** -np.arange(1, 7),
            'ANN__activation': ['relu', 'tanh', 'logistic'],
        }
        clf = basicResults(pipe, x_train, y_train, x_test,
                           y_test, params, 'ANN', dataset)
    else:
        print('Using presolved hyperparameters for ' + dataset)
        clf = pipe.set_params(**solved_params)
    # plot_learning_curve(clf, dataset + ' neural network',
    #                     x, y, cv=5, n_jobs=4, scoring=scorer)
    # plt.savefig('./graphs/' + dataset + '-ann.png')
    print('Creating timing curve for ' + dataset)
    plot_timing_curve(clf, x, y, 'neural network', dataset)
    plt.savefig('./graphs/' + dataset + '-ANN-timing.png')
    print('Creating iteration curve for ' + dataset)
    plot_iteration_curve(clf, x_train, y_train, x_test, y_test, iter_adjust, 'neural network', dataset)
    plt.savefig('./graphs/' + dataset + '-ANN-iteration.png')


if __name__ == '__main__':
    run_ann(read_wine(), 'wine')
    run_ann(read_gestures(), 'gestures')
    # plt.show();
