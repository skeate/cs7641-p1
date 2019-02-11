"""
Decision tree running code
"""

import warnings
import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from helpers import dtclf_pruned, basicResults, scorer
from parse_data import read_gestures, read_sarcasm, read_wine
from plot import plot_learning_curve

ALPHAS = [x * (10 ** y) for x in [-1, 1] for y in range(-4, 4)]


def run_dt(data, title, solved_params=None):
    """
    run the decision tree algo on the data given
    """
    x, y, pipeline = data
    pipe = Pipeline([
        *pipeline,
        ('DT', dtclf_pruned()),
    ])
    print("Splitting into train/test")
    x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y)
    if solved_params is None:
        print("Doing a GridSearch for best hyperparameters")
        params = {
            'DT__criterion': ['gini', 'entropy'],
            'DT__alpha': ALPHAS,
            'DT__class_weight': ['balanced'],
            'DT__min_samples_split': [2, 3, 4, 5],
        }
        clf = basicResults(pipe, x_train, y_train, x_test,
                           y_test, params, 'DT', title)
    else:
        print("Using pre-solved hyperparameters")
        clf = pipe.set_params(**solved_params)
    print ("Plotting learning curve")
    # plot_learning_curve(clf, title + ' decision tree', x,
    #                     y, n_jobs=4, scoring=scorer, ylim=(0, 1))
    # plt.savefig('./graphs/' + title + '-dt.png')
    y_pred = clf.predict(x_test)
    conf = confusion_matrix(y_test, clf.predict(x_test))
    conf = conf.astype('float') / conf.sum(axis=1)[:, np.newaxis]
    print('Confusion matrix:')
    print(conf)
    np.savetxt('./output/DT_{}_confusion.csv'.format(title), conf, delimiter=',', fmt='%.2f')


if __name__ == '__main__':
    run_dt(read_gestures(), 'gestures')
    run_dt(read_wine(), 'wine')
    # plt.show()
