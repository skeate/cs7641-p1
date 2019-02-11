import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import ensemble
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from helpers import dtclf_pruned, basicResults, scorer
from parse_data import read_gestures, read_wine
from plot import plot_learning_curve, plot_timing_curve, plot_iteration_curve

alphas = [x * (10 ** y) for x in [-1, 1] for y in range(-4, 4)]

def run_boost(data, dataset, dtparams={}):
    x, y, pipeline = data
    pipe = Pipeline([
        *pipeline,
        ('Boost', ensemble.AdaBoostClassifier(algorithm='SAMME',base_estimator=dtclf_pruned(**dtparams))),
    ])
    x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y)
    params = {
        'Boost__n_estimators': [2 ** i for i in range(8)],
        'Boost__algorithm': ['SAMME', 'SAMME.R'],
    }
    clf = basicResults(pipe, x_train, y_train, x_test, y_test, params, 'boosted', dataset)
    # plot_learning_curve(clf, dataset + ' boosted', x, y,
    #                     ylim=(0.0, 1.01), cv=5, n_jobs=4, scoring=scorer)
    # plt.savefig('./graphs/' + dataset + '-boost.png')
    plot_timing_curve(clf, x, y, 'boost', dataset)
    plt.savefig('./graphs/' + dataset + '-boost-timing.png')
    plot_iteration_curve(clf, x_train, y_train, x_test, y_test, params, 'boosted', dataset)
    plt.savefig('./graphs/' + dataset + '-boost-iteration.png')
    conf = confusion_matrix(y_test, clf.predict(x_test))
    conf = conf.astype('float') / conf.sum(axis=1)[:, np.newaxis]
    print('Confusion matrix:')
    print(conf)
    np.savetxt('./output/Boosted_{}_confusion.csv'.format(dataset), conf, delimiter=',', fmt='%.2f')

if __name__ == '__main__':
    run_boost(read_wine(), 'wine', dtparams={
        'alpha': -100,
        'class_weight': 'balanced',
        'criterion': 'entropy',
        'min_samples_split': 2
    })
    run_boost(read_gestures(), 'gestures', dtparams={
        'alpha': 0.0001,
        'class_weight': 'balanced',
        'criterion': 'gini',
        'min_samples_split': 4
    })
