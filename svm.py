import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from helpers import basicResults, scorer
from parse_data import read_gestures, read_wine
from plot import plot_learning_curve, plot_timing_curve, plot_iteration_curve

iter_adjust = {
    'SVM__max_iter': [2**x for x in range(12)] +
                     [2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000],
}
params = {
    'SVM__C': np.arange(.5, 1.0, .1),
}


def run_svm_linear(data, dataset):
    x, y, pipeline = data
    pipe = Pipeline([
        *pipeline,
        ('SVM', svm.LinearSVC(class_weight='balanced', dual=False)),
    ])
    print('Splitting data SVM Linear -- ' + dataset)
    x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y)
    print('Computing hyperparameters SVM Linear -- ' + dataset)
    clf = basicResults(pipe, x_train, y_train, x_test,
                       y_test, params, 'SVM-Linear', dataset)

    plot_timing_curve(clf, x, y, 'linear svm', dataset)
    plt.savefig('./graphs/' + dataset + '-svm-linear-timing.png')
    plot_iteration_curve(clf, x_train, y_train, x_test,
                         y_test, iter_adjust, 'linear svm', dataset)
    plt.savefig('./graphs/' + dataset + '-svm-linear-iteration.png')


def run_svm_rbf(data, dataset):
    x, y, pipeline = data
    pipe = Pipeline([
        *pipeline,
        ('SVM', svm.SVC(class_weight='balanced')),
    ])
    print('Splitting data SVM RBF -- ' + dataset)
    x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y)
    print('Computing hyperparameters SVM RBF -- ' + dataset)
    clf = basicResults(pipe, x_train, y_train, x_test,
                       y_test, params, 'SVM-RBF', dataset)

    plot_timing_curve(clf, x, y, 'rbf svm', dataset)
    plt.savefig('./graphs/' + dataset + '-svm-rbf-timing.png')
    plot_iteration_curve(clf, x_train, y_train, x_test,
                         y_test, iter_adjust, 'rbf svm', dataset)
    plt.savefig('./graphs/' + dataset + '-svm-rbf-iteration.png')


if __name__ == '__main__':
    wine = read_wine()
    gestures = read_gestures()
    run_svm_linear(wine, 'wine')
    run_svm_linear(gestures, 'gestures')
    run_svm_rbf(wine, 'wine')
    run_svm_rbf(gestures, 'gestures')
