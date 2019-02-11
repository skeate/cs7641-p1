import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from helpers import basicResults, scorer
from parse_data import read_gestures, read_wine
from plot import plot_learning_curve, plot_timing_curve, plot_iteration_curve

params = {
    'KNN__metric': ['euclidean', 'manhattan', 'minkowski'],
    'KNN__n_neighbors': range(3, 10),
}

def run_knn(data, dataset):
    x, y, pipeline = data
    pipe = Pipeline([
        *pipeline,
        ('KNN', neighbors.KNeighborsClassifier()),
    ])
    print('Splitting data ' + dataset)
    x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y)
    print('Calculating hyperparameters ' + dataset)
    clf = basicResults(pipe, x_train, y_train, x_test, y_test, params, 'KNN', dataset)
    conf = confusion_matrix(y_test, clf.predict(x_test))
    conf = conf.astype('float') / conf.sum(axis=1)[:, np.newaxis]
    print('Confusion matrix:')
    print(conf)
    np.savetxt('./output/KNN_{}_confusion.csv'.format(dataset), conf, delimiter=',', fmt='%.2f')


if __name__ == '__main__':
    run_knn(read_wine(), 'wine')
    run_knn(read_gestures(), 'gestures')
