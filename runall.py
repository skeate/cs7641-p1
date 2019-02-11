from multiprocessing import Process
from dt import run_dt
from ann import run_ann
from bdt import run_boost
from knn import run_knn
from svm import run_svm_linear, run_svm_rbf
from parse_data import read_gestures, read_wine

def run(data, dataset):
    # print('running decision tree for ' + dataset)
    # run_dt(data, dataset)
    print('running neural network for ' + dataset)
    run_ann(data, dataset)
    print('running boosted dt for ' + dataset)
    run_boost(data, dataset)
    print('running knn for ' + dataset)
    run_knn(data, dataset)
    print('running svm (linear) for ' + dataset)
    run_svm_linear(data, dataset)
    print('running svm (rbf) for ' + dataset)
    run_svm_rbf(data, dataset)

if __name__ == '__main__':
    run(read_gestures(), 'gestures')
    run(read_wine(), 'wine')
