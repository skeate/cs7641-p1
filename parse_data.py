"""
Utility functions for reading/fixing up datasets
"""

import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from NLTKPreprocessor import NLTKPreprocessor


def read_gestures():
    """Reads in gesture dataset"""
    data0 = np.genfromtxt('./data/gesture/0.csv', delimiter=',')
    data1 = np.genfromtxt('./data/gesture/1.csv', delimiter=',')
    data2 = np.genfromtxt('./data/gesture/2.csv', delimiter=',')
    data3 = np.genfromtxt('./data/gesture/3.csv', delimiter=',')
    data = np.concatenate((data0, data1, data2, data3))
    x = data[:, :-1]
    y = data[:, -1]
    pipeline = [
        ('Scale', StandardScaler())
    ]
    return (x, y, pipeline)


def read_wine():
  """reads in wine dataset"""
  whites = np.genfromtxt('./data/wine/winequality-white.csv', delimiter=';')[1:]
  reds = np.genfromtxt('./data/wine/winequality-red.csv', delimiter=';')[1:]
  data = np.r_[
      np.c_[whites, np.zeros(whites.shape[0])],
      np.c_[reds, np.ones(reds.shape[0])],
  ]
  x = data[:, :-1]
  y = data[:, -1]
  pipeline = [
      ('Scale', StandardScaler())
  ]
  return (x, y, pipeline)


def read_sarcasm():
    """Reads in sarcasm dataset"""
    fp = open('./data/sarcasm/Sarcasm_Headlines_Dataset.json')
    sarcasm = json.load(fp)
    fp.close()
    x = [s['headline'] for s in sarcasm]
    y = np.array([s['is_sarcastic'] for s in sarcasm])
    pipeline = [
        ('preprocessor', NLTKPreprocessor()),
        ('vectorizer', TfidfVectorizer(
            tokenizer=lambda x: x, preprocessor=None, lowercase=False)),
    ]
    return (x, y, pipeline)
