#!python
# -*- coding: utf-8 -*-

"""Thermal comfort predictor.
Python version used: 3.6.8
Styling guide: PEP 8 -- Style Guide for Python Code
    (https://www.python.org/dev/peps/pep-0008/) and
    PEP 257 -- Docstring Conventions
    (https://www.python.org/dev/peps/pep-0257/)
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os

import numpy

import pandas as pd
from sklearn.datasets import make_regression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Module metadata dunders
__author__ = "Rob Garcia"
__copyright__ = "Copyright 2019-2020, Rob Garcia"
__email__ = "rgarcia@rgprogramming.com"
__license__ = "MIT"

CSV_COLUMN_NAMES = ["air_speed", "rel_humid",
                    "meta_rate", "cloth_lvl", "oper_temp", "sens_desc"]
CLASSES = ["Cold", "Cool", "Slightly Cool",
           "Neutral", "Slightly Warm", "Warm", "Hot"]
FEATURES = ["air_speed", "rel_humid", "meta_rate", "cloth_lvl", "oper_temp"]

dataset = pd.read_csv("thermal_comfort.csv", names=CSV_COLUMN_NAMES, header=0)
"""
print("Dataset shape:", dataset.shape)
print("Data check (first 20 rows:):")
print(dataset.head(20))
print("Dataset feature descriptions:")
print(dataset.describe())
print("Dataset class distribution:")
print(dataset.groupby("sens_desc").size())
"""
# Split-out validation dataset
array = dataset.values
X = array[:, 0:5]
y = array[:, 5]
X_train, X_validation, Y_train, Y_validation = train_test_split(
    X, y, test_size=0.20, random_state=1)

# Spot Check Algorithms
models = []
models.append(("LR", LogisticRegression(
    solver="liblinear", multi_class="ovr")))
models.append(("LDA", LinearDiscriminantAnalysis()))
models.append(("KNN", KNeighborsClassifier()))
models.append(("CART", DecisionTreeClassifier()))
models.append(("NB", GaussianNB()))
models.append(("SVM", SVC(gamma="auto")))
# evaluate each model in turn
results = []
names = []
print("Models and accuracy estimations:")
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(
        model, X_train, Y_train, cv=kfold, scoring="accuracy")
    results.append(cv_results)
    names.append(name)
    print("%s: %f (%f)" % (name, cv_results.mean(), cv_results.std()))

print("\nUsing Decision Tree Classifier...")
observation = [[0.1, 76.0, 1.0, 0.6, 28.0]]
print("Sample data:", observation)
clf = DecisionTreeClassifier(criterion='gini', random_state=0)
model = clf.fit(X, y)
print("Prediction:", model.predict(observation))
# print(model.predict_proba(observation))

print("\nUsing Support Vector Machine...")
observation = [[0.1, 50.0, 1.0, 0.5, 23.0]]
print("Sample data:", observation)
clf = SVC(kernel='linear', C = 1.0)
clf.fit(X, y)
print("Prediction:", clf.predict(observation))
