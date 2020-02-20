#!python
# -*- coding: utf-8 -*-

"""Thermal comfort predictor using Scikit Learn machine learning.
Python version used: 3.6.8
See requirements.txt for additional dependencies
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
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier

# Module metadata dunders
__author__ = "Rob Garcia"
__copyright__ = "Copyright 2019-2020, Rob Garcia"
__email__ = "rgarcia@rgprogramming.com"
__license__ = "MIT"

# Read CSV data into dataset
CSV_COLUMN_NAMES = ["air_speed", "rel_humid",
                    "meta_rate", "cloth_lvl", "oper_temp", "sens_desc"]
dataset = pd.read_csv("thermal_comfort.csv", names=CSV_COLUMN_NAMES, header=0)

# Name classes and features
CLASSES = ["Cold", "Cool", "Slightly Cool",
           "Neutral", "Slightly Warm", "Warm", "Hot"]
FEATURES = ["air_speed", "rel_humid", "meta_rate", "cloth_lvl", "oper_temp"]

"""
# Utility functions to verify dataset
print("Dataset shape:", dataset.shape)
print("Data check (first 20 rows:):")
print(dataset.head(20))
print("Dataset feature descriptions:")
print(dataset.describe())
print("Dataset class distribution:")
print(dataset.groupby("sens_desc").size())
"""

# Split the dataset. Use 80% for training and 20% for testing
# (as determined by test_size). Remove random_state for production.
array = dataset.values
X = array[:, 0:5]
y = array[:, 5]
X_train, X_validation, Y_train, Y_validation = train_test_split(
    X, y, test_size=0.20, random_state=1)

# Get confidence for select algorithms
# (https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html)
models = []
models.append(("Logistic Regression", LogisticRegression(
    solver="liblinear", multi_class="ovr")))
models.append(("Linear Support Vector Classification (LinearSVC)", LinearSVC(dual=False)))
models.append(("Stochastic Gradient Descent (SGD)", SGDClassifier()))
models.append(("k-Nearest Neighbors Classifier (k-NN)", KNeighborsClassifier()))
models.append(("Support Vector Classification (SVC)", SVC(kernel='linear', C = 1.0)))
models.append(("Gaussian Naive Bayes (GaussianNB)", GaussianNB()))
models.append(("Random Forest Classifier", RandomForestClassifier()))
models.append(("Extra Trees Classifier", ExtraTreesClassifier()))
models.append(("Decision Tree Classifier", DecisionTreeClassifier()))
models.append(("AdaBoost Classifier", AdaBoostClassifier()))
models.append(("Gradient Boosting Classifier", GradientBoostingClassifier()))
models.append(("Linear Discriminant Analysis (LDA)", LinearDiscriminantAnalysis()))

# Evaluate each model
scores = []
names = []
print("Predicted accuracy:")
for name, model in models:
    clf = model
    clf.fit(X_train, Y_train)
    prediction = clf.predict(X_validation)
    scores.append(accuracy_score(Y_validation,prediction))
    names.append(name)

# Get and sort the results
results = []
for s, n in zip(scores, names):
    results.append((s, n))

results.sort(reverse=True)

for score, name in results:
    print("{0:.2f}%: {1}".format(score, name))

"""
print("Mean, Standard Deviation, and Model Name:")
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_score = cross_val_score(
        model, X_train, Y_train, cv=kfold, scoring="accuracy")
    scores.append(cv_score)
    names.append(name)

# Get and sort the results
results = []

for s, n in zip(scores, names):
    results.append((s.mean(), s.std(), n))

results.sort(reverse=True)

for mean, std, name in results:
    print("{0:.6f} ({1:.6f}): {2}".format(mean, std, name))
"""

print()

data = [[[0.1, 50.0, 1.0, 0.5, 23.0]], [[0.1, 60.0, 1.0, 0.6, 26.0]], [[0.1, 76.0, 1.0, 0.6, 28.0]]]
data_class = [[["Slightly Cool"]], [["Neutral"]], [["Slightly Warm"]]] 
for d, dc in zip(data, data_class):
    print("Sample data: {} (Predict: {})".format(d, dc))

print("\nUsing Logistic Regression...")
clf = LogisticRegression(solver="liblinear", multi_class="ovr")
model = clf.fit(X, y)
for d in data:
    print("Prediction:", model.predict(d))
    # print(model.predict_proba(d))


print("\nUsing Linear Discriminant Analysis (LDA)...")
clf = LinearDiscriminantAnalysis()
model = clf.fit(X, y)
for d in data:
    print("Prediction:", model.predict(d))

print("\nUsing Decision Tree Classifier...")
clf = DecisionTreeClassifier()
model = clf.fit(X, y)
for d in data:
    print("Prediction:", model.predict(d))

print("\nUsing Support Vector Machine...")
clf = SVC(kernel='linear', C = 1.0)
model = clf.fit(X, y)
for d in data:
    print("Prediction:", model.predict(d))
