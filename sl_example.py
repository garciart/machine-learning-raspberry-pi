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

import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import (LogisticRegression, SGDClassifier)
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import (LinearSVC, SVC)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier, RandomForestClassifier)

# Module metadata dunders
__author__ = "Rob Garcia"
__copyright__ = "Copyright 2019-2020, Rob Garcia"
__email__ = "rgarcia@rgprogramming.com"
__license__ = "MIT"


def scikit_learn_classification_test():
    """
    Train, test, and run different classification models, selected from
    https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html
    """
    # Read CSV data into dataset
    csv_column_names = ["air_speed", "rel_humid",
                        "meta_rate", "cloth_lvl", "oper_temp", "sens_desc"]
    dataset = pd.read_csv("thermal_comfort.csv",
                          names=csv_column_names, header=0)

    # Name classes and features
    # classes = ["Cold", "Cool", "Slightly Cool",
    #            "Neutral", "Slightly Warm", "Warm", "Hot"]
    # features = ["air_speed", "rel_humid",
    #             "meta_rate", "cloth_lvl", "oper_temp"]

    """ Utility functions to verify dataset
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
    x_train, x_validation, y_train, y_validation = train_test_split(
        X, y, test_size=0.20, random_state=1)

    # Get confidence for select algorithms
    models = []
    models.append(("Logistic Regression", LogisticRegression(
        solver="liblinear", multi_class="ovr")))
    models.append(
        ("Linear Support Vector Classification (LinearSVC)", LinearSVC(dual=False)))
    models.append(("Stochastic Gradient Descent (SGD)", SGDClassifier()))
    models.append(
        ("k-Nearest Neighbors Classifier (k-NN)", KNeighborsClassifier()))
    models.append(("Support Vector Classification (SVC)",
                   SVC(kernel='linear', C=1.0)))
    models.append(("Gaussian Naive Bayes (GaussianNB)", GaussianNB()))
    models.append(("Random Forest Classifier", RandomForestClassifier()))
    models.append(("Extra Trees Classifier", ExtraTreesClassifier()))
    models.append(("Decision Tree Classifier", DecisionTreeClassifier()))
    models.append(("AdaBoost Classifier", AdaBoostClassifier()))
    models.append(("Gradient Boosting Classifier",
                   GradientBoostingClassifier()))
    models.append(("Linear Discriminant Analysis (LDA)",
                   LinearDiscriminantAnalysis()))

    # Evaluate each model
    results = []
    print("Test set accuracy against training set:")
    for name, clf in models:
        clf.fit(x_train, y_train)
        prediction = clf.predict(x_validation)
        score = accuracy_score(y_validation, prediction)
        results.append((name, clf, score))

    # Get and sort the results
    results.sort(key=lambda r: r[2], reverse=True)

    for name, clf, score in results:
        print("{0:.2f}%: {1}".format(score, name))

    print()

    # Show sample data
    sample_data = []
    sample_data.append(([[0.1, 50.0, 1.0, 0.5, 23.0]], "Slightly Cool"))
    sample_data.append(([[0.1, 60.0, 1.0, 0.6, 26.0]], "Neutral"))
    sample_data.append(([[0.1, 76.0, 1.0, 0.6, 28.0]], "Slightly Warm"))

    for i, (data, expected) in enumerate(sample_data):
        print("Sample #{}: {} = {}".format(i, data, expected))

    print()

    # Run samples against the dataset using the classifiers in accuracy order
    for name, clf, score in results:
        print("Running samples against the dataset using {0}\n(test score was {1:.2f}%)...".format(
            name, score))
        model = clf.fit(X, y)
        for j, (data, expected) in enumerate(sample_data):
            print("Sample #{}: Prediction: {} (should be {})".format(
                j + 1, model.predict(data), expected))
        print()


def main():
    """Application entry point."""
    print("scikit-learn Classification Test.\n")
    scikit_learn_classification_test()


if __name__ == "__main__":
    main()
