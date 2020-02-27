#!/usr/bin/python3
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

import time

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

# Module metadata dunders
__author__ = "Rob Garcia"
__copyright__ = "Copyright 2019-2020, Rob Garcia"
__email__ = "rgarcia@rgprogramming.com"
__license__ = "MIT"


def scikit_learn_classification_test(file_name, label_names):
    """Train, test, and run different classification estimators, selected from
    https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html
    :param sample_data: The data to be analyzed.
    :type sample_data: tuple
    """
    # Limit decimal places to three and do not use scientific notation
    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)

    # Import and parse the training dataset
    # Broken down for tutorial. Can be optimized into fewer lines.
    column_titles = []
    feature_titles = []
    # label_title = ""
    num_of_inputs = 0
    # num_of_outputs = len(label_names)
    feature_values = []
    label_values = []

    with open(file_name) as csv_file:
        dataframe = pd.read_csv(csv_file, header=0)
        column_titles = dataframe.columns.values
        feature_titles = column_titles[:-1]
        label_title = column_titles[-1]
        num_of_inputs = len(feature_titles)
        values = dataframe.values
        feature_values = values[:, 0:num_of_inputs]
        label_values = values[:, num_of_inputs]

    """Utility functions to verify dataframe
    print("Dataframe shape:", dataframe.shape)
    print("Data check (first 20 rows:):")
    print(dataframe.head(20))
    print("Dataframe feature descriptions:")
    print(dataframe.describe().round(3))
    print("Dataframe class distribution:")
    print(dataframe.groupby("sens_desc").size())
    """

    # Split the dataframe, then use 80% for training and 20% for testing
    # (as determined by test_size). Remove random_state for production.
    x_train, x_validation, y_train, y_validation = train_test_split(
        feature_values, label_values, test_size=0.20, random_state=1)

    # Create tuple of estimators
    estimators = []
    estimators.append(("Logistic Regression", LogisticRegression(
        solver="liblinear", multi_class="ovr")))
    estimators.append(
        ("Linear Support Vector Classification (LinearSVC)", LinearSVC(dual=False)))
    estimators.append(("Stochastic Gradient Descent (SGD)", SGDClassifier()))
    estimators.append(
        ("k-Nearest Neighbors Classifier (k-NN)", KNeighborsClassifier()))
    estimators.append(("Support Vector Classification (SVC)",
                       SVC(kernel="linear", C=1.0)))
    estimators.append(("Gaussian Naive Bayes (GaussianNB)", GaussianNB()))
    estimators.append(("Random Forest Classifier", RandomForestClassifier()))
    estimators.append(("Extra Trees Classifier", ExtraTreesClassifier()))
    estimators.append(("Decision Tree Classifier", DecisionTreeClassifier()))
    estimators.append(("AdaBoost Classifier", AdaBoostClassifier()))
    estimators.append(("Gradient Boosting Classifier",
                       GradientBoostingClassifier()))
    estimators.append(("Linear Discriminant Analysis (LDA)",
                       LinearDiscriminantAnalysis()))

    # Evaluate the accuracy of each estimator
    results = []
    for name, classifier in estimators:
        classifier.fit(x_train, y_train)
        prediction = classifier.predict(x_validation)
        score = accuracy_score(y_validation, prediction)
        results.append((name, classifier, score))

    # Get and sort the results
    results.sort(key=lambda r: r[2], reverse=True)

    # Select best classifer
    selected_classifier = results[0]
    return selected_classifier


def main():
    """Application entry point."""
    start_time = time.time()
    print("Environmental monitoring using scikit-learn.\n")

    # Step 1: Prepare and train the model
    file_name = "thermal_comfort.csv"
    label_names = ["Cold", "Cool", "Slightly Cool",
                   "Neutral", "Slightly Warm", "Warm", "Hot"]
    selected_classifier = scikit_learn_classification_test(
        file_name, label_names)
    classifier_name = selected_classifier[0]
    classifier_model = selected_classifier[1]
    classifier_score = selected_classifier[2]
    print("{0} selected (Test accuracy = {1:.1f}%).\n".format(
        classifier_name, classifier_score * 100))

    # Step 2: Collect the data

    # Step 3: Process the data

    print("Data to be evaluated:")
    unlabeled_x = []
    unlabeled_x.append(
        ([[1013.25, 0.1, 50.0, 1.0, 0.61, 23.0]], "Slightly Cool"))
    unlabeled_x.append(([[1013.25, 0.1, 60.0, 1.0, 0.61, 26.0]], "Neutral"))
    unlabeled_x.append(
        ([[1013.25, 0.1, 76.0, 1.0, 0.61, 28.0]], "Slightly Warm"))
    # expected_y = ["Slightly Cool", "Neutral", "Slightly Warm"]
    for i, (data, expected_label) in enumerate(unlabeled_x, start=1):
        print("Sample #{}: {} | Prediction: {}".format(i, data, expected_label))
    print()

    # Run samples using all classifiers in accuracy order
    print("Running samples using {0}...".format(selected_classifier[0]))

    for j, (data, expected_label) in enumerate(unlabeled_x, start=1):
        print("Sample #{}: Prediction: {} (expected {})".format(
            j, label_names[int(classifier_model.predict(data))], expected_label))

    print()
    print("Elapsed time: {} seconds.".format((time.time() - start_time)))
    print("Job complete. Have an excellent day.")


if __name__ == "__main__":
    main()
