#!python
# -*- coding: utf-8 -*-

"""Thermal comfort predictor using Scikit Learn machine learning.
#!/usr/bin/python3
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


def scikit_learn_classification_test(*sample_data):
    """Train, test, and run different classification estimators, selected from
    https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html
    """
    # Limit decimal places to three and do not use scientific notation
    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)
    # Read CSV dataset into a dataframe
    dataframe = pd.read_csv("thermal_comfort.csv")
    # Get feature_names from column headers: ["atmo_pres", "air_speed",
    #     "rel_humid", "meta_rate", "cloth_lvl", "oper_temp", "sens_desc"]
    column_names = dataframe.columns.values
    feature_names = column_names[:-1]
    label_name = column_names[-1]
    # Assign names to the labels/classes
    class_names = ["Cold", "Cool", "Slightly Cool",
                   "Neutral", "Slightly Warm", "Warm", "Hot"]

    """Utility functions to verify dataframe"""
    print("Dataframe shape:", dataframe.shape)
    print("Data check (first 20 rows:):")
    print(dataframe.head(20))
    print("Dataframe feature descriptions:")
    print(dataframe.describe().round(3))
    print("Dataframe class distribution:")
    print(dataframe.groupby("sens_desc").size())
    """ """

    # Split the dataframe, then use 80% for training and 20% for testing
    # (as determined by test_size). Remove random_state for production.
    array = dataframe.values
    feature_values = array[:, 0:len(feature_names)]
    label_values = array[:, len(feature_names)]
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
    print("Test set accuracy against training set:")
    for name, classifier in estimators:
        classifier.fit(x_train, y_train)
        prediction = classifier.predict(x_validation)
        score = accuracy_score(y_validation, prediction)
        results.append((name, classifier, score))

    # Get and sort the results
    results.sort(key=lambda r: r[2], reverse=True)

    for name, classifier, score in results:
        print("{0:.2f}%: {1}".format(score, name))
        
    print()
    
    print("Data to be evaluated:")
    for i, (data, expected_label) in enumerate(sample_data, start=1):
        print("Sample #{}: {} = {}".format(i, data, expected_label))
    print()

    # Run samples using all classifiers in accuracy order
    for name, classifier, score in results:
        print("Running samples using {0}\n(test score was {1:.2f}%)...".format(
            name, score))
        model = classifier.fit(feature_values, label_values)
        for j, (data, expected_label) in enumerate(sample_data, start=1):
            print("Sample #{}: Prediction: {} (expected {})".format(
                j, class_names[int(model.predict(data))], expected_label))
        print()


def main():
    """Application entry point."""
    start_time = time.time()
    print("scikit-learn Classification Test.")
    # Sample data to be evaluated
    sample_data = []
    sample_data.append(([[1013.25, 0.1, 50.0, 1.0, 0.6, 23.0]], "Slightly Cool"))
    sample_data.append(([[1013.25, 0.1, 60.0, 1.0, 0.6, 26.0]], "Neutral"))
    sample_data.append(([[1013.25, 0.1, 76.0, 1.0, 0.6, 28.0]], "Slightly Warm"))
    print()
    scikit_learn_classification_test(*sample_data)
    print("Elapsed time: {} seconds.".format((time.time() - start_time)))
    print("Job complete. Have an excellent day.")


if __name__ == "__main__":
    main()
