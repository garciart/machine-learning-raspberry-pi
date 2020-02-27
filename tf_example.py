#!python
# -*- coding: utf-8 -*-

"""Thermal comfort predictor using Keras and TensorFlow.
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
import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import Bunch

# Module metadata dunders
__author__ = "Rob Garcia"
__copyright__ = "Copyright 2019-2020, Rob Garcia"
__email__ = "rgarcia@rgprogramming.com"
__license__ = "MIT"


def tensorflow_classification_test(file_name, label_names, unlabeled_x):
    """Classification function using Keras and TensorFlow

    :param file_name: The name of the csv file with the data.
    :type file_name: str
    :param label_names: The list of labels.
    :type label_names: list
    :param unlabeled_x: Unlabeled data to be classified.
    :type unlabeled_x: tuple
    :param expected_y: The expected results of classifying the unlabeled data.
    :type expected_y: list
    :return: Numpy array of predicted labels.
    :rtype: list
    """
    print("TensorFlow version: {}".format(tf.__version__))
    # Limit decimal places to three and do not use scientific notation
    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)

    # Import and parse the dataset
    # Broken down for tutorial. Can be optimized into fewer lines.
    column_titles = []
    feature_titles = []
    # label_title = ""
    num_of_inputs = 0
    num_of_outputs = len(label_names)
    feature_values = []
    label_values = []

    with open(file_name) as csv_file:
        dataframe = pd.read_csv(csv_file, header=0)
        column_titles = dataframe.columns.values
        feature_titles = column_titles[:-1]
        # label_title = column_titles[-1]
        num_of_inputs = len(feature_titles)
        values = dataframe.values
        feature_values = values[:, 0:num_of_inputs]
        label_values = values[:, num_of_inputs]

    dataset = Bunch(data=feature_values, target=label_values)

    x_values = dataset.data
    # Break down target array into multidimensional list of label values
    y_raw = dataset.target.reshape(-1, 1)

    # One hot encode the labels
    encoder = OneHotEncoder(sparse=False)
    y_values = encoder.fit_transform(y_raw)

    # Split the data into training and test sets using a 80:20 ratio.
    train_x, test_x, train_y, test_y = train_test_split(x_values, y_values, test_size=0.20)

    # Build the model
    model = Sequential()
    model.add(Dense(10, input_shape=(num_of_inputs,),
                    activation="relu", name="fc1"))
    model.add(Dense(10, activation="relu", name="fc2"))
    model.add(Dense(num_of_outputs, activation="softmax", name="output"))

    """
    Select an optimizer and loss function.
    If your loss and accuracy do not change, try changing the optimizer or loss function;

    Keras optimizers (https://keras.io/optimizers/)
    SGD - Stochastic gradient descent optimizer.
    RMSprop - RMSProp optimizer.
    Adagrad - Adagrad optimizer.
    Adadelta - Adadelta optimizer.
    Adam - Adam optimizer.
    Adamax - Adamax optimizer from Adam papers Section 7.
    Nadam - Nesterov Adam optimizer.

    Keras losses (https://keras.io/losses/)
    mean_squared_error
    mean_absolute_error
    mean_absolute_percentage_error
    mean_squared_logarithmic_error
    squared_hinge
    hinge
    categorical_hinge
    logcosh
    huber_loss
    categorical_crossentropy
    sparse_categorical_crossentropy
    binary_crossentropy
    kullback_leibler_divergence
    poisson
    cosine_proximity
    is_categorical_crossentropy
    """

    optimizer = "SGD"
    loss = "mean_squared_error"
    model.compile(optimizer, loss, metrics=["accuracy"])

    print("Neural Network Model Summary:")
    print(model.summary())

    # Train the model
    model.fit(train_x, train_y, verbose=2, batch_size=32, epochs=256)

    # Test the model
    results = model.evaluate(test_x, test_y)

    print("Final test set loss: {:4f}".format(results[0]))
    print("Final test set accuracy: {:4f}".format(results[1]))

    return model


def main():
    """Application entry point."""
    start_time = time.time()
    print("TensorFlow Classification Test using Keras.\n")
    # Sample data to be evaluated

    # Iris test
    file_name = "iris.csv"
    label_names = ["Iris setosa", "Iris versicolor", "Iris virginica"]
    unlabeled_x = np.array([
        [5.9, 3.0, 4.2, 1.5],
        [5.1, 3.3, 1.7, 0.5],
        [6.9, 3.1, 5.4, 2.1]
    ])
    expected_y = ["Iris versicolor", "Iris setosa", "Iris virginica"]
    """
    file_name = "thermal_comfort.csv"
    labels = ["Cold", "Cool", "Slightly Cool",
              "Neutral", "Slightly Warm", "Warm", "Hot"]
    unlabeled_x = np.array([
        [1013.25, 0.1, 50.0, 1.0, 0.61, 23.0],
        [1013.25, 0.1, 60.0, 1.0, 0.61, 26.0],
        [1013.25, 0.1, 76.0, 1.0, 0.61, 28.0]
    ])
    expected_y = ["Slightly Cool", "Neutral", "Slightly Warm"]
    """
    model = tensorflow_classification_test(
        file_name, label_names, unlabeled_x)
    # Make predictions for the unlabeled data
    predictions = model.predict_classes(unlabeled_x, verbose=1)
    print()
    for i in range(len(unlabeled_x)):
        print("X={}, Predicted: {} ({}), Expected {}".format(
            unlabeled_x[i], label_names[predictions[i]], predictions[i], expected_y[i]))
    print("Elapsed time: {} seconds.".format((time.time() - start_time)))
    print("Job complete. Have an excellent day.")


if __name__ == "__main__":
    main()
